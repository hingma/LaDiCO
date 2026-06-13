"""
latent_difusco.py

Phase 2 of the LDM-TSP pipeline: Gaussian diffusion in the VQ-VAE node-latent space.

Training (x0-parameterisation):
  1. Encode ground-truth TSP solution through the frozen VQ-VAE → z_0 (N, d)
  2. Sample timestep t; forward diffusion: z_t = √ᾱ_t · z_0 + √(1-ᾱ_t) · ε
  3. Denoise with AdaGNN (conditioned on node coords + t) → z̃_0 prediction
  4. Loss = MSE(z̃_0, z_0)

Inference:
  1. Sample z_T ~ N(0, I)
  2. Iterative DDPM / DDIM denoising via AdaGNN
  3. Decode z̃_0 → edge probabilities via the frozen VQ-VAE decoder
  4. Tour extraction: merge_tours + batched 2-opt
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from torch_geometric.data import DataLoader as GraphDataLoader
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum, mean as sparse_mean, max as sparse_max

from co_datasets.tsp_graph_dataset import TSPGraphDataset
from models.edge_vqvae import TSPEdgeVQVAE
from models.gnn_encoder import PositionEmbeddingSine, ScalarEmbeddingSine1D
from models.nn import timestep_embedding, zero_module, linear
from utils.diffusion_schedulers import GaussianDiffusion, InferenceSchedule
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive Layer Normalisation (DiT-style)
# ──────────────────────────────────────────────────────────────────────────────

class AdaLayerNorm(nn.Module):
    """
    y = (1 + γ) · LN(x) + β,   where  [γ, β] = Linear(cond).

    Zero-initialised projection so the layer starts as the identity and training
    stabilises from the first step.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (N, dim), cond: (N, cond_dim) → (N, dim)"""
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return (1.0 + gamma) * self.norm(x) + beta


# ──────────────────────────────────────────────────────────────────────────────
# Gated GCN layer with AdaLN node conditioning
# ──────────────────────────────────────────────────────────────────────────────

class AdaGNNLayer(nn.Module):
    """
    Gated graph convolution layer with Adaptive Layer Norm for node features.

    Node update:  h ← h_in + AdaLN(ReLU(U·h + Σ_j σ(e_ij) ⊙ V·h_j),  cond)
    Edge update:  e ← e_in + LN(ReLU(A·h_j + B·h_i + C·e))

    The A/B convention (A on target node, B on source) matches GNNLayer in
    gnn_encoder.py so the two can be swapped without changing results.
    Sparse-only (no dense / batched mode needed here).
    """

    def __init__(self, hidden_dim: int, cond_dim: int, aggregation: str = "sum"):
        super().__init__()
        self.aggregation = aggregation

        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)
        self.A = nn.Linear(hidden_dim, hidden_dim)
        self.B = nn.Linear(hidden_dim, hidden_dim)
        self.C = nn.Linear(hidden_dim, hidden_dim)

        self.ada_norm_h = AdaLayerNorm(hidden_dim, cond_dim)
        self.norm_e     = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h:          torch.Tensor,   # (N, H)
        e:          torch.Tensor,   # (E, H)
        adj:        SparseTensor,   # (N×N) unit-weight adjacency
        edge_index: torch.Tensor,   # (2, E)  [0]=source, [1]=target
        cond:       torch.Tensor,   # (N, cond_dim) per-node conditioning
    ):
        h_in, e_in = h, e
        src, tgt = edge_index[0], edge_index[1]

        Uh = self.U(h)           # (N, H)
        Vh = self.V(h[tgt])      # (E, H)  target-node features per directed edge
        Ah = self.A(h)           # (N, H)
        Bh = self.B(h)           # (N, H)
        Ce = self.C(e)           # (E, H)

        # Edge update: A(target) + B(source) + C(edge)
        e_new = Ah[tgt] + Bh[src] + Ce      # (E, H)
        gates = torch.sigmoid(e_new)         # (E, H)

        # Gated neighbourhood aggregation
        sp = SparseTensor(
            row=src, col=tgt,
            value=gates * Vh,
            sparse_sizes=(h.shape[0], h.shape[0]),
        )
        if self.aggregation == "mean":
            h_agg = sparse_mean(sp, dim=1)
        elif self.aggregation == "max":
            h_agg = sparse_max(sp, dim=1)
        else:
            h_agg = sparse_sum(sp, dim=1)    # (N, H)

        # Node update: AdaLN conditioned on (time + coords)
        h_new = F.relu(Uh + h_agg)                          # (N, H)
        h     = h_in + self.ada_norm_h(h_new, cond)         # (N, H)

        # Edge update: standard LayerNorm
        e_new = F.relu(e_new)
        e     = e_in + self.norm_e(e_new)                   # (E, H)

        return h, e


# ──────────────────────────────────────────────────────────────────────────────
# Latent-space GNN denoiser
# ──────────────────────────────────────────────────────────────────────────────

class LatentDiffuscoDenoiser(nn.Module):
    """
    GNN denoiser that operates in the VQ-VAE node-latent space.

    Uses x0-parameterisation: predicts the clean latent z̃_0 directly rather
    than the noise.  Each AdaGNNLayer is conditioned on a per-node signal that
    combines the timestep embedding with the sinusoidal coordinate embedding,
    giving spatially-varying and time-aware layer normalisation throughout.

    Input:
        x:          (N, 2)   node coordinates
        z_t:        (N, d)   noisy VQ latent at diffusion step t
        timesteps:  (N,)     float timestep per node (identical within one graph)
        edge_index: (2, E)   sparse K-NN edge indices

    Output:
        z̃_0:       (N, d)   predicted clean latent
    """

    def __init__(
        self,
        n_layers:    int,
        hidden_dim:  int,
        latent_dim:  int,
        aggregation: str = "sum",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        cond_dim = hidden_dim // 2

        # ── Node input ────────────────────────────────────────────────────────
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed   = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.pos_proj    = nn.Linear(hidden_dim, hidden_dim)

        # ── Edge input ────────────────────────────────────────────────────────
        self.dist_embed = ScalarEmbeddingSine1D(hidden_dim, normalize=False)
        self.edge_proj  = nn.Linear(hidden_dim, hidden_dim)

        # ── Per-node conditioning: timestep + coordinate ──────────────────────
        self.time_mlp = nn.Sequential(
            linear(hidden_dim, cond_dim),
            nn.SiLU(),
            linear(cond_dim, cond_dim),
        )
        # Reuse pos_embed; project its H-dim output to cond_dim
        self.coord_cond_proj = nn.Linear(hidden_dim, cond_dim)

        # ── AdaGNN backbone ───────────────────────────────────────────────────
        self.layers = nn.ModuleList([
            AdaGNNLayer(hidden_dim, cond_dim, aggregation)
            for _ in range(n_layers)
        ])

        # ── Output: z̃_0 prediction — zero-init for training stability ─────────
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = zero_module(nn.Linear(hidden_dim, latent_dim))

    def forward(
        self,
        x:          torch.Tensor,   # (N, 2)
        z_t:        torch.Tensor,   # (N, d)
        timesteps:  torch.Tensor,   # (N,) float
        edge_index: torch.Tensor,   # (2, E)
    ) -> torch.Tensor:
        edge_index = edge_index.long()
        src, tgt = edge_index[0], edge_index[1]

        # ── Node input features ───────────────────────────────────────────────
        pos_h = self.pos_embed(x.unsqueeze(0)).squeeze(0)          # (N, H)
        h = self.latent_proj(z_t) + self.pos_proj(pos_h)           # (N, H)

        # ── Edge input features (K-NN distances) ──────────────────────────────
        dists = (x[src] - x[tgt]).pow(2).sum(-1).sqrt()            # (E,)
        e = self.edge_proj(self.dist_embed(dists))                  # (E, H)

        # ── Per-node conditioning signal ───────────────────────────────────────
        t_emb = self.time_mlp(
            timestep_embedding(timesteps, self.hidden_dim))         # (N, cond_dim)
        c_emb = self.coord_cond_proj(pos_h)                        # (N, cond_dim)
        cond  = t_emb + c_emb                                       # (N, cond_dim)

        # ── GNN layers with AdaLN ──────────────────────────────────────────────
        adj = SparseTensor(
            row=src, col=tgt,
            value=torch.ones(edge_index.shape[1], device=x.device),
            sparse_sizes=(x.shape[0], x.shape[0]),
        )
        for layer in self.layers:
            h, e = layer(h, e, adj, edge_index, cond)

        # Aggregate last-layer edge activations into source nodes so the final
        # AdaGNNLayer's edge-norm parameters receive gradients (same fix as VQ-VAE encoder).
        e_agg = torch.zeros_like(h)
        e_agg.scatter_add_(0, src.unsqueeze(-1).expand_as(e), e)
        counts = src.bincount(minlength=h.shape[0]).float().clamp(min=1).unsqueeze(-1)
        h = h + e_agg / counts

        return self.out_proj(F.silu(self.out_norm(h)))              # (N, d)


# ──────────────────────────────────────────────────────────────────────────────
# Lightning module
# ──────────────────────────────────────────────────────────────────────────────

class LatentDifusco(pl.LightningModule):
    """
    Phase 2 LDM: Gaussian diffusion in the VQ-VAE node-latent space.

    The VQ-VAE is loaded from a Phase-1 checkpoint and permanently frozen;
    only the AdaGNN denoiser (LatentDiffuscoDenoiser) is trained.

    At inference time the denoised latent is decoded back to edge-probability
    heatmaps via the frozen VQ-VAE decoder, then tours are extracted with
    merge_tours and refined by batched 2-opt.
    """

    def __init__(self, args):
        super().__init__()
        self.args          = args
        self.sparse_factor = args.sparse_factor

        # ── Frozen VQ-VAE ─────────────────────────────────────────────────────
        self.vqvae = TSPEdgeVQVAE(
            n_enc_layers    = args.n_enc_layers,
            n_dec_layers    = args.n_dec_layers,
            hidden_dim      = args.vqvae_hidden_dim,
            latent_dim      = args.latent_dim,
            num_codes       = args.num_codes,
            commitment_cost = args.vq_commitment_cost,
            aggregation     = args.aggregation,
        )
        self._load_frozen_vqvae(args.vqvae_checkpoint)

        # ── Trainable denoiser ────────────────────────────────────────────────
        self.denoiser = LatentDiffuscoDenoiser(
            n_layers    = args.n_layers,
            hidden_dim  = args.hidden_dim,
            latent_dim  = args.latent_dim,
            aggregation = args.aggregation,
        )

        # ── Gaussian diffusion schedule ───────────────────────────────────────
        self.diffusion = GaussianDiffusion(
            T        = args.diffusion_steps,
            schedule = args.diffusion_schedule,
        )

        # ── Datasets ──────────────────────────────────────────────────────────
        self.train_dataset      = self._make_dataset(args.training_split)
        self.validation_dataset = TSPGraphDataset(
            data_file     = os.path.join(args.storage_path, args.validation_split),
            sparse_factor = args.sparse_factor,
        )
        self.test_dataset = TSPGraphDataset(
            data_file     = os.path.join(args.storage_path, args.test_split),
            sparse_factor = args.sparse_factor,
        )

    # ── Dataset helper ────────────────────────────────────────────────────────

    def _make_dataset(self, split_spec: str):
        files    = [f.strip() for f in split_spec.split(",")]
        datasets = [
            TSPGraphDataset(
                data_file     = os.path.join(self.args.storage_path, f),
                sparse_factor = self.args.sparse_factor,
            )
            for f in files
        ]
        return datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)

    # ── VQ-VAE checkpoint loading ─────────────────────────────────────────────

    def _load_frozen_vqvae(self, ckpt_path: str):
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"VQ-VAE checkpoint not found: {ckpt_path!r}\n"
                "Run Phase 1 (train_vqvae.py) first and pass --vqvae_checkpoint."
            )
        saved = torch.load(ckpt_path, map_location="cpu")
        # Phase-1 saves {'state_dict': ..., 'args': ...}; fall back for raw dicts
        state = saved.get("state_dict", saved)
        self.vqvae.load_state_dict(state)
        for p in self.vqvae.parameters():
            p.requires_grad_(False)
        self.vqvae.eval()
        rank_zero_info(f"Loaded frozen VQ-VAE from {ckpt_path}")

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        _, graph_data, point_indicator, _, _ = batch

        x          = graph_data.x.float()         # (B·N, 2)
        edge_index = graph_data.edge_index         # (2, B·N·K)
        edge_attr  = graph_data.edge_attr.float()  # (B·N·K, 1)

        batch_size      = point_indicator.shape[0]
        nodes_per_graph = int((point_indicator.sum() // batch_size).item())

        # Step 1 — encode solution to z_0 with the frozen VQ-VAE
        with torch.no_grad():
            z_e = self.vqvae.encode(x, edge_attr, edge_index)    # (B·N, d)
            _, z_0, _, _ = self.vqvae.quantize(z_e)              # (B·N, d)

        # Step 2 — sample t, add forward noise:  z_t = √ᾱ_t · z_0 + √(1-ᾱ_t) · ε
        t_batch    = np.random.randint(1, self.diffusion.T + 1, batch_size).astype(int)
        t_per_node = np.repeat(t_batch, nodes_per_graph)          # (B·N,)

        atbar   = torch.from_numpy(
            self.diffusion.alphabar[t_per_node]).float().to(z_0.device).unsqueeze(-1)
        epsilon = torch.randn_like(z_0)
        z_t     = torch.sqrt(atbar) * z_0 + torch.sqrt(1.0 - atbar) * epsilon

        # Step 3 — predict z̃_0 (x0-parameterisation)
        t_tensor = torch.from_numpy(t_per_node).float().to(z_0.device)   # (B·N,)
        z_0_pred = self.denoiser(x, z_t, t_tensor, edge_index)           # (B·N, d)

        # Step 4 — MSE on the clean latent
        loss = F.mse_loss(z_0_pred, z_0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ── Gaussian posterior (copied from COMetaModel; takes predicted ε) ───────

    def gaussian_posterior(self, target_t, t, pred, xt):
        """DDPM or DDIM posterior step given predicted noise ε."""
        diffusion = self.diffusion
        if target_t is None:
            target_t = t - 1
        else:
            target_t = torch.from_numpy(target_t).view(1)

        atbar        = diffusion.alphabar[t]
        atbar_target = diffusion.alphabar[target_t]

        if getattr(self.args, "inference_trick", None) is None or t <= 1:
            at         = diffusion.alpha[t]
            z          = torch.randn_like(xt)
            atbar_prev = diffusion.alphabar[t - 1]
            beta_tilde = diffusion.beta[t - 1] * (1 - atbar_prev) / (1 - atbar)
            xt_target  = (
                (1 / np.sqrt(at)).item() *
                (xt - ((1 - at) / np.sqrt(1 - atbar)).item() * pred)
                + np.sqrt(beta_tilde).item() * z
            )
        elif self.args.inference_trick == "ddim":
            xt_target = (
                np.sqrt(atbar_target / atbar).item() *
                (xt - np.sqrt(1 - atbar).item() * pred)
                + np.sqrt(1 - atbar_target).item() * pred
            )
        else:
            raise ValueError(f"Unknown inference_trick: {self.args.inference_trick!r}")
        return xt_target

    def duplicate_edge_index(self, edge_index, num_nodes, device):
        """Tile edge_index for parallel sampling, adding per-copy node offsets."""
        ei     = edge_index.reshape(2, 1, -1)
        indent = (
            torch.arange(self.args.parallel_sampling).view(1, -1, 1).to(device)
            * num_nodes
        )
        return (ei + indent).reshape(2, -1)

    # ── Denoising step ────────────────────────────────────────────────────────

    def _denoise_step(self, points, z_t, t, device, edge_index, target_t=None):
        """
        One reverse-diffusion step.

        The denoiser produces z̃_0 (x0 prediction); we convert to the implied
        noise ε̃ = (z_t − √ᾱ_t · z̃_0) / √(1−ᾱ_t), then call gaussian_posterior
        so both DDPM and DDIM code paths are reused unchanged.
        """
        with torch.no_grad():
            t_tensor   = torch.from_numpy(t).view(1)                      # (1,)
            N          = points.shape[0]
            t_per_node = t_tensor.float().expand(N).to(device)            # (N,)

            z_0_pred = self.denoiser(
                points.float().to(device),
                z_t.to(device),
                t_per_node,
                edge_index.long().to(device),
            )                                                              # (N, d)

            # Convert x0 prediction → implied noise for gaussian_posterior
            atbar_t   = float(self.diffusion.alphabar[t])
            eps_impl  = (
                (z_t.to(device) - np.sqrt(atbar_t) * z_0_pred) /
                np.sqrt(max(1.0 - atbar_t, 1e-8))
            )

        return self.gaussian_posterior(target_t, t_tensor, eps_impl, z_t.to(device))

    # ── Test / Validation ─────────────────────────────────────────────────────

    def test_step(self, batch, batch_idx, split="test"):
        _, graph_data, _, _, gt_tour = batch
        device = gt_tour.device

        points     = graph_data.x.reshape(-1, 2)           # (N, 2)
        edge_index = graph_data.edge_index.reshape(2, -1)  # (2, E)

        np_points     = points.cpu().numpy()
        np_edge_index = edge_index.cpu().numpy()
        np_gt_tour    = gt_tour.cpu().numpy().reshape(-1)
        N             = points.shape[0]

        if self.args.parallel_sampling > 1:
            points     = points.repeat(self.args.parallel_sampling, 1)
            edge_index = self.duplicate_edge_index(edge_index, N, device)

        stacked_tours = []
        ns, merge_iterations = 0, 0

        for _ in range(self.args.sequential_sampling):
            z_t = torch.randn(points.shape[0], self.args.latent_dim, device=device)

            time_schedule = InferenceSchedule(
                inference_schedule = self.args.inference_schedule,
                T                  = self.diffusion.T,
                inference_T        = self.args.inference_diffusion_steps,
            )

            for i in range(self.args.inference_diffusion_steps):
                t1, t2 = time_schedule(i)
                z_t = self._denoise_step(
                    points, z_t,
                    np.array([t1]).astype(int), device, edge_index,
                    target_t=np.array([t2]).astype(int),
                )

            # Decode latent → per-edge probabilities
            with torch.no_grad():
                edge_probs_nk = self.vqvae.decode(
                    z_t, points.float(), edge_index, self.sparse_factor)  # (P·N, K)
            adj_mat = np.clip(edge_probs_nk.view(-1).float().cpu().detach().numpy(), 1e-6, 1.0)

            tours, merge_iterations = merge_tours(
                adj_mat, np_points, np_edge_index,
                sparse_graph=True,
                parallel_sampling=self.args.parallel_sampling,
            )
            solved_tours, ns = batched_two_opt_torch(
                np_points.astype("float64"),
                np.array(tours).astype("int64"),
                max_iterations=self.args.two_opt_iterations,
                device=device,
            )
            stacked_tours.append(solved_tours)

        solved_tours = np.concatenate(stacked_tours, axis=0)

        tsp_solver = TSPEvaluator(np_points)
        gt_cost    = tsp_solver.evaluate(np_gt_tour)
        total_s    = self.args.parallel_sampling * self.args.sequential_sampling
        all_costs  = [tsp_solver.evaluate(solved_tours[i]) for i in range(total_s)]
        best_cost  = float(np.min(all_costs))

        metrics = {
            f"{split}/gt_cost":          gt_cost,
            f"{split}/2opt_iterations":  ns,
            f"{split}/merge_iterations": merge_iterations,
        }
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True)
        self.log(f"{split}/solved_cost", best_cost, prog_bar=True, on_epoch=True, sync_dist=True)
        return metrics

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, split="val")

    def test_epoch_end(self, outputs):
        unmerged = {}
        for m in outputs:
            for k, v in m.items():
                unmerged.setdefault(k, []).append(v)
        self.logger.log_metrics(
            {k: float(np.mean(v)) for k, v in unmerged.items()},
            step=self.global_step,
        )

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        n_params = sum(p.numel() for p in self.denoiser.parameters())
        rank_zero_info(f"Latent denoiser trainable parameters: {n_params:,}")
        optimizer = torch.optim.AdamW(
            self.denoiser.parameters(),
            lr           = self.args.learning_rate,
            weight_decay = self.args.weight_decay,
        )
        if self.args.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.num_epochs, eta_min=1e-6)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        return optimizer

    # ── Data loaders ──────────────────────────────────────────────────────────

    def train_dataloader(self):
        return GraphDataLoader(
            self.train_dataset,
            batch_size         = self.args.batch_size,
            shuffle            = True,
            num_workers        = self.args.num_workers,
            pin_memory         = True,
            persistent_workers = (self.args.num_workers > 0),
            drop_last          = True,
        )

    def val_dataloader(self):
        n = min(self.args.validation_examples, len(self.validation_dataset))
        return GraphDataLoader(
            torch.utils.data.Subset(self.validation_dataset, range(n)),
            batch_size  = 1,
            shuffle     = False,
            num_workers = self.args.num_workers,
        )

    def test_dataloader(self):
        return GraphDataLoader(self.test_dataset, batch_size=1, shuffle=False)
