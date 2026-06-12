"""
Edge VQ-VAE for TSP: learns discrete node-level latent codes over sparse K-NN graphs.

Architecture
------------
Encoder  : Anisotropic (gated) multi-layer GNN  →  (N, d) continuous node latents
VQ       : Per-node nearest-codebook lookup with straight-through estimator
Decoder  : Mirror GNN  →  (N, K) continuous edge-probability heatmap  [Sigmoid output]
Loss     : BCE  +  VQ-commitment  +  α · BHH-energy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

from models.gnn_encoder import GNNLayer, PositionEmbeddingSine, ScalarEmbeddingSine1D
from models.nn import zero_module


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _adj(edge_index: torch.Tensor, num_nodes: int) -> SparseTensor:
    """Build a unit-weight SparseTensor adjacency from a (2, E) edge_index."""
    ei = edge_index.long()
    return SparseTensor(
        row=ei[0], col=ei[1],
        value=torch.ones(ei.shape[1], device=ei.device),
        sparse_sizes=(num_nodes, num_nodes),
    )


def _edge_dists(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Return Euclidean distances for each directed edge.  (E,)"""
    src, tgt = edge_index[0].long(), edge_index[1].long()
    return (x[src] - x[tgt]).pow(2).sum(-1).sqrt()


# ──────────────────────────────────────────────────────────────────────────────
# Vector Quantizer (van den Oord et al. 2017)
# ──────────────────────────────────────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    """
    Straight-through vector quantizer.

    Maps a continuous (N, d) tensor to the nearest codebook vector per row,
    returning gradients to the encoder via the straight-through estimator.
    """

    def __init__(self, num_codes: int, latent_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_codes, latent_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: (N, d) continuous encoder output
        Returns:
            z_q_st : (N, d)  quantized, carries encoder gradients via STE
            z_q    : (N, d)  quantized, detached from the encoder graph
            indices: (N,)    codebook index for each node
            vq_loss: scalar  codebook + commitment loss
        """
        # (N, V)  squared L2 distances to every codebook entry
        d = (z_e.pow(2).sum(1, keepdim=True)
             - 2.0 * (z_e @ self.embedding.weight.t())
             + self.embedding.weight.pow(2).sum(1))

        indices = d.argmin(1)                          # (N,)
        z_q = self.embedding(indices)                  # (N, d)

        loss_codebook  = F.mse_loss(z_q, z_e.detach())
        loss_commit    = F.mse_loss(z_e, z_q.detach())
        vq_loss = loss_codebook + self.commitment_cost * loss_commit

        z_q_st = z_e + (z_q - z_e).detach()           # straight-through
        return z_q_st, z_q, indices, vq_loss


# ──────────────────────────────────────────────────────────────────────────────
# Shared GNN backbone (no time conditioning)
# ──────────────────────────────────────────────────────────────────────────────

class _GNNStack(nn.Module):
    """L-layer gated (anisotropic) GNN with residual connections."""

    def __init__(self, n_layers: int, hidden_dim: int, aggregation: str = "sum"):
        super().__init__()
        self.layers = nn.ModuleList([
            GNNLayer(hidden_dim, aggregation,
                     norm="layer", learn_norm=True, track_norm=False, gated=True)
            for _ in range(n_layers)
        ])
        # Zero-initialised output gate on edges (stabilises early training)
        self.edge_gates = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                zero_module(nn.Linear(hidden_dim, hidden_dim)),
            ) for _ in range(n_layers)
        ])

    def forward(self, h: torch.Tensor, e: torch.Tensor,
                edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h: (V, H)  node features
        e: (E, H)  edge features
        edge_index: (2, E)
        Returns updated h, e of the same shapes.
        """
        adj = _adj(edge_index, h.shape[0])
        for layer, gate in zip(self.layers, self.edge_gates):
            h_in, e_in = h, e
            h, e = layer(h, e, adj, mode="direct", edge_index=edge_index, sparse=True)
            h = h_in + h
            e = e_in + gate(e)
        return h, e


# ──────────────────────────────────────────────────────────────────────────────
# Encoder  –  (x, edge_labels) → (N, d)
# ──────────────────────────────────────────────────────────────────────────────

class TSPVQVAEEncoder(nn.Module):
    """
    Anisotropic GNN encoder.

    Embeds node coordinates and K-NN binary edge labels into a shared hidden
    space, runs L gated-GCN layers, and reads out node-level latents (N, d).
    Edge distances are included as a second edge feature to give the model
    geometric awareness alongside the solution label.
    """

    def __init__(self, n_layers: int, hidden_dim: int, latent_dim: int,
                 aggregation: str = "sum"):
        super().__init__()
        # 2-D sinusoidal embedding for node coordinates  →  (N, hidden_dim)
        self.pos_embed  = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.node_proj  = nn.Linear(hidden_dim, hidden_dim)

        # Sinusoidal embeddings for scalar edge features (label + distance)
        # Each maps a 1-D scalar to hidden_dim; we add them element-wise.
        self.label_embed = ScalarEmbeddingSine1D(hidden_dim, normalize=False)
        self.dist_embed  = ScalarEmbeddingSine1D(hidden_dim, normalize=False)
        self.edge_proj   = nn.Linear(hidden_dim, hidden_dim)

        self.gnn = _GNNStack(n_layers, hidden_dim, aggregation)

        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        x:          (V, 2)     node coordinates  (V = B·N for batched graphs)
        edge_attr:  (E, 1)     binary tour-membership labels  (E = V·K)
        edge_index: (2, E)     directed K-NN edge indices
        Returns z_e: (V, latent_dim)
        """
        edge_index = edge_index.long()
        dists = _edge_dists(x, edge_index)                            # (E,)

        # Node features from 2-D coordinates
        h = self.node_proj(
            self.pos_embed(x.unsqueeze(0)).squeeze(0))                # (V, H)

        # Edge features: sinusoidal(label) + sinusoidal(distance)
        e = self.edge_proj(
            self.label_embed(edge_attr.squeeze(-1).float()) +         # (E, H)
            self.dist_embed(dists)                                     # (E, H)
        )                                                              # (E, H)

        h, e = self.gnn(h, e, edge_index)

        # Aggregate last-layer edge activations back to their source nodes so
        # the encoder's edge-gate parameters (last layer) receive gradients.
        src = edge_index[0]                                            # (E,)
        e_agg = torch.zeros_like(h)                                    # (V, H)
        e_agg.scatter_add_(
            0, src.unsqueeze(-1).expand_as(e), e)
        counts = src.bincount(minlength=h.shape[0]).float().clamp(min=1).unsqueeze(-1)
        h = h + e_agg / counts                                         # (V, H)

        return self.out(h)                                             # (V, d)


# ──────────────────────────────────────────────────────────────────────────────
# Decoder  –  (z_q, x) → (N, K)  edge probabilities
# ──────────────────────────────────────────────────────────────────────────────

class TSPVQVAEDecoder(nn.Module):
    """
    Mirror-GNN decoder.

    Takes quantized node latents and node coordinates, runs L gated-GCN layers,
    and outputs per-edge probabilities via a Sigmoid readout on edge features.
    The edge probabilities are returned flat (E, 1); the caller reshapes to (N, K).
    """

    def __init__(self, n_layers: int, hidden_dim: int, latent_dim: int,
                 aggregation: str = "sum"):
        super().__init__()
        # Node features: quantized latent + coordinate residual
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed   = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.pos_proj    = nn.Linear(hidden_dim, hidden_dim)

        # Edge features: distance only (labels are what we predict)
        self.dist_embed = ScalarEmbeddingSine1D(hidden_dim, normalize=False)
        self.edge_proj  = nn.Linear(hidden_dim, hidden_dim)

        self.gnn = _GNNStack(n_layers, hidden_dim, aggregation)

        # Edge-level readout with Sigmoid
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_q: torch.Tensor, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        z_q:        (V, d)    quantized node latents
        x:          (V, 2)    node coordinates
        edge_index: (2, E)    directed K-NN edge indices
        Returns edge_probs: (E, 1)  ∈ [0, 1]
        """
        edge_index = edge_index.long()
        dists = _edge_dists(x, edge_index)                            # (E,)

        # Node features: latent + positional residual
        h = (self.latent_proj(z_q) +
             self.pos_proj(self.pos_embed(x.unsqueeze(0)).squeeze(0)))  # (V, H)

        # Edge features from geometry (no label info — predicting it)
        e = self.edge_proj(self.dist_embed(dists))                     # (E, H)

        h, e = self.gnn(h, e, edge_index)

        # Fuse last-layer node features into edge predictions so the decoder's
        # node-update parameters (U, V, norm_h in the last layer) receive gradients.
        src, tgt = edge_index[0], edge_index[1]
        e = e + h[src] + h[tgt]                                        # (E, H)

        return self.out(e)                                             # (E, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Full VQ-VAE
# ──────────────────────────────────────────────────────────────────────────────

class TSPEdgeVQVAE(nn.Module):
    """
    Full VQ-VAE over the K-NN sparse TSP edge graph.

    Encodes each node's neighbourhood into a discrete codebook index,
    then decodes back to a (N, K) soft tour-membership heatmap.

    Attributes:
        latent_dim   : d (per-node latent dimension)
        sparse_factor: K (number of nearest neighbours)
    """

    def __init__(self, n_enc_layers: int, n_dec_layers: int,
                 hidden_dim: int, latent_dim: int,
                 num_codes: int, commitment_cost: float = 0.25,
                 aggregation: str = "sum"):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = TSPVQVAEEncoder(
            n_enc_layers, hidden_dim, latent_dim, aggregation)
        self.vq = VectorQuantizer(num_codes, latent_dim, commitment_cost)
        self.decoder = TSPVQVAEDecoder(
            n_dec_layers, hidden_dim, latent_dim, aggregation)

    # ------------------------------------------------------------------
    def encode(self, x, edge_attr, edge_index):
        """(V, 2), (E, 1), (2, E) → z_e: (V, d)"""
        return self.encoder(x, edge_attr, edge_index)

    def quantize(self, z_e):
        """z_e: (V, d) → (z_q_st, z_q, indices, vq_loss)"""
        return self.vq(z_e)

    def decode(self, z_q, x, edge_index, sparse_factor: int):
        """Returns edge_probs_nk: (V, K) ∈ [0, 1]"""
        flat = self.decoder(z_q, x, edge_index)             # (E, 1)
        return flat.view(-1, sparse_factor)                  # (V, K)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor,
                edge_index: torch.Tensor, sparse_factor: int):
        """
        Args:
            x:            (V, 2)    node coordinates  [V = B·N]
            edge_attr:    (E, 1)    binary tour-membership labels  [E = V·K]
            edge_index:   (2, E)    K-NN edge indices
            sparse_factor: K

        Returns:
            edge_probs_nk : (V, K)   per-edge probabilities in node × neighbour layout
            z_q           : (V, d)   quantized latents (detached from encoder graph)
            indices       : (V,)     codebook indices
            vq_loss       : scalar   VQ + commitment loss
        """
        z_e = self.encode(x, edge_attr, edge_index)
        z_q_st, z_q, indices, vq_loss = self.quantize(z_e)
        edge_probs_nk = self.decode(z_q_st, x, edge_index, sparse_factor)
        return edge_probs_nk, z_q, indices, vq_loss


# ──────────────────────────────────────────────────────────────────────────────
# BHH energy regularizer
# ──────────────────────────────────────────────────────────────────────────────

def bhh_energy_loss(edge_probs_nk: torch.Tensor,
                    x: torch.Tensor,
                    edge_index: torch.Tensor,
                    sparse_factor: int,
                    nodes_per_graph: int) -> torch.Tensor:
    """
    Scale-invariant soft tour energy regularizer.

    Based on the Beardwood-Halton-Hammersley (BHH) theorem, which states that
    the optimal TSP tour length in a unit square scales as β√N (β ≈ 0.7124).
    Dividing by √N makes the energy comparable across problem sizes.

    Computes:  E_BHH = (1/√N) · Σ_{i,k} p_{i,k} · ‖x_i − x_{nn(i,k)}‖₂
    averaged over the batch.  Minimising this encourages the model to assign
    probability mass to geometrically short (near-optimal) edges.

    Args:
        edge_probs_nk  : (B·N, K)  soft edge probabilities from the decoder
        x              : (B·N, 2)  node coordinates
        edge_index     : (2, B·N·K) K-NN directed edge indices
        sparse_factor  : K
        nodes_per_graph: N  (assumes uniform batch; use point_indicator[0])

    Returns:
        Scalar BHH energy (lower = shorter expected tour).
    """
    dists   = _edge_dists(x, edge_index)                              # (B·N·K,)
    dists_nk = dists.view(-1, sparse_factor)                          # (B·N, K)

    batch_size = edge_probs_nk.shape[0] // nodes_per_graph
    weighted = (edge_probs_nk * dists_nk).view(
        batch_size, nodes_per_graph, sparse_factor)                   # (B, N, K)

    per_graph_energy = weighted.sum(dim=[1, 2]) / (nodes_per_graph ** 0.5)  # (B,)
    return per_graph_energy.mean()
