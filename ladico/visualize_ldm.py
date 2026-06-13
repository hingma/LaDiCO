"""
visualize_ldm.py

Visualizes the full LaDiCO / LDM-TSP inference pipeline for a single test instance.

Layout
------
Row 0  –  n_vis_steps panels: intermediate K-NN edge-probability heatmaps decoded
           directly from z_t at evenly-spaced reverse-diffusion timesteps, showing
           how structure emerges from pure noise.
Row 1  –  4-5 summary panels:
           (a) Dense N×N heatmap (symmetrised mean)
           (b) Final decoder output as a K-NN graph (edge color/width = probability)
           (c) Greedy tour from merge_tours (before refinement)
           (d) 2-opt refined tour
           (e) Ground-truth tour  [optional, --show_gt]

Usage
-----
python visualize_ldm.py \\
    --ldm_checkpoint   checkpoints/ldm_tsp50/last.ckpt \\
    --vqvae_checkpoint checkpoints/vqvae_tsp50/best_vqvae.pt \\
    --test_file        data/tsp/tsp50_test_concorde.txt \\
    --instance_idx 0 --n_vis_steps 6 --output_file results/vis.png --show_gt

Cross-scale example (model trained on TSP-50, visualised on TSP-100):
    ... --eval_sparse_factor 10 --two_opt_iterations 500
"""

import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # headless-safe; overridden by --show below
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from co_datasets.tsp_graph_dataset import TSPGraphDataset
from models.edge_vqvae import TSPEdgeVQVAE
from latent_difusco import LatentDiffuscoDenoiser
from utils.diffusion_schedulers import GaussianDiffusion, InferenceSchedule
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours
from torch_geometric.data import DataLoader as GraphDataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────

def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Model loading  (same logic as evaluate_ldm.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_models(args):
    vqvae = TSPEdgeVQVAE(
        n_enc_layers    = args.n_enc_layers,
        n_dec_layers    = args.n_dec_layers,
        hidden_dim      = args.vqvae_hidden_dim,
        latent_dim      = args.latent_dim,
        num_codes       = args.num_codes,
        commitment_cost = args.vq_commitment_cost,
        aggregation     = args.aggregation,
    )
    if not os.path.isfile(args.vqvae_checkpoint):
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {args.vqvae_checkpoint}")
    saved = torch.load(args.vqvae_checkpoint, map_location="cpu")
    vqvae.load_state_dict(saved.get("state_dict", saved))
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)

    denoiser = LatentDiffuscoDenoiser(
        n_layers    = args.n_layers,
        hidden_dim  = args.hidden_dim,
        latent_dim  = args.latent_dim,
        aggregation = args.aggregation,
    )
    if not os.path.isfile(args.ldm_checkpoint):
        raise FileNotFoundError(f"LDM checkpoint not found: {args.ldm_checkpoint}")
    pl_ckpt  = torch.load(args.ldm_checkpoint, map_location="cpu")
    pl_state = pl_ckpt.get("state_dict", pl_ckpt)
    den_state = {
        k[len("denoiser."):]: v
        for k, v in pl_state.items()
        if k.startswith("denoiser.")
    }
    denoiser.load_state_dict(den_state)
    denoiser.eval()

    diffusion = GaussianDiffusion(T=args.diffusion_steps, schedule=args.diffusion_schedule)
    return vqvae, denoiser, diffusion


# ──────────────────────────────────────────────────────────────────────────────
# Single reverse-diffusion step  (x0-parameterisation → DDPM/DDIM)
# ──────────────────────────────────────────────────────────────────────────────

def _denoise_step(denoiser, diffusion, args, x, z_t, t_arr, edge_index, device):
    t_val = int(t_arr[0])
    with torch.no_grad():
        N        = x.shape[0]
        t_tensor = torch.full((N,), float(t_val), device=device)
        z_0_pred = denoiser(x.float(), z_t, t_tensor, edge_index.long())

        atbar_t  = float(diffusion.alphabar[t_arr])
        eps_impl = (z_t - np.sqrt(atbar_t) * z_0_pred) / np.sqrt(max(1.0 - atbar_t, 1e-8))

    from torch import from_numpy
    t_tensor_1 = from_numpy(t_arr).view(1)

    atbar = diffusion.alphabar[t_tensor_1]
    at    = diffusion.alpha[t_tensor_1]

    inference_trick = getattr(args, "inference_trick", None)
    if inference_trick is None or t_val <= 1:
        atbar_prev = diffusion.alphabar[t_tensor_1 - 1]
        beta_tilde = diffusion.beta[t_val - 1] * (1 - atbar_prev) / (1 - atbar)
        z          = torch.randn_like(z_t) if t_val > 1 else torch.zeros_like(z_t)
        z_t_new = (
            (1 / np.sqrt(at)).item() *
            (z_t - ((1 - at) / np.sqrt(1 - atbar)).item() * eps_impl)
            + np.sqrt(beta_tilde).item() * z
        )
    else:
        target_t_    = from_numpy(np.array([t_val - 1]).astype(int)).view(1)
        atbar_target = diffusion.alphabar[target_t_]
        z_t_new = (
            np.sqrt(atbar_target / atbar).item() *
            (z_t - np.sqrt(1 - atbar).item() * eps_impl)
            + np.sqrt(1 - atbar_target).item() * eps_impl
        )
    return z_t_new


# ──────────────────────────────────────────────────────────────────────────────
# Inference with intermediate-step capture
# ──────────────────────────────────────────────────────────────────────────────

def run_with_capture(vqvae, denoiser, diffusion, args, x, edge_index, device,
                     n_vis_steps):
    """
    Run reverse diffusion; decode z_t at n_vis_steps evenly-spaced checkpoints.

    Returns
    -------
    snapshots     : list of (t_value, probs_nk: np.ndarray (N, K))
    adj_sparse    : (N*K,) flat edge-probability array for merge_tours
    adj_dense     : (N, N) symmetrised mean heatmap
    np_edge_index : (2, E)
    """
    N = x.shape[0]
    K = args.eval_sparse_factor

    x_dev  = x.float().to(device)
    ei_dev = edge_index.long().to(device)

    schedule = InferenceSchedule(
        inference_schedule = args.inference_schedule,
        T                  = diffusion.T,
        inference_T        = args.inference_diffusion_steps,
    )
    T_steps = args.inference_diffusion_steps

    # Pick capture indices; always include the final step
    n_snap = max(1, min(n_vis_steps, T_steps))
    if n_snap >= T_steps:
        capture_at = set(range(T_steps))
    else:
        capture_at = {
            int(round(i * (T_steps - 1) / (n_snap - 1)))
            for i in range(n_snap)
        }
    capture_at.add(T_steps - 1)
    capture_order = sorted(capture_at)

    z_t = torch.randn(N, args.latent_dim, device=device)
    snapshots = []

    for i in tqdm(range(T_steps), desc="Reverse diffusion", leave=False):
        t1, _t2 = schedule(i)
        z_t = _denoise_step(
            denoiser, diffusion, args, x_dev, z_t,
            np.array([t1]).astype(int), ei_dev, device,
        )
        if i in capture_order:
            with torch.no_grad():
                probs = vqvae.decode(z_t, x_dev, ei_dev, K)   # (N, K)
            snapshots.append((t1, probs.cpu().numpy()))

    # Final state
    np_ei = edge_index.cpu().numpy()
    with torch.no_grad():
        probs_final = vqvae.decode(z_t, x_dev, ei_dev, K)     # (N, K)

    adj_sparse = probs_final.view(-1).float().cpu().numpy()

    dense = np.zeros((N, N))
    dense[np_ei[0], np_ei[1]] = adj_sparse
    adj_dense = (dense + dense.T) * 0.5

    return snapshots, adj_sparse, adj_dense, np_ei


# ──────────────────────────────────────────────────────────────────────────────
# Drawing utilities
# ──────────────────────────────────────────────────────────────────────────────

_NODE_S  = 18
_NODE_C  = "#222222"
_MARGIN  = 0.05


def _draw_knn_graph(ax, np_points, edge_index_np, probs_flat,
                    title="", cmap="plasma"):
    """
    K-NN graph: nodes at their 2-D coordinates; edges colored and weighted
    by their predicted probability (high-prob = bright/thick).
    """
    src, tgt = edge_index_np[0], edge_index_np[1]
    segs = np.stack([np_points[src], np_points[tgt]], axis=1)   # (E, 2, 2)
    p    = np.clip(probs_flat, 0.0, 1.0)

    lc = LineCollection(
        segs, cmap=cmap,
        norm=mcolors.Normalize(vmin=0.0, vmax=1.0),
        linewidths=0.2 + p * 1.8,
        zorder=1,
    )
    lc.set_array(p)
    ax.add_collection(lc)
    ax.scatter(np_points[:, 0], np_points[:, 1],
               s=_NODE_S, c=_NODE_C, zorder=3)
    _finish_ax(ax, title)


def _draw_tour(ax, np_points, tour, title="", cost=None, color="#1a7abf"):
    """Closed tour drawn as a connected path."""
    pts  = np_points[tour]                                       # (N+1, 2)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)                 # (N, 2, 2)
    lc   = LineCollection(segs, colors=color, linewidths=0.9, zorder=1)
    ax.add_collection(lc)
    ax.scatter(np_points[:, 0], np_points[:, 1],
               s=_NODE_S, c=_NODE_C, zorder=3)
    label = title if cost is None else f"{title}\n{cost:.4f}"
    _finish_ax(ax, label)


def _draw_dense_heatmap(ax, adj_dense, title=""):
    """Symmetric N×N adjacency heatmap."""
    im = ax.imshow(
        adj_dense, cmap="hot", vmin=0.0, vmax=1.0,
        interpolation="nearest", aspect="equal",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=7, pad=2)
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)


def _finish_ax(ax, title=""):
    ax.set_xlim(-_MARGIN, 1 + _MARGIN)
    ax.set_ylim(-_MARGIN, 1 + _MARGIN)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=7, pad=2)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    if args.eval_sparse_factor <= 0:
        args.eval_sparse_factor = args.sparse_factor

    device = _device()
    print(f"Device: {device}")

    vqvae, denoiser, diffusion = load_models(args)
    vqvae    = vqvae.to(device)
    denoiser = denoiser.to(device)

    dataset = TSPGraphDataset(
        data_file     = args.test_file,
        sparse_factor = args.eval_sparse_factor,
    )
    loader = GraphDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # ── Fetch the requested instance ──────────────────────────────────────────
    target_batch = None
    for idx, batch in enumerate(loader):
        if idx == args.instance_idx:
            target_batch = batch
            break
    if target_batch is None:
        raise IndexError(
            f"Instance {args.instance_idx} not found "
            f"(dataset has {len(dataset)} entries)."
        )

    _, graph_data, _, _, gt_tour = target_batch
    x          = graph_data.x.reshape(-1, 2)
    edge_index = graph_data.edge_index.reshape(2, -1)
    np_points  = x.cpu().numpy()
    np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
    N          = np_points.shape[0]
    K          = args.eval_sparse_factor

    evaluator = TSPEvaluator(np_points)
    gt_cost   = evaluator.evaluate(np_gt_tour)

    # ── Inference with step captures ──────────────────────────────────────────
    print(f"Instance {args.instance_idx}: N={N}, K={K}, "
          f"steps={args.inference_diffusion_steps}, vis_steps={args.n_vis_steps}")
    snapshots, adj_sparse, adj_dense, np_ei = run_with_capture(
        vqvae, denoiser, diffusion, args, x, edge_index,
        device, args.n_vis_steps,
    )

    # ── Greedy tour (merge_tours) ──────────────────────────────────────────────
    tours_greedy, _ = merge_tours(
        adj_sparse, np_points, np_ei,
        sparse_graph      = True,
        parallel_sampling = 1,
    )
    greedy_tour = tours_greedy[0]
    greedy_cost = evaluator.evaluate(greedy_tour)

    # ── 2-opt refinement ───────────────────────────────────────────────────────
    solved, _ = batched_two_opt_torch(
        np_points.astype("float64"),
        np.array([greedy_tour]).astype("int64"),
        max_iterations = args.two_opt_iterations,
        device         = device,
    )
    twoopt_tour = solved[0].tolist()
    twoopt_cost = evaluator.evaluate(twoopt_tour)
    gap_pct     = (twoopt_cost - gt_cost) / gt_cost * 100.0

    print(f"GT={gt_cost:.4f}  |  Greedy={greedy_cost:.4f}  |  "
          f"2-opt={twoopt_cost:.4f}  (gap {gap_pct:.2f}%)")

    # ── Close-loop GT tour for drawing ────────────────────────────────────────
    gt_tour_closed = np_gt_tour.tolist()
    if gt_tour_closed[-1] != gt_tour_closed[0]:
        gt_tour_closed.append(gt_tour_closed[0])

    # ─────────────────────────────────────────────────────────────────────────
    # Build figure
    #
    # Row 0  : diffusion-step snapshots  (n_snap columns)
    # Row 1  : dense heatmap | final K-NN graph | greedy tour | 2-opt [| GT]
    # ─────────────────────────────────────────────────────────────────────────
    n_snap  = len(snapshots)
    n_bot   = 5 if args.show_gt else 4
    n_cols  = max(n_snap, n_bot)

    col_w   = 2.5
    fig_w   = max(col_w * n_cols, 10)
    fig     = plt.figure(figsize=(fig_w, 5.5))
    gs      = GridSpec(2, n_cols, figure=fig,
                       hspace=0.40, wspace=0.08,
                       top=0.90, bottom=0.04,
                       left=0.02, right=0.98)

    fig.suptitle(
        f"LaDiCO  ·  Instance {args.instance_idx}  (N={N}, K={K})   "
        f"GT={gt_cost:.4f}   Greedy={greedy_cost:.4f}   "
        f"2-opt={twoopt_cost:.4f}   gap={gap_pct:.2f}%",
        fontsize=8.5,
    )

    # ── Row 0: reverse-diffusion snapshots ────────────────────────────────────
    for col, (t_val, probs_nk) in enumerate(snapshots):
        ax = fig.add_subplot(gs[0, col])
        label = f"t = {t_val}" if col < n_snap - 1 else f"t = {t_val}  (final)"
        _draw_knn_graph(ax, np_points, np_ei, probs_nk.reshape(-1),
                        title=label, cmap="plasma")

    # Fill remaining top-row slots if n_snap < n_cols
    for col in range(n_snap, n_cols):
        ax = fig.add_subplot(gs[0, col])
        ax.axis("off")

    # ── Row 1: summary panels ─────────────────────────────────────────────────

    # (a) Dense N×N heatmap
    ax_h = fig.add_subplot(gs[1, 0])
    _draw_dense_heatmap(ax_h, adj_dense, title="Dense heatmap\n(symmetrised)")

    # (b) Final decoder output — K-NN graph
    ax_knn = fig.add_subplot(gs[1, 1])
    _draw_knn_graph(ax_knn, np_points, np_ei, adj_sparse,
                    title="Decoder output\n(K-NN graph)", cmap="plasma")

    # (c) Greedy tour
    ax_g = fig.add_subplot(gs[1, 2])
    _draw_tour(ax_g, np_points, greedy_tour,
               title="Greedy (merge_tours)", cost=greedy_cost, color="#e05a22")

    # (d) 2-opt refined tour
    ax_t = fig.add_subplot(gs[1, 3])
    _draw_tour(ax_t, np_points, twoopt_tour,
               title="2-opt refined", cost=twoopt_cost, color="#1a7abf")

    # (e) GT tour (optional)
    if args.show_gt and n_bot == 5:
        ax_gt = fig.add_subplot(gs[1, 4])
        _draw_tour(ax_gt, np_points, gt_tour_closed,
                   title="Ground truth", cost=gt_cost, color="#2cb85c")

    # Fill remaining bottom slots
    start_fill = n_bot
    for col in range(start_fill, n_cols):
        ax = fig.add_subplot(gs[1, col])
        ax.axis("off")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = args.output_file
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {out_path}")

    if args.show:
        matplotlib.use("TkAgg")   # switch to interactive backend
        plt.show()

    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def arg_parser():
    p = ArgumentParser(description="Visualise LDM-TSP inference pipeline.")

    # Checkpoints
    p.add_argument("--ldm_checkpoint",     required=True)
    p.add_argument("--vqvae_checkpoint",   required=True)

    # Test data
    p.add_argument("--test_file",          required=True)
    p.add_argument("--instance_idx",       type=int, default=0,
                   help="0-based index of the test instance to visualise.")
    p.add_argument("--eval_sparse_factor", type=int, default=-1)

    # VQ-VAE architecture (must match Phase-1 checkpoint)
    p.add_argument("--vqvae_hidden_dim",   type=int,   default=128)
    p.add_argument("--n_enc_layers",       type=int,   default=4)
    p.add_argument("--n_dec_layers",       type=int,   default=4)
    p.add_argument("--latent_dim",         type=int,   default=8)
    p.add_argument("--num_codes",          type=int,   default=512)
    p.add_argument("--vq_commitment_cost", type=float, default=0.25)
    p.add_argument("--sparse_factor",      type=int,   default=10)
    p.add_argument("--aggregation",        type=str,   default="sum")

    # Denoiser architecture
    p.add_argument("--n_layers",           type=int,   default=6)
    p.add_argument("--hidden_dim",         type=int,   default=256)

    # Diffusion
    p.add_argument("--diffusion_steps",           type=int,   default=1000)
    p.add_argument("--diffusion_schedule",         type=str,   default="linear")
    p.add_argument("--inference_diffusion_steps",  type=int,   default=50)
    p.add_argument("--inference_schedule",         type=str,   default="linear")
    p.add_argument("--inference_trick",            type=str,   default=None,
                   choices=[None, "ddim"])

    # Solver
    p.add_argument("--two_opt_iterations", type=int, default=1000)

    # Visualisation options
    p.add_argument("--n_vis_steps",  type=int,   default=6,
                   help="Number of intermediate diffusion steps to show (top row).")
    p.add_argument("--show_gt",      action="store_true",
                   help="Add a ground-truth tour panel in the bottom row.")
    p.add_argument("--output_file",  type=str,   default="results/ldm_vis.png",
                   help="Path for the saved figure.")
    p.add_argument("--dpi",          type=int,   default=150)
    p.add_argument("--show",         action="store_true",
                   help="Also open an interactive window (requires a display).")

    return p.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
