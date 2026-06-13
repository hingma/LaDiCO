"""
evaluate_ldm.py

Standalone evaluation script for the Phase-2 LDM-TSP model.

Runs the full inference pipeline (reverse diffusion → VQ-VAE decode → heatmap)
and feeds the resulting edge-probability heatmap into a choice of solvers.

Solvers
-------
greedy   – greedy edge-insertion (merge_tours) + batched 2-opt (built-in; fast)
save     – save dense numpy heatmaps to disk for offline tsp_mcts processing
lkh3     – call the LKH-3 binary with heatmap-guided candidate lists

Cross-scale evaluation
----------------------
The LDM architecture is fully O(N) and graph-based; it generalises to any
problem size at inference time.  To test a model trained on TSP-50 on TSP-500
instances, just pass the TSP-500 test file and set --sparse_factor to the
value used during training (controls the VQ-VAE decoder reshape) and optionally
set --eval_sparse_factor to a larger K for a denser K-NN graph at eval time.

Examples
--------
# greedy + 2-opt  (in-scale)
python ladico/evaluate_ldm.py \\
    --ldm_checkpoint   checkpoints/ldm_tsp50/last.ckpt \\
    --vqvae_checkpoint checkpoints/vqvae_tsp50/best_vqvae.pt \\
    --test_file        data/tsp/tsp50_test_concorde.txt

# cross-scale: model trained on TSP-50, tested on TSP-500
python ladico/evaluate_ldm.py \\
    --ldm_checkpoint   checkpoints/ldm_tsp50/last.ckpt \\
    --vqvae_checkpoint checkpoints/vqvae_tsp50/best_vqvae.pt \\
    --test_file        data/tsp/tsp500_test_concorde.txt \\
    --eval_sparse_factor 10 --two_opt_iterations 1000

# save heatmaps for offline MCTS  (compatible with tsp_mcts/solve-500.sh)
python ladico/evaluate_ldm.py ... --solver save --heatmap_dir results/heatmaps/

# LKH-3 guided by heatmap
python ladico/evaluate_ldm.py ... --solver lkh3 \\
    --lkh_binary LKH-3.0.6/LKH --lkh_runs 1 --lkh_candidates 20
"""

import os
import sys
import time
import tempfile
import shutil
import subprocess
from argparse import ArgumentParser
from typing import Optional

import numpy as np
import scipy.spatial
import torch
from torch_geometric.data import DataLoader as GraphDataLoader
from tqdm import tqdm

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, '..', 'difusco'))

from co_datasets.tsp_graph_dataset import TSPGraphDataset
from edge_vqvae import TSPEdgeVQVAE
from latent_difusco import LatentDiffuscoDenoiser
from utils.diffusion_schedulers import GaussianDiffusion, InferenceSchedule
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_models(args):
    """
    Load VQ-VAE and denoiser from checkpoints.

    VQ-VAE is always loaded from `--vqvae_checkpoint` (the Phase-1 bare state_dict).
    Denoiser is loaded from the PL checkpoint at `--ldm_checkpoint`; its state_dict
    keys are prefixed with "denoiser." inside the PL checkpoint.
    """
    # ── VQ-VAE ────────────────────────────────────────────────────────────────
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

    # ── Denoiser ─────────────────────────────────────────────────────────────
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
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def _duplicate_edge_index(edge_index, num_nodes, parallel_sampling, device):
    """Tile edge_index P times with per-copy node offsets."""
    ei     = edge_index.reshape(2, 1, -1).to(device)
    indent = torch.arange(parallel_sampling, device=device).view(1, -1, 1) * num_nodes
    return (ei + indent).reshape(2, -1)


def _denoise_step(denoiser, diffusion, args, x, z_t, t_arr, edge_index, device):
    """
    One DDPM / DDIM reverse step with x0-parameterisation.

    Computes implied noise ε̃ = (z_t − √ᾱ · z̃_0) / √(1−ᾱ), then applies the
    standard gaussian_posterior (same equations as COMetaModel / LatentDifusco).
    """
    t_val = int(t_arr[0])

    with torch.no_grad():
        N        = x.shape[0]
        t_tensor = torch.full((N,), float(t_val), device=device)
        z_0_pred = denoiser(x.float(), z_t, t_tensor, edge_index.long())   # (N, d)

        atbar_t  = float(diffusion.alphabar[t_arr])
        eps_impl = (z_t - np.sqrt(atbar_t) * z_0_pred) / np.sqrt(max(1.0 - atbar_t, 1e-8))

    # gaussian_posterior (ε-based; DDPM or DDIM)
    from torch import from_numpy
    t_tensor_1 = from_numpy(t_arr).view(1)
    target_t_  = from_numpy(np.array([t_val - 1]).astype(int)).view(1)

    atbar = diffusion.alphabar[t_tensor_1]
    at    = diffusion.alpha[t_tensor_1]

    inference_trick = getattr(args, "inference_trick", None)
    if inference_trick is None or t_val <= 1:
        atbar_prev = diffusion.alphabar[t_tensor_1 - 1]
        beta_tilde = diffusion.beta[t_val - 1] * (1 - atbar_prev) / (1 - atbar)
        z          = torch.randn_like(z_t) if t_val > 1 else torch.zeros_like(z_t)
        z_t_new    = (
            (1 / np.sqrt(at)).item() *
            (z_t - ((1 - at) / np.sqrt(1 - atbar)).item() * eps_impl)
            + np.sqrt(beta_tilde).item() * z
        )
    else:  # ddim
        atbar_target = diffusion.alphabar[target_t_]
        z_t_new = (
            np.sqrt(atbar_target / atbar).item() *
            (z_t - np.sqrt(1 - atbar).item() * eps_impl)
            + np.sqrt(1 - atbar_target).item() * eps_impl
        )
    return z_t_new


def generate_heatmap(vqvae, denoiser, diffusion, args, x, edge_index, device):
    """
    Run the full reverse diffusion chain for one problem instance.

    Supports `parallel_sampling` independent chains (run together on GPU) and
    `sequential_sampling` repeated runs (run one after another).

    Args:
        x:          (N, 2) float tensor  — node coordinates
        edge_index: (2, E) long tensor   — K-NN graph (K = eval_sparse_factor)

    Returns:
        adj_sparse: (S·P·N·K,) numpy array — flat edge probs, matches the
                    sparse-graph input expected by merge_tours.
                    S = sequential_sampling, P = parallel_sampling.
        adj_dense:  (N, N) numpy array  — symmetrised mean heatmap across all
                    samples; use for MCTS / LKH-3 candidate generation.
        np_edge_index: (2, E) numpy array
    """
    N  = x.shape[0]
    K  = args.eval_sparse_factor
    P  = args.parallel_sampling

    x_dev  = x.float().to(device)
    ei_dev = edge_index.long().to(device)

    x_rep  = x_dev.repeat(P, 1)                             if P > 1 else x_dev
    ei_rep = _duplicate_edge_index(ei_dev, N, P, device)    if P > 1 else ei_dev

    schedule = InferenceSchedule(
        inference_schedule = args.inference_schedule,
        T                  = diffusion.T,
        inference_T        = args.inference_diffusion_steps,
    )

    all_sparse_probs = []
    dense_accumulator = np.zeros((N, N), dtype=np.float64)

    for _ in range(args.sequential_sampling):
        z_t = torch.randn(P * N, args.latent_dim, device=device)

        for i in range(args.inference_diffusion_steps):
            t1, _t2 = schedule(i)
            z_t = _denoise_step(
                denoiser, diffusion, args, x_rep, z_t,
                np.array([t1]).astype(int),
                ei_rep, device,
            )

        with torch.no_grad():
            probs = vqvae.decode(z_t, x_rep, ei_rep, K)     # (P·N, K)

        sparse_flat = probs.view(-1).float().cpu().numpy()   # (P·N·K,)
        all_sparse_probs.append(sparse_flat)

        # Accumulate dense heatmap from the first parallel copy
        _probs_nk    = probs[:N].cpu().numpy()               # (N, K)
        _ei_np       = ei_dev.cpu().numpy()
        src, tgt     = _ei_np[0], _ei_np[1]
        dense_sample = np.zeros((N, N))
        dense_sample[src, tgt] = _probs_nk.reshape(-1)
        dense_accumulator += (dense_sample + dense_sample.T) * 0.5

    adj_sparse    = np.concatenate(all_sparse_probs, axis=0)        # (S·P·N·K,)
    adj_dense     = dense_accumulator / args.sequential_sampling     # (N, N) mean
    np_edge_index = edge_index.cpu().numpy()

    return adj_sparse, adj_dense, np_edge_index


# ──────────────────────────────────────────────────────────────────────────────
# Solver 1: greedy edge-insertion + batched 2-opt
# ──────────────────────────────────────────────────────────────────────────────

def solve_greedy_2opt(adj_sparse, np_edge_index, np_points, args, device):
    """
    Apply merge_tours (greedy insertion) then batched 2-opt.

    Returns a list of tours (one per parallel × sequential sample) and the
    best tour among them.
    """
    tours, _ = merge_tours(
        adj_sparse, np_points, np_edge_index,
        sparse_graph   = True,
        parallel_sampling = args.parallel_sampling * args.sequential_sampling,
    )

    solved, _ = batched_two_opt_torch(
        np_points.astype("float64"),
        np.array(tours).astype("int64"),
        max_iterations = args.two_opt_iterations,
        device         = device,
    )
    return solved


# ──────────────────────────────────────────────────────────────────────────────
# Solver 2: save numpy heatmaps (compatible with tsp_mcts/convert_numpy_to_txt.py)
# ──────────────────────────────────────────────────────────────────────────────

def save_numpy_heatmap(adj_dense, np_points, heatmap_dir, instance_idx):
    """Save dense heatmap and points to disk for offline MCTS processing."""
    os.makedirs(heatmap_dir, exist_ok=True)
    np.save(os.path.join(heatmap_dir, f"test-heatmap-{instance_idx}.npy"), adj_dense)
    np.save(os.path.join(heatmap_dir, f"test-points-{instance_idx}.npy"),  np_points)


# ──────────────────────────────────────────────────────────────────────────────
# Solver 3: LKH-3 with heatmap-guided candidate lists
# ──────────────────────────────────────────────────────────────────────────────

class LKH3Solver:
    """
    Thin Python wrapper around the LKH-3 binary.

    Pipeline for one instance:
      1. Write TSPLIB-format .tsp file  (EUC_2D, coordinates scaled to integers)
      2. Write candidate file (.cand)   (top-K neighbours per node from heatmap;
                                         alpha ∝ dist × (1 − prob) so short,
                                         high-probability edges rank first)
      3. Optionally write initial tour  (.tour) from greedy+2-opt warm-start
      4. Write LKH parameter file       (.par)
      5. Call: subprocess.run([lkh_binary, par_file])
      6. Parse OUTPUT_TOUR_FILE         (.sol)
      7. Return the tour as a 0-indexed list and its cost
    """

    COORD_SCALE = 1_000_000   # [0,1] float → integer grid

    def __init__(self, lkh_binary: str, runs: int = 1,
                 time_limit: float = 60.0, k_candidates: int = 20,
                 seed: int = 42):
        if not os.path.isfile(lkh_binary):
            raise FileNotFoundError(
                f"LKH-3 binary not found: {lkh_binary!r}\n"
                "Compile it first: cd LKH-3.0.6 && make"
            )
        self.binary      = os.path.abspath(lkh_binary)
        self.runs        = runs
        self.time_limit  = time_limit
        self.k_candidates = k_candidates
        self.seed        = seed

    # ------------------------------------------------------------------
    # Public interface

    def solve(self, np_points: np.ndarray, adj_dense: np.ndarray,
              initial_tour: Optional[np.ndarray] = None) -> tuple:
        """
        Solve one TSP instance using LKH-3 guided by adj_dense.

        Args:
            np_points:    (N, 2)  node coordinates in [0, 1]
            adj_dense:    (N, N)  symmetrised edge-probability matrix
            initial_tour: (N+1,) 0-indexed tour for warm-start (optional)

        Returns:
            tour: 0-indexed list of N+1 nodes (first == last)
            cost: float tour length
        """
        N       = np_points.shape[0]
        tmp_dir = tempfile.mkdtemp(prefix="lkh3_")
        try:
            tsp_file   = os.path.join(tmp_dir, "instance.tsp")
            cand_file  = os.path.join(tmp_dir, "candidates.cand")
            par_file   = os.path.join(tmp_dir, "instance.par")
            sol_file   = os.path.join(tmp_dir, "output.tour")
            init_file  = os.path.join(tmp_dir, "initial.tour") if initial_tour is not None else None

            self._write_tsp(tsp_file, np_points)
            self._write_candidates(cand_file, np_points, adj_dense)
            if initial_tour is not None:
                self._write_tour(init_file, initial_tour, N)
            self._write_par(par_file, tsp_file, cand_file, sol_file, init_file)

            result = subprocess.run(
                [self.binary, par_file],
                capture_output=True, text=True,
                timeout=max(self.time_limit * self.runs * 2, 120),
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"LKH-3 failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
                )

            tour = self._parse_tour(sol_file, N)
            cost = self._tour_cost(tour, np_points)
            return tour, cost
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Private helpers

    def _write_tsp(self, path: str, np_points: np.ndarray):
        N = np_points.shape[0]
        scaled = np_points * self.COORD_SCALE
        with open(path, "w") as f:
            f.write(f"NAME : ldm_instance\nTYPE : TSP\nDIMENSION : {N}\n"
                    f"EDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
            for i, (x, y) in enumerate(scaled):
                f.write(f"{i+1} {x:.2f} {y:.2f}\n")
            f.write("EOF\n")

    def _write_candidates(self, path: str, np_points: np.ndarray, adj_dense: np.ndarray):
        """
        Write candidate file for LKH-3.

        Format per line (1-indexed, no blank lines):
            node_id  Pi  k  cand_1  alpha_1  cand_2  alpha_2 ...
        """
        N    = np_points.shape[0]
        K    = min(self.k_candidates, N - 1)
        dmat = scipy.spatial.distance_matrix(np_points, np_points)

        MAX_ALPHA = 100_000

        with open(path, "w") as f:
            for i in range(N):
                # rank neighbours by descending heatmap probability
                probs_i = adj_dense[i].copy()
                probs_i[i] = -1.0           # exclude self
                cand_idx = np.argsort(-probs_i)[:K]

                parts = [str(i + 1), "0", str(K)]
                for j in cand_idx:
                    # alpha: lower = more preferred; combine distance and (1-prob)
                    p_ij   = float(adj_dense[i, j])
                    d_ij   = float(dmat[i, j]) * self.COORD_SCALE
                    alpha  = max(0, round(d_ij * (1.0 - p_ij)))
                    alpha  = min(alpha, MAX_ALPHA)
                    parts += [str(j + 1), str(alpha)]
                f.write(" ".join(parts) + "\n")
        # LKH expects the file to end with EOF marker on its own line
        with open(path, "a") as f:
            f.write("EOF\n")

    def _write_tour(self, path: str, tour: np.ndarray, N: int):
        """Write a TSPLIB tour file (for INITIAL_TOUR_FILE)."""
        with open(path, "w") as f:
            f.write(f"NAME : initial_tour\nTYPE : TOUR\nDIMENSION : {N}\nTOUR_SECTION\n")
            for node in tour[:-1]:   # exclude the closing node (==0)
                f.write(f"{int(node) + 1}\n")   # 1-indexed
            f.write("-1\nEOF\n")

    def _write_par(self, par_file, tsp_file, cand_file, sol_file, init_file):
        lines = [
            f"PROBLEM_FILE = {tsp_file}",
            f"CANDIDATE_FILE = {cand_file}",
            f"OUTPUT_TOUR_FILE = {sol_file}",
            f"RUNS = {self.runs}",
            f"TIME_LIMIT = {self.time_limit:.0f}",
            f"SEED = {self.seed}",
            f"MAX_CANDIDATES = {self.k_candidates}",
            "TRACE_LEVEL = 0",
        ]
        if init_file:
            lines.append(f"INITIAL_TOUR_FILE = {init_file}")
        with open(par_file, "w") as f:
            f.write("\n".join(lines) + "\n")

    @staticmethod
    def _parse_tour(path: str, _N: int) -> list[int]:
        """Parse TSPLIB TOUR_SECTION → 0-indexed list (length N+1, first==last)."""
        tour = []
        in_section = False
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line == "TOUR_SECTION":
                    in_section = True
                    continue
                if not in_section:
                    continue
                if line in ("-1", "EOF"):
                    break
                tour.append(int(line) - 1)     # 1-indexed → 0-indexed
        if not tour:
            raise ValueError(f"No tour found in {path!r}")
        tour.append(tour[0])                   # close the tour
        return tour

    @staticmethod
    def _tour_cost(tour: list[int], np_points: np.ndarray) -> float:
        dists = np.linalg.norm(np_points[tour[:-1]] - np_points[tour[1:]], axis=1)
        return float(dists.sum())


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(args):
    # ── Setup ──────────────────────────────────────────────────────────────────
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
    n_instances = min(args.num_instances, len(dataset)) if args.num_instances > 0 else len(dataset)

    lkh_solver = None
    if "lkh3" in args.solver:
        lkh_solver = LKH3Solver(
            lkh_binary   = args.lkh_binary,
            runs         = args.lkh_runs,
            time_limit   = args.lkh_time_limit,
            k_candidates = args.lkh_candidates,
            seed         = args.seed,
        )

    if "save" in args.solver and args.heatmap_dir:
        os.makedirs(args.heatmap_dir, exist_ok=True)

    # ── Per-instance results storage ───────────────────────────────────────────
    results = {s: [] for s in ("greedy", "lkh3")}
    gt_costs = []
    times    = {s: [] for s in ("inference", "greedy", "lkh3")}

    print(f"\nEvaluating {n_instances} instances  |  "
          f"N=auto  K={args.eval_sparse_factor}  "
          f"P={args.parallel_sampling}×S={args.sequential_sampling}  "
          f"solver={'|'.join(args.solver)}")
    print("─" * 70)

    for idx, batch in enumerate(tqdm(loader, total=n_instances)):
        if idx >= n_instances:
            break

        _, graph_data, _, _, gt_tour = batch
        x          = graph_data.x.reshape(-1, 2)           # (N, 2)
        edge_index = graph_data.edge_index.reshape(2, -1)  # (2, E)
        np_points  = x.cpu().numpy()
        np_gt_tour = gt_tour.cpu().numpy().reshape(-1)

        evaluator = TSPEvaluator(np_points)
        gt_cost   = evaluator.evaluate(np_gt_tour)
        gt_costs.append(gt_cost)

        # ── Inference ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        adj_sparse, adj_dense, np_edge_index = generate_heatmap(
            vqvae, denoiser, diffusion, args, x, edge_index, device)
        times["inference"].append(time.perf_counter() - t0)

        # ── Save numpy heatmap ─────────────────────────────────────────────────
        if "save" in args.solver:
            save_numpy_heatmap(adj_dense, np_points, args.heatmap_dir, idx)

        # ── Greedy + 2-opt ─────────────────────────────────────────────────────
        if "greedy" in args.solver:
            t0 = time.perf_counter()
            solved = solve_greedy_2opt(adj_sparse, np_edge_index, np_points, args, device)
            times["greedy"].append(time.perf_counter() - t0)

            total_samples = args.parallel_sampling * args.sequential_sampling
            costs  = [evaluator.evaluate(solved[i]) for i in range(total_samples)]
            best   = float(np.min(costs))
            gap    = (best - gt_cost) / gt_cost * 100.0
            results["greedy"].append({"cost": best, "gap_pct": gap})

        # ── LKH-3 ─────────────────────────────────────────────────────────────
        if "lkh3" in args.solver:
            # Optionally provide warm-start from greedy
            warm_tour = None
            if "greedy" in args.solver and len(solved) > 0:
                warm_tour = solved[0]   # use the first greedy tour as initial

            t0 = time.perf_counter()
            try:
                _lkh_tour, lkh_cost = lkh_solver.solve(np_points, adj_dense, warm_tour)
                times["lkh3"].append(time.perf_counter() - t0)
                gap = (lkh_cost - gt_cost) / gt_cost * 100.0
                results["lkh3"].append({"cost": lkh_cost, "gap_pct": gap})
            except Exception as exc:
                print(f"\n  [WARN] LKH-3 failed for instance {idx}: {exc}")
                results["lkh3"].append({"cost": float("nan"), "gap_pct": float("nan")})
                times["lkh3"].append(float("nan"))

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"Results over {len(gt_costs)} instances  (N={np_points.shape[0]})")
    print(f"  GT mean cost : {np.mean(gt_costs):.4f}")
    print(f"  Inference    : {np.mean(times['inference'])*1000:.1f} ms/instance")
    print()

    for solver_name in ("greedy", "lkh3"):
        if not results[solver_name]:
            continue
        costs = [r["cost"] for r in results[solver_name]]
        gaps  = [r["gap_pct"] for r in results[solver_name]
                 if not np.isnan(r["gap_pct"])]
        t_avg = np.nanmean(times[solver_name]) * 1000
        print(f"  [{solver_name:6s}]  "
              f"avg_cost={np.nanmean(costs):.4f}  "
              f"gap={np.nanmean(gaps):.2f}%±{np.nanstd(gaps):.2f}%  "
              f"solve_time={t_avg:.1f} ms")
    print("═" * 70)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if args.output_csv:
        import csv
        rows = []
        for i, gt in enumerate(gt_costs):
            row = {"instance": i, "gt_cost": gt,
                   "inference_ms": times["inference"][i] * 1000}
            for sn in ("greedy", "lkh3"):
                if i < len(results[sn]):
                    row[f"{sn}_cost"] = results[sn][i]["cost"]
                    row[f"{sn}_gap_pct"] = results[sn][i]["gap_pct"]
            rows.append(row)
        fields = list(rows[0].keys()) if rows else []
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {args.output_csv}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def arg_parser():
    p = ArgumentParser(description="Evaluate LDM-TSP (Phase 2).")

    # ── Checkpoints ───────────────────────────────────────────────────────────
    p.add_argument("--ldm_checkpoint",   type=str, required=True,
                   help="PL checkpoint from train_ldm.py (contains 'denoiser.*' keys).")
    p.add_argument("--vqvae_checkpoint", type=str, required=True,
                   help="Phase-1 best_vqvae.pt (bare TSPEdgeVQVAE state_dict).")

    # ── Test data ─────────────────────────────────────────────────────────────
    p.add_argument("--test_file",        type=str, required=True,
                   help="TSP data file (any size; can differ from training size).")
    p.add_argument("--num_instances",    type=int, default=0,
                   help="Number of test instances (0 = all).")
    p.add_argument("--eval_sparse_factor", type=int, default=-1,
                   help="K-NN K at eval time. Default = same as --sparse_factor.")

    # ── VQ-VAE architecture (must match Phase-1 checkpoint) ───────────────────
    p.add_argument("--vqvae_hidden_dim", type=int,   default=128)
    p.add_argument("--n_enc_layers",     type=int,   default=4)
    p.add_argument("--n_dec_layers",     type=int,   default=4)
    p.add_argument("--latent_dim",       type=int,   default=8)
    p.add_argument("--num_codes",        type=int,   default=512)
    p.add_argument("--vq_commitment_cost", type=float, default=0.25)
    p.add_argument("--sparse_factor",    type=int,   default=10,
                   help="K used in training (controls VQ-VAE decoder reshape).")
    p.add_argument("--aggregation",      type=str,   default="sum")

    # ── Denoiser architecture (must match LDM checkpoint) ─────────────────────
    p.add_argument("--n_layers",         type=int,   default=6)
    p.add_argument("--hidden_dim",       type=int,   default=256)

    # ── Diffusion ─────────────────────────────────────────────────────────────
    p.add_argument("--diffusion_steps",          type=int,   default=1000)
    p.add_argument("--diffusion_schedule",       type=str,   default="linear")
    p.add_argument("--inference_diffusion_steps",type=int,   default=50)
    p.add_argument("--inference_schedule",       type=str,   default="linear")
    p.add_argument("--inference_trick",          type=str,   default=None,
                   choices=[None, "ddim"])

    # ── Sampling ──────────────────────────────────────────────────────────────
    p.add_argument("--parallel_sampling",  type=int, default=1)
    p.add_argument("--sequential_sampling",type=int, default=1)

    # ── Solvers ───────────────────────────────────────────────────────────────
    p.add_argument("--solver", type=str, nargs="+",
                   default=["greedy"],
                   choices=["greedy", "save", "lkh3"],
                   help="One or more solvers to apply (space-separated).")
    p.add_argument("--two_opt_iterations", type=int, default=1000)

    # ── LKH-3 options ─────────────────────────────────────────────────────────
    p.add_argument("--lkh_binary",     type=str,   default="LKH-3.0.6/LKH",
                   help="Path to the compiled LKH-3 binary.")
    p.add_argument("--lkh_runs",       type=int,   default=1,
                   help="Number of independent LKH-3 runs per instance.")
    p.add_argument("--lkh_time_limit", type=float, default=60.0,
                   help="Wall-clock time limit per LKH-3 run (seconds).")
    p.add_argument("--lkh_candidates", type=int,   default=20,
                   help="Candidates per node for the LKH-3 candidate file.")
    p.add_argument("--seed",           type=int,   default=42)

    # ── Save options ──────────────────────────────────────────────────────────
    p.add_argument("--heatmap_dir",  type=str, default="results/heatmaps",
                   help="Output directory for saved numpy heatmaps (solver=save).")
    p.add_argument("--output_csv",   type=str, default="",
                   help="Optional: save per-instance results to this CSV file.")

    return p.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
