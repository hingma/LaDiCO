# LaDiCO: Latent Diffusion Model for Combinatorial Optimization

A two-phase **Latent Diffusion Model (LDM)** for the Travelling Salesman Problem, built on the [DIFUSCO](https://arxiv.org/abs/2302.08224) codebase.

Rather than diffusing directly over the edge-probability heatmap, this approach learns a compact discrete latent space over TSP solutions via a **VQ-VAE** (Phase 1) and then runs **Gaussian diffusion** inside that latent space (Phase 2).

---

## Method Overview

```
Phase 1 — VQ-VAE                Phase 2 — Latent Diffusion
─────────────────────           ──────────────────────────────────────────
 TSP solution (edges)           z_T ~ N(0, I)
        │                              │
   Encoder (GNN)                Denoiser (AdaGNN, t-conditioned)
        │                              │  × T reverse steps
   VQ bottleneck                       │
  (discrete codes)              z̃_0 (predicted clean latent)
        │                              │
   Decoder (GNN)             frozen VQ-VAE decoder
        │                              │
edge-prob heatmap            edge-prob heatmap → tour
```

### Phase 1 — Edge VQ-VAE (`ladico/edge_vqvae.py`)

| Component | Details |
|-----------|---------|
| Encoder | L-layer anisotropic gated-GCN; inputs: node coordinates + binary tour-edge labels + K-NN distances → **(N, d)** continuous latents |
| Quantizer | Nearest-codebook lookup with straight-through estimator; codebook size V |
| Decoder | Mirror gated-GCN; inputs: quantized latents + coordinates → **(N, K)** edge-probability heatmap (Sigmoid) |
| Loss | BCE reconstruction + VQ commitment + α · BHH-energy regularizer |

The **BHH energy** regularizer (Beardwood–Halton–Hammersley theorem) penalises probability mass on geometrically long edges, normalised by √N so it is scale-invariant across problem sizes.

### Phase 2 — Latent Gaussian Diffusion (`ladico/latent_difusco.py`)

Training uses **x0-parameterisation**:
1. Encode a ground-truth solution through the frozen VQ-VAE → **z_0**
2. Sample t, add Gaussian noise: **z_t = √ᾱ_t · z_0 + √(1−ᾱ_t) · ε**
3. Predict clean latent **z̃_0** with `LatentDiffuscoDenoiser`
4. Loss = MSE(z̃_0, z_0)

The denoiser is an **AdaGNN**: a gated graph convolution network whose node updates are conditioned on a per-node signal combining a sinusoidal timestep embedding and the node's 2-D coordinate embedding via **DiT-style Adaptive Layer Normalisation**.

Inference supports both DDPM and DDIM sampling schedules.

---

## Codebase Structure

```
ladico/                         LaDiCO-specific code (VQ-VAE + LDM)
├── edge_vqvae.py               VQ-VAE (encoder, VQ, decoder, BHH loss)
├── latent_difusco.py           Phase-2 Lightning module + AdaGNN denoiser
├── train_vqvae.py              Phase-1 training script
├── train_ldm.py                Phase-2 training script
└── evaluate_ldm.py             Standalone evaluation (greedy, MCTS, LKH-3)

difusco/                        Original DIFUSCO code + shared infrastructure
├── models/
│   └── gnn_encoder.py          Shared GNN primitives (used by VQ-VAE and DIFUSCO)
├── co_datasets/
│   └── tsp_graph_dataset.py    K-NN graph dataset for TSP
└── utils/
    ├── diffusion_schedulers.py  Gaussian diffusion + inference schedule
    └── tsp_utils.py             merge_tours, batched 2-opt, TSPEvaluator
```

`ladico/` scripts add both `ladico/` and `difusco/` to `sys.path` at startup, so
LaDiCO modules (e.g. `edge_vqvae`) are imported from `ladico/` while shared
infrastructure (`co_datasets`, `models/gnn_encoder`, `utils`) is resolved from
`difusco/` without duplication.

---

## Setup

```bash
conda env create -f environment.yml
conda activate difusco
```

Build the Cython tour-merging extension (required for TSP evaluation):

```bash
cd difusco/utils/cython_merge
python setup.py build_ext --inplace
cd -
```

To use the **LKH-3** solver at evaluation time, compile the binary:

```bash
cd LKH-3.0.6 && make
```

---

## Data

Place data files under a `storage_path` root. The dataset files follow the DIFUSCO format (one instance per line). See the `data/` folder for examples.

```
storage_path/
└── tsp/
    ├── tsp50_train_concorde.txt
    ├── tsp50_test_concorde.txt
    ├── tsp100_train_concorde.txt
    └── tsp100_test_concorde.txt
```

---

## Training

### Phase 1 — Train the VQ-VAE

```bash
python ladico/train_vqvae.py \
    --storage_path data/ \
    --training_split   tsp/tsp50_train_concorde.txt \
    --validation_split tsp/tsp50_test_concorde.txt \
    --test_split       tsp/tsp50_test_concorde.txt \
    --sparse_factor 10 \
    --latent_dim 8 --num_codes 512 \
    --hidden_dim 128 --n_enc_layers 4 --n_dec_layers 4 \
    --batch_size 64 --num_epochs 100 \
    --output_dir checkpoints/vqvae_tsp50 \
    --do_train
```

`--training_split`: Relative path(s) to training file(s).

Mix TSP-50 and TSP-100 with a comma-separated `--training_split`:

```bash
--training_split "tsp/tsp50_train_concorde.txt,tsp/tsp100_train_concorde.txt"
```

The best checkpoint is saved as `checkpoints/vqvae_tsp50/best_vqvae.pt` (a bare `state_dict`, no Lightning wrapper).

**Key Phase-1 hyperparameters**

| Flag | Default | Description |
|------|---------|-------------|
| `--latent_dim` | 8 | Per-node latent dimension d |
| `--num_codes` | 512 | VQ codebook size V |
| `--hidden_dim` | 128 | GNN hidden dimension |
| `--sparse_factor` | 10 | K-NN neighbours per node |
| `--bhh_alpha` | 0.01 | BHH energy weight α |
| `--vq_commitment_cost` | 0.25 | VQ commitment cost β |

**Monitored validation metrics**

- `val/bce_loss` — reconstruction quality (checkpoint criterion)
- `val/tour_recall` — fraction of true tour edges recovered
- `val/codebook_util` — fraction of codebook entries used

---

### Phase 2 — Train the Latent Diffusion Model

Requires the `best_vqvae.pt` produced by Phase 1. Architecture flags that must match Phase 1 are marked *(Phase-1 match)*.

```bash
python ladico/train_ldm.py \
    --storage_path /path/to/data \
    --training_split   tsp/tsp50_train_concorde.txt \
    --validation_split tsp/tsp50_test_concorde.txt \
    --test_split       tsp/tsp50_test_concorde.txt \
    --vqvae_checkpoint checkpoints/vqvae_tsp50/best_vqvae.pt \
    --sparse_factor 10 \
    --latent_dim 8 --num_codes 512 \        # (Phase-1 match)
    --vqvae_hidden_dim 128 \                # (Phase-1 match)
    --n_enc_layers 4 --n_dec_layers 4 \     # (Phase-1 match)
    --hidden_dim 256 --n_layers 6 \
    --diffusion_steps 1000 --diffusion_schedule linear \
    --batch_size 64 --num_epochs 200 \
    --output_dir checkpoints/ldm_tsp50 \
    --do_train --do_test
```

**Key Phase-2 hyperparameters**

| Flag | Default | Description |
|------|---------|-------------|
| `--hidden_dim` | 256 | AdaGNN denoiser hidden dimension |
| `--n_layers` | 6 | AdaGNN denoiser layers |
| `--diffusion_steps` | 1000 | Training diffusion steps T |
| `--inference_diffusion_steps` | 50 | Reverse steps at inference |
| `--inference_trick` | None | `ddim` for deterministic sampling |
| `--parallel_sampling` | 1 | Independent chains per instance |
| `--sequential_sampling` | 1 | Sequential restarts per instance |

**Checkpoint criterion:** `val/solved_cost` (best tour length after 2-opt).

---

## Evaluation

Use `evaluate_ldm.py` for flexible, solver-agnostic evaluation without Lightning overhead.

### Greedy edge-insertion + batched 2-opt (fast)

```bash
python ladico/evaluate_ldm.py \
    --ldm_checkpoint   checkpoints/ldm_tsp50/last.ckpt \
    --vqvae_checkpoint checkpoints/vqvae_tsp50/best_vqvae.pt \
    --test_file        data/tsp/tsp50_test_concorde.txt \
    --solver greedy \
    --parallel_sampling 8 --two_opt_iterations 1000
```

### LKH-3 guided by heatmap

```bash
python ladico/evaluate_ldm.py \
    --ldm_checkpoint   checkpoints/ldm_tsp50/last.ckpt \
    --vqvae_checkpoint checkpoints/vqvae_tsp50/best_vqvae.pt \
    --test_file        data/tsp/tsp50_test_concorde.txt \
    --solver greedy lkh3 \
    --lkh_binary LKH-3.0.6/LKH --lkh_runs 1 --lkh_candidates 20
```

### Save heatmaps for offline MCTS

```bash
python ladico/evaluate_ldm.py \
    --ldm_checkpoint   checkpoints/ldm_tsp50/last.ckpt \
    --vqvae_checkpoint checkpoints/vqvae_tsp50/best_vqvae.pt \
    --test_file        data/tsp/tsp500_test_concorde.txt \
    --solver save \
    --heatmap_dir results/heatmaps/tsp500
```

### Cross-scale generalisation

The LDM architecture is fully graph-based and O(N); it can be applied to problem sizes not seen during training.

```bash
# Model trained on TSP-50, evaluated on TSP-500
python ladico/evaluate_ldm.py \
    --ldm_checkpoint   checkpoints/ldm_tsp50/last.ckpt \
    --vqvae_checkpoint checkpoints/vqvae_tsp50/best_vqvae.pt \
    --test_file        data/tsp/tsp500_test_concorde.txt \
    --sparse_factor 10 \           # K used during VQ-VAE training
    --eval_sparse_factor 10 \      # K for the K-NN graph at eval
    --solver greedy
```

---

## Original DIFUSCO

This codebase extends [DIFUSCO](https://arxiv.org/abs/2302.08224) (Sun & Yang, NeurIPS 2023). The original Bernoulli and Gaussian diffusion models over TSP and MIS remain available via `difusco/train.py`.

```
@inproceedings{sun2023difusco,
    title={{DIFUSCO}: Graph-based Diffusion Solvers for Combinatorial Optimization},
    author={Zhiqing Sun and Yiming Yang},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=JV8Ff0lgVV}
}
```
