"""
Training script for Phase 2: Latent Diffusion Model on VQ-VAE codes.

Usage examples
--------------
# Train on TSP-50 (VQ-VAE checkpoint required from Phase 1)
python ladico/train_ldm.py \
    --storage_path /path/to/repo \
    --training_split   data/tsp/tsp50_train_concorde.txt \
    --validation_split data/tsp/tsp50_test_concorde.txt \
    --test_split       data/tsp/tsp50_test_concorde.txt \
    --vqvae_checkpoint /path/to/checkpoints/vqvae_tsp50/best_vqvae.pt \
    --sparse_factor 10 --latent_dim 8 \
    --do_train --do_test
"""

import os
import sys
from argparse import ArgumentParser

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
from pytorch_lightning.utilities import rank_zero_info

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, '..', 'difusco'))
from latent_difusco import LatentDifusco


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def arg_parser():
    p = ArgumentParser(description="Train LDM-TSP Phase 2: latent Gaussian diffusion.")

    # ── Data ─────────────────────────────────────────────────────────────────
    p.add_argument("--storage_path",      type=str, required=True)
    p.add_argument("--training_split",    type=str, required=True,
                   help="Relative path(s) to training file(s). Comma-separate for mixed sizes.")
    p.add_argument("--validation_split",  type=str, required=True)
    p.add_argument("--test_split",        type=str, required=True)
    p.add_argument("--validation_examples", type=int, default=256)
    p.add_argument("--sparse_factor",     type=int, default=10,
                   help="K: nearest neighbours per node in the K-NN graph.")

    # ── Frozen VQ-VAE (Phase 1 output) ───────────────────────────────────────
    p.add_argument("--vqvae_checkpoint",  type=str, required=True,
                   help="Path to best_vqvae.pt produced by train_vqvae.py.")
    p.add_argument("--vqvae_hidden_dim",  type=int, default=128,
                   help="VQ-VAE hidden dim — must match the Phase-1 checkpoint.")
    p.add_argument("--n_enc_layers",      type=int, default=4,
                   help="VQ-VAE encoder GNN layers (must match Phase-1).")
    p.add_argument("--n_dec_layers",      type=int, default=4,
                   help="VQ-VAE decoder GNN layers (must match Phase-1).")
    p.add_argument("--latent_dim",        type=int, default=8,
                   help="Per-node latent dimension d (must match Phase-1).")
    p.add_argument("--num_codes",         type=int, default=512,
                   help="VQ codebook size (must match Phase-1).")
    p.add_argument("--vq_commitment_cost",type=float, default=0.25,
                   help="VQ commitment cost (must match Phase-1).")

    # ── Denoiser ──────────────────────────────────────────────────────────────
    p.add_argument("--n_layers",          type=int, default=6,
                   help="AdaGNN denoiser layers.")
    p.add_argument("--hidden_dim",        type=int, default=256,
                   help="Denoiser GNN hidden dimension H.")
    p.add_argument("--aggregation",       type=str, default="sum",
                   choices=["sum", "mean", "max"])

    # ── Diffusion ─────────────────────────────────────────────────────────────
    p.add_argument("--diffusion_steps",          type=int, default=1000)
    p.add_argument("--diffusion_schedule",       type=str, default="linear",
                   choices=["linear", "cosine"])
    p.add_argument("--inference_diffusion_steps",type=int, default=50)
    p.add_argument("--inference_schedule",       type=str, default="linear",
                   choices=["linear", "cosine"])
    p.add_argument("--inference_trick",          type=str, default=None,
                   choices=[None, "ddim"])

    # ── Sampling / evaluation ─────────────────────────────────────────────────
    p.add_argument("--parallel_sampling",  type=int,   default=1)
    p.add_argument("--sequential_sampling",type=int,   default=1)
    p.add_argument("--two_opt_iterations", type=int,   default=1000)

    # ── Training ─────────────────────────────────────────────────────────────
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--num_epochs",    type=int,   default=200)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-5)
    p.add_argument("--lr_scheduler",  type=str,   default="cosine",
                   choices=["constant", "cosine"])
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--fp16",          action="store_true")

    # ── Logging / checkpointing ───────────────────────────────────────────────
    p.add_argument("--project_name",      type=str, default="tsp_ldm")
    p.add_argument("--wandb_entity",      type=str, default=None)
    p.add_argument("--wandb_logger_name", type=str, default="ldm_run")
    p.add_argument("--resume_id",         type=str, default=None)
    p.add_argument("--ckpt_path",         type=str, default=None,
                   help="PL checkpoint to resume from.")
    p.add_argument("--output_dir",        type=str, default="checkpoints/ldm",
                   help="Directory for saving denoiser weights.")

    # ── Actions ──────────────────────────────────────────────────────────────
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_test",  action="store_true")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    model = LatentDifusco(args)

    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
    wandb_logger = WandbLogger(
        name    = args.wandb_logger_name,
        project = args.project_name,
        entity  = args.wandb_entity,
        save_dir= os.path.join(args.storage_path, "models"),
        id      = args.resume_id or wandb_id,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor  = "val/solved_cost",
        mode     = "min",
        save_top_k = 3,
        save_last  = True,
        dirpath    = args.output_dir,
        filename   = "ldm-{epoch:03d}-{val/solved_cost:.4f}",
    )
    lr_callback = LearningRateMonitor(logging_interval="epoch")

    # ── Hardware ─────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices     = torch.cuda.device_count()
        strategy    = DDPStrategy(static_graph=True) if devices > 1 else "auto"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices     = 1
        strategy    = SingleDeviceStrategy(device=torch.device("mps"))
    else:
        accelerator = "cpu"
        devices     = None
        strategy    = SingleDeviceStrategy(device=torch.device("cpu"))

    trainer = Trainer(
        accelerator = accelerator,
        devices     = devices,
        max_epochs  = args.num_epochs,
        callbacks   = [TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback],
        logger      = wandb_logger,
        check_val_every_n_epoch = 5,
        strategy    = strategy,
        precision   = 16 if args.fp16 else 32,
    )

    rank_zero_info(f"\n{'─'*60}\n{str(model.denoiser)}\n{'─'*60}\n")

    if args.do_train:
        trainer.fit(model, ckpt_path=args.ckpt_path)
        if args.do_test:
            trainer.test(ckpt_path=checkpoint_callback.best_model_path)

    elif args.do_test:
        trainer.test(model, ckpt_path=args.ckpt_path)

    trainer.logger.finalize("success")


if __name__ == "__main__":
    args = arg_parser()
    main(args)
