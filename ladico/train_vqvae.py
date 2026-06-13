"""
Training script for the TSP Edge VQ-VAE (Phase 1 of the LDM pipeline).

Usage examples
--------------
# Train on TSP-50
python ladico/train_vqvae.py \
    --storage_path /path/to/data \
    --training_split   tsp/tsp50_train_concorde.txt \
    --validation_split tsp/tsp50_test_concorde.txt \
    --test_split       tsp/tsp50_test_concorde.txt \
    --sparse_factor 10 --latent_dim 8 --num_codes 512 \
    --do_train

# Train on TSP-100
python ladico/train_vqvae.py ... --training_split tsp/tsp100_train_concorde.txt ...
"""

import os
import sys
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
from pytorch_lightning.utilities import rank_zero_info
from torch_geometric.data import DataLoader as GraphDataLoader

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, '..', 'difusco'))
from co_datasets.tsp_graph_dataset import TSPGraphDataset
from edge_vqvae import TSPEdgeVQVAE, bhh_energy_loss


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Lightning module
# ──────────────────────────────────────────────────────────────────────────────

class VQVAELitModule(pl.LightningModule):
    """
    Lightning wrapper for TSPEdgeVQVAE.

    Loss:  BCE_reconstruction  +  VQ_commitment  +  alpha · BHH_energy
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model = TSPEdgeVQVAE(
            n_enc_layers=args.n_enc_layers,
            n_dec_layers=args.n_dec_layers,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_codes=args.num_codes,
            commitment_cost=args.vq_commitment_cost,
            aggregation=args.aggregation,
        )

        self.sparse_factor = args.sparse_factor
        self.bhh_alpha = args.bhh_alpha

        # ── Datasets ──────────────────────────────────────────────────────────
        self.train_dataset = self._make_dataset(args.training_split)
        self.val_dataset   = TSPGraphDataset(
            data_file=os.path.join(args.storage_path, args.validation_split),
            sparse_factor=args.sparse_factor,
        )
        self.test_dataset  = TSPGraphDataset(
            data_file=os.path.join(args.storage_path, args.test_split),
            sparse_factor=args.sparse_factor,
        )

    def _make_dataset(self, split_spec: str):
        """
        Support comma-separated split paths so users can mix TSP-50 and TSP-100.
        e.g. --training_split "tsp/tsp50_train.txt,tsp/tsp100_train.txt"
        """
        files = [f.strip() for f in split_spec.split(",")]
        datasets = [
            TSPGraphDataset(
                data_file=os.path.join(self.args.storage_path, f),
                sparse_factor=self.args.sparse_factor,
            ) for f in files
        ]
        if len(datasets) == 1:
            return datasets[0]
        return torch.utils.data.ConcatDataset(datasets)

    # ── Core forward ──────────────────────────────────────────────────────────

    def _step(self, batch):
        """
        Shared logic for train / val / test steps.

        Returns a dict of individual loss components and metric tensors so each
        step function can log what it needs.
        """
        _, graph_data, point_indicator, _edge_indicator, _ = batch

        x          = graph_data.x.float()                 # (B·N, 2)
        edge_index = graph_data.edge_index                 # (2, B·N·K)
        edge_attr  = graph_data.edge_attr.float()          # (B·N·K, 1)

        batch_size       = point_indicator.shape[0]
        nodes_per_graph  = (point_indicator.sum() // batch_size).item()  # N

        # ── Forward ───────────────────────────────────────────────────────────
        edge_probs_nk, _z_q, indices, vq_loss = self.model(
            x, edge_attr, edge_index, self.sparse_factor
        )
        # edge_probs_nk: (B·N, K) ∈ (0, 1)

        # Ground-truth labels reshaped to (B·N, K)
        targets_nk = edge_attr.view(-1, self.sparse_factor)

        # BCE reconstruction loss (clamp avoids log(0))
        bce = F.binary_cross_entropy(
            edge_probs_nk.clamp(1e-6, 1.0 - 1e-6), targets_nk
        )

        # BHH energy regularizer
        energy = bhh_energy_loss(
            edge_probs_nk, x, edge_index,
            self.sparse_factor, int(nodes_per_graph),
        )

        total = bce + vq_loss + self.bhh_alpha * energy

        return dict(
            total=total, bce=bce, vq=vq_loss, energy=energy,
            probs=edge_probs_nk.detach(), targets=targets_nk.detach(),
            indices=indices.detach(),
        )

    # ── PL steps ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        out = self._step(batch)
        self.log("train/total_loss", out["total"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/bce_loss",   out["bce"],   on_step=False, on_epoch=True)
        self.log("train/vq_loss",    out["vq"],    on_step=False, on_epoch=True)
        self.log("train/bhh_energy", out["energy"],on_step=False, on_epoch=True)
        return out["total"]

    def validation_step(self, batch, batch_idx):
        out  = self._step(batch)
        preds    = (out["probs"] > 0.5).float()
        targets  = out["targets"]

        accuracy = (preds == targets).float().mean()

        # Recall on tour edges (positive class)
        tp     = (preds * targets).sum()
        fn     = ((1.0 - preds) * targets).sum()
        recall = tp / (tp + fn + 1e-8)

        # Codebook utilisation
        util = out["indices"].unique().numel() / self.model.vq.num_codes

        self.log("val/total_loss",  out["total"],  on_epoch=True, sync_dist=True)
        self.log("val/bce_loss",    out["bce"],    on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val/vq_loss",     out["vq"],     on_epoch=True, sync_dist=True)
        self.log("val/bhh_energy",  out["energy"], on_epoch=True, sync_dist=True)
        self.log("val/accuracy",    accuracy,      on_epoch=True, sync_dist=True)
        self.log("val/tour_recall", recall,        on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val/codebook_util", util,        on_epoch=True, sync_dist=True)
        return out["total"]

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        rank_zero_info(f"VQ-VAE trainable parameters: {n_params:,}")
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        if self.args.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.num_epochs, eta_min=1e-6
            )
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        return optimizer

    # ── Data loaders ──────────────────────────────────────────────────────────

    def train_dataloader(self):
        return GraphDataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=(self.args.num_workers > 0),
            drop_last=True,
        )

    def val_dataloader(self):
        n = min(self.args.validation_examples, len(self.val_dataset))
        subset = torch.utils.data.Subset(self.val_dataset, range(n))
        return GraphDataLoader(
            subset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

    def test_dataloader(self):
        return GraphDataLoader(
            self.test_dataset, batch_size=1, shuffle=False
        )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def arg_parser():
    p = ArgumentParser(description="Train TSP Edge VQ-VAE (Phase 1).")

    # ── Data ─────────────────────────────────────────────────────────────────
    p.add_argument("--storage_path",    type=str, required=True,
                   help="Root directory containing data splits.")
    p.add_argument("--training_split",  type=str, required=True,
                   help="Relative path(s) to training file(s). Comma-separate for mixed sizes.")
    p.add_argument("--validation_split",type=str, required=True)
    p.add_argument("--test_split",      type=str, required=True)
    p.add_argument("--validation_examples", type=int, default=256)
    p.add_argument("--sparse_factor",   type=int, default=10,
                   help="K: number of nearest neighbours per node.")

    # ── Model ────────────────────────────────────────────────────────────────
    p.add_argument("--hidden_dim",          type=int,   default=128,
                   help="GNN hidden dimension H.")
    p.add_argument("--latent_dim",          type=int,   default=8,
                   help="Per-node latent dimension d (e.g. 4 or 8).")
    p.add_argument("--num_codes",           type=int,   default=512,
                   help="VQ codebook size V.")
    p.add_argument("--n_enc_layers",        type=int,   default=4,
                   help="GNN layers in the encoder.")
    p.add_argument("--n_dec_layers",        type=int,   default=4,
                   help="GNN layers in the decoder.")
    p.add_argument("--vq_commitment_cost",  type=float, default=0.25)
    p.add_argument("--aggregation",         type=str,   default="sum",
                   choices=["sum", "mean", "max"])

    # ── Loss ─────────────────────────────────────────────────────────────────
    p.add_argument("--bhh_alpha", type=float, default=0.01,
                   help="Weight α for the BHH energy regularizer.")

    # ── Training ─────────────────────────────────────────────────────────────
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--num_epochs",    type=int,   default=100)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-5)
    p.add_argument("--lr_scheduler",  type=str,   default="cosine",
                   choices=["constant", "cosine"])
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--fp16",          action="store_true")

    # ── Logging / checkpointing ───────────────────────────────────────────────
    p.add_argument("--project_name",       type=str, default="tsp_vqvae")
    p.add_argument("--wandb_entity",       type=str, default=None)
    p.add_argument("--wandb_logger_name",  type=str, default="vqvae_run")
    p.add_argument("--resume_id",          type=str, default=None)
    p.add_argument("--ckpt_path",          type=str, default=None,
                   help="Path to a PL checkpoint to resume from.")
    p.add_argument("--output_dir",         type=str, default="checkpoints",
                   help="Directory for saving model weights.")

    # ── Actions ──────────────────────────────────────────────────────────────
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_test",  action="store_true")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    model = VQVAELitModule(args)

    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
    wandb_logger = WandbLogger(
        name=args.wandb_logger_name,
        project=args.project_name,
        entity=args.wandb_entity,
        save_dir=os.path.join(args.storage_path, "models"),
        id=args.resume_id or wandb_id,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/bce_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        dirpath=args.output_dir,
        filename="vqvae-{epoch:03d}-{val/bce_loss:.4f}",
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
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.num_epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback],
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        strategy=strategy,
        precision=16 if args.fp16 else 32,
    )

    rank_zero_info(f"\n{'─'*60}\n{str(model.model)}\n{'─'*60}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.do_train:
        trainer.fit(model, ckpt_path=args.ckpt_path)

        # Save only the TSPEdgeVQVAE state dict for easy loading in Phase 2.
        # Extract from the PL checkpoint file to avoid save_hyperparameters issues.
        best_pl_path = checkpoint_callback.best_model_path
        if best_pl_path and os.path.isfile(best_pl_path):
            pl_ckpt = torch.load(best_pl_path, map_location="cpu")
            vqvae_state = {
                k[len("model."):]: v
                for k, v in pl_ckpt["state_dict"].items()
                if k.startswith("model.")
            }
            out_path = os.path.join(args.output_dir, "best_vqvae.pt")
            torch.save({"state_dict": vqvae_state, "args": vars(args)}, out_path)
            rank_zero_info(f"Best VQ-VAE weights saved → {out_path}")

        if args.do_test:
            trainer.test(ckpt_path=best_pl_path)

    # ── Test only ─────────────────────────────────────────────────────────────
    elif args.do_test:
        trainer.validate(model, ckpt_path=args.ckpt_path)
        trainer.test(model, ckpt_path=args.ckpt_path)

    trainer.logger.finalize("success")


if __name__ == "__main__":
    args = arg_parser()
    main(args)
