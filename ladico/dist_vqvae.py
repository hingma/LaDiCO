"""
Distance-based VQ-VAE for TSP: replaces the learned codebook with K-NN
aggregation in coordinate space.

Architecture
------------
Encoder  : Same anisotropic (gated) multi-layer GNN  →  (N, d) continuous node latents
DistVQ   : Per-node mean of GNN embeddings of K coordinate-nearest neighbours (STE)
Decoder  : Same mirror GNN  →  (N, K) continuous edge-probability heatmap  [Sigmoid output]
Loss     : BCE  +  commitment  +  α · BHH-energy

Difference from EdgeVQVAE
--------------------------
EdgeVQVAE  → learned codebook; each node embedding mapped to nearest codebook
             vector in *GNN embedding space*.  The codebook has its own loss term.

DistVQVAE  → no learnable codebook; the 'code' for node i is the mean of its
             K neighbours' GNN embeddings, where neighbours are selected by
             Euclidean coordinate distance (i.e. the pre-built K-NN graph).
             Only a commitment loss is used (no codebook loss to update).

Because the K-NN graph (edge_index) is already constructed from node coordinates,
we reuse it directly — no O(V²) distance recomputation is needed and batching is
automatically respected (edges only connect within-graph nodes).

Note on train_vqvae.py compatibility
--------------------------------------
The validation step logs val/codebook_util using model.vq.num_codes.
DistVQVAE exposes model.vq.num_nodes_in_batch instead; update that log line to:
    util = out["indices"].unique().numel() / out["indices"].numel()
which reports what fraction of nodes were selected as someone's nearest neighbour.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from edge_vqvae import (
    TSPVQVAEEncoder,
    TSPVQVAEDecoder,
    _edge_dists,
)


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate-distance quantizer
# ──────────────────────────────────────────────────────────────────────────────

class CoordKNNQuantizer(nn.Module):
    """
    Coordinate-distance K-NN quantizer.

    For each node i the 'code' z_q[i] is the mean GNN embedding of its K
    nearest neighbours in Euclidean coordinate space.  Because the K-NN graph
    (edge_index) is already built from those distances, we aggregate directly
    over existing edges rather than recomputing a (V, V) distance matrix.

    Straight-through estimator (STE): gradients to the encoder flow through z_e
    unchanged; the quantised value z_q is detached from the encoder graph.

    No learnable parameters — only a commitment loss is returned.
    """

    def __init__(self, commitment_cost: float = 0.25):
        super().__init__()
        self.commitment_cost = commitment_cost

    def forward(
        self,
        z_e: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        sparse_factor: int,
    ):
        """
        Args:
            z_e:           (V, d)   continuous encoder output
            x:             (V, 2)   node coordinates (used to find nearest neighbour index)
            edge_index:    (2, E)   K-NN directed edges  [E = V · K]
                           edge_index[0] = source node  (the one that has K neighbours)
                           edge_index[1] = target node  (a neighbour)
            sparse_factor: K        neighbours per node

        Returns:
            z_q_st : (V, d)   mean-aggregated code, encoder gradients via STE
            z_q    : (V, d)   mean-aggregated code, detached from encoder graph
            indices: (V,)     node index of each node's coordinate-nearest neighbour
            vq_loss: scalar   commitment loss  (β · ||z_e − z_q.detach()||²)
        """
        V, d = z_e.shape
        src = edge_index[0].long()   # (E,)  source node i
        tgt = edge_index[1].long()   # (E,)  neighbour node j

        # ── Mean of K neighbour embeddings ────────────────────────────────────
        z_q = torch.zeros(V, d, device=z_e.device, dtype=z_e.dtype)
        z_q.scatter_add_(0, src.unsqueeze(-1).expand(-1, d), z_e[tgt])
        z_q = z_q / sparse_factor    # (V, d)

        # ── Commitment loss (no codebook parameters to update) ────────────────
        vq_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())

        # ── Straight-through estimator ────────────────────────────────────────
        z_q_st = z_e + (z_q - z_e).detach()

        # ── Nearest-neighbour index per node (for analysis / logging) ─────────
        # Find which of the K neighbours is the geometrically closest.
        dists_nk = _edge_dists(x, edge_index).view(V, sparse_factor)   # (V, K)
        nn_local  = dists_nk.argmin(dim=1)                              # (V,)
        tgt_nk    = tgt.view(V, sparse_factor)                          # (V, K)
        indices   = tgt_nk[torch.arange(V, device=z_e.device), nn_local]  # (V,)

        return z_q_st, z_q, indices, vq_loss


# ──────────────────────────────────────────────────────────────────────────────
# Full Distance-based VQ-VAE
# ──────────────────────────────────────────────────────────────────────────────

class TSPDistVQVAE(nn.Module):
    """
    Distance-based VQ-VAE over the K-NN sparse TSP edge graph.

    Shares the encoder and decoder with TSPEdgeVQVAE; differs only in the
    quantisation step, which uses coordinate-space K-NN mean aggregation
    instead of a learned codebook lookup in embedding space.

    Attributes:
        latent_dim : d  (per-node latent dimension)
    """

    def __init__(
        self,
        n_enc_layers: int,
        n_dec_layers: int,
        hidden_dim: int,
        latent_dim: int,
        commitment_cost: float = 0.25,
        aggregation: str = "sum",
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = TSPVQVAEEncoder(n_enc_layers, hidden_dim, latent_dim, aggregation)
        self.vq      = CoordKNNQuantizer(commitment_cost)
        self.decoder = TSPVQVAEDecoder(n_dec_layers, hidden_dim, latent_dim, aggregation)

    # ------------------------------------------------------------------
    def encode(self, x, edge_attr, edge_index):
        """(V, 2), (E, 1), (2, E) → z_e: (V, d)"""
        return self.encoder(x, edge_attr, edge_index)

    def quantize(self, z_e, x, edge_index, sparse_factor):
        """z_e: (V, d) → (z_q_st, z_q, indices, vq_loss)"""
        return self.vq(z_e, x, edge_index, sparse_factor)

    def decode(self, z_q, x, edge_index, sparse_factor):
        """Returns edge_probs_nk: (V, K) ∈ [0, 1]"""
        flat = self.decoder(z_q, x, edge_index)    # (E, 1)
        return flat.view(-1, sparse_factor)         # (V, K)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        sparse_factor: int,
    ):
        """
        Args:
            x:             (V, 2)   node coordinates  [V = B·N]
            edge_attr:     (E, 1)   binary tour-membership labels  [E = V·K]
            edge_index:    (2, E)   K-NN edge indices (built from coordinate distances)
            sparse_factor: K

        Returns:
            edge_probs_nk : (V, K)   per-edge probabilities in node × neighbour layout
            z_q           : (V, d)   coordinate-aggregated latents (detached from encoder)
            indices       : (V,)     node index of each node's coordinate-nearest neighbour
            vq_loss       : scalar   commitment loss
        """
        z_e = self.encode(x, edge_attr, edge_index)
        z_q_st, z_q, indices, vq_loss = self.quantize(z_e, x, edge_index, sparse_factor)
        edge_probs_nk = self.decode(z_q_st, x, edge_index, sparse_factor)
        return edge_probs_nk, z_q, indices, vq_loss
