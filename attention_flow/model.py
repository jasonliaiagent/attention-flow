"""AttentionDiffusionNet: temporal encoder + graph attention, in pure PyTorch.

Design constraints, all evidence-driven:
- UNDIRECTED message passing (Phase 1b: no directional asymmetry exists).
- Small (~20k params): four graphs of 15-34 nodes and ~6,500 days of history
  do not justify more capacity, and honest evaluation beats a big model.
- Dense [n, n] adjacency instead of torch_geometric: our graphs are tiny, and
  zero extra dependencies keeps the repo runnable everywhere.

The `use_graph` flag is the experiment: the identical model with adjacency
replaced by the identity matrix is the ablation that isolates what the graph
itself contributes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GraphAttentionLayer(nn.Module):
    """Single-head GAT-style layer on a dense adjacency mask."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn = nn.Linear(2 * dim, 1, bias=False)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: [B, n, d], adj: [n, n] (1 = edge or self-loop)
        z = self.proj(h)
        n = z.shape[1]
        zi = z.unsqueeze(2).expand(-1, -1, n, -1)  # [B, n, n, d]
        zj = z.unsqueeze(1).expand(-1, n, -1, -1)
        scores = self.act(self.attn(torch.cat([zi, zj], dim=-1))).squeeze(-1)  # [B, n, n]
        scores = scores.masked_fill(adj == 0, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        return torch.relu(torch.einsum("bij,bjd->bid", alpha, z))


class AttentionDiffusionNet(nn.Module):
    def __init__(self, hidden: int = 32, gnn_layers: int = 2, use_graph: bool = True):
        super().__init__()
        self.use_graph = use_graph
        self.encoder = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.gnn = nn.ModuleList(GraphAttentionLayer(hidden) for _ in range(gnn_layers))
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, n, W] shock windows; adj: [n, n]
        B, n, W = x.shape
        _, h = self.encoder(x.reshape(B * n, W, 1))
        h = h.squeeze(0).reshape(B, n, -1)  # [B, n, hidden]

        if not self.use_graph:
            adj = torch.eye(n, device=x.device)
        g = h
        for layer in self.gnn:
            g = layer(g, adj) + g  # residual
        return self.head(torch.cat([h, g], dim=-1)).squeeze(-1)  # [B, n]
