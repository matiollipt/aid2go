"""Relation-specific graph attention with a reproducible incoming-neighbor budget."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv


class DegreeCappedGATConv(nn.Module):
    def __init__(self, hidden_channels: int, heads: int, budget: int, dropout: float,
                 relation_seed: int):
        super().__init__()
        if budget < 1:
            raise ValueError("Attention neighbor budgets must be positive")
        self.budget = budget
        self.relation_seed = relation_seed
        self.conv = GATv2Conv(
            (hidden_channels, hidden_channels), hidden_channels, heads=heads,
            concat=False, dropout=dropout, add_self_loops=False,
        )
        self.last_attention = None

    def capped_edges(self, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return edge_index
        source, target = edge_index
        # Sort by destination and a stable integer hash. The selection is
        # independent of edge order, model weights, and train/eval mode.
        tie = (source.long() * 1103515245 + target.long() * 12345
               + self.relation_seed) & 0x7fffffff
        order = torch.argsort(target.long() * 2147483648 + tie, stable=True)
        sorted_target = target[order]
        positions = torch.arange(order.numel(), device=edge_index.device)
        starts = torch.where(
            torch.cat((torch.ones(1, dtype=torch.bool, device=edge_index.device),
                       sorted_target[1:] != sorted_target[:-1])), positions, 0,
        )
        rank = positions - torch.cummax(starts, dim=0).values
        return edge_index[:, order[rank < self.budget]]

    def forward(self, x, edge_index):
        selected = self.capped_edges(edge_index)
        out, attention = self.conv(x, selected, return_attention_weights=True)
        self.last_attention = (attention[0].detach(), attention[1].detach())
        return out


class DegreeAttentionModel(nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers=2,
                 dropout=0.0, heads=4, attention_budgets=None):
        super().__init__()
        self.dropout = dropout
        budgets = attention_budgets or {}
        self.project = nn.ModuleDict({node_type: nn.LazyLinear(hidden_channels)
                                      for node_type in metadata[0]})
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            relations = {}
            for relation_index, edge_type in enumerate(metadata[1]):
                relations[edge_type] = DegreeCappedGATConv(
                    hidden_channels, heads, budgets.get(edge_type[1], 16), dropout,
                    relation_seed=layer * 1009 + relation_index,
                )
            self.convs.append(HeteroConv(relations, aggr="mean"))
        self.output = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index,
                protein_n_id=None, go_node_ids=None):
        states = {name: F.relu(self.project[name](features))
                  for name, features in x_dict.items()}
        for conv in self.convs:
            updates = conv(states, edge_index_dict)
            states = {
                name: F.dropout(F.elu(states[name] + updates.get(name, torch.zeros_like(states[name]))),
                                p=self.dropout, training=self.training)
                for name in states
            }
        protein = self.output(states["protein"])[edge_label_index[0]]
        go = self.output(states["go"])[edge_label_index[1]]
        return (protein * go).sum(dim=-1)

    def attention_by_relation(self):
        # PyG HeteroConv stores tuple edge types as joined string module keys.
        # Return tuple keys so explanation code can match data.edge_index_dict.
        return [{tuple(str(edge_type).removeprefix("<").removesuffix(">").split("___")):
                 layer.convs[edge_type].last_attention
                 for edge_type in layer.convs} for layer in self.convs]
