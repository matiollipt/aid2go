import os
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.utils import degree
from torch_geometric.nn import GATConv

# Set CUDA and PyTorch environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TORCH"] = torch.__version__

# Sampling by node degree (Focused Attention)
class FocusedGATConv(GATConv):
    def __init__(self, *args, sample_size=8, high_degree_threshold=10, **kwargs):
        super(FocusedGATConv, self).__init__(*args, **kwargs)
        self.sample_size = sample_size
        self.high_degree_threshold = high_degree_threshold

    def forward(self, x, edge_index, return_attention_weights=False):
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)

        # Create mask for high-degree nodes
        high_deg_mask = deg > self.high_degree_threshold

        # Keep all edges for low-degree nodes
        low_deg_edges = edge_index[:, ~high_deg_mask[row]]

        # Sample edges for high-degree nodes
        high_deg_edges = []
        high_deg_nodes = high_deg_mask.nonzero(as_tuple=True)[0]
        for node in high_deg_nodes:
            neighbors = col[row == node]
            if len(neighbors) > self.sample_size:
                sampled_neighbors = neighbors[torch.randperm(len(neighbors))[:self.sample_size]]
            else:
                sampled_neighbors = neighbors
            sampled_edges = torch.stack([torch.full_like(sampled_neighbors, node), sampled_neighbors], dim=0)
            high_deg_edges.append(sampled_edges)

        if high_deg_edges:
            high_deg_edges = torch.cat(high_deg_edges, dim=1)
            sampled_edges = torch.cat([low_deg_edges, high_deg_edges], dim=1)
        else:
            sampled_edges = low_deg_edges

        # Call the original forward method to compute attention
        out = super().forward(x, sampled_edges, return_attention_weights=True)
        x, (sampled_edges, alpha) = out

        # Apply softmax normalization to the attention coefficients
        alpha = F.softmax(alpha, dim=1)

        if return_attention_weights:
            return x, (sampled_edges, alpha)
        else:
            return x

class FocusedHGNNGAT(torch.nn.Module):
    def __init__(
        self,
        protein_features,
        go_features,
        out_features,
        dropout=0.6,
        heads=8,
        concat=True,
        protein_sample_size=8,
        protein_high_degree_threshold=10,
        go_sample_size=8,
        go_high_degree_threshold=20,
    ):
        super(FocusedHGNNGAT, self).__init__()
        per_head_features = out_features // heads if concat else out_features

        self.conv1_protein = FocusedGATConv(
            protein_features,
            per_head_features,
            heads=heads,
            concat=concat,
            dropout=dropout,
            sample_size=protein_sample_size,
            high_degree_threshold=protein_high_degree_threshold,
        )
        self.conv1_go = FocusedGATConv(
            go_features,
            per_head_features,
            heads=heads,
            concat=concat,
            dropout=dropout,
            sample_size=go_sample_size,
            high_degree_threshold=go_high_degree_threshold,
        )
        self.conv2_protein = FocusedGATConv(
            out_features,
            per_head_features,
            heads=heads,
            concat=concat,
            dropout=dropout,
            sample_size=protein_sample_size,
            high_degree_threshold=protein_high_degree_threshold,
        )
        self.conv2_go = FocusedGATConv(
            out_features,
            per_head_features,
            heads=heads,
            concat=concat,
            dropout=dropout,
            sample_size=go_sample_size,
            high_degree_threshold=go_high_degree_threshold,
        )

        final_out_features = out_features if concat else per_head_features
        self.protein_to_go = Linear(final_out_features * 2, 1)

    def forward(self, x_dict, edge_index_dict, mask=None):
        x_protein = F.elu(
            self.conv1_protein(
                x_dict["protein"], edge_index_dict[("protein", "ppi", "protein")]
            )
        )
        x_go = F.elu(
            self.conv1_go(x_dict["go"], edge_index_dict[("go", "relation", "go")])
        )

        x_protein = F.elu(
            self.conv2_protein(
                x_protein, edge_index_dict[("protein", "ppi", "protein")]
            )
        )
        x_go = F.elu(self.conv2_go(x_go, edge_index_dict[("go", "relation", "go")]))

        edge_index_protein_go = edge_index_dict[("protein", "associates_with", "go")]
        if mask is not None:
            edge_index_protein_go = edge_index_protein_go[:, mask]
        protein_features = x_protein[edge_index_protein_go[0]]
        go_features = x_go[edge_index_protein_go[1]]

        association_features = torch.cat([protein_features, go_features], dim=1)
        association_scores = self.protein_to_go(association_features).squeeze()
        return association_scores

    def get_attention_weights(self, x_dict, edge_index_dict):
        self.eval()
        with torch.no_grad():
            _, attention_protein_1 = self.conv1_protein(
                x_dict["protein"],
                edge_index_dict[("protein", "ppi", "protein")],
                return_attention_weights=True,
            )
            _, attention_protein_2 = self.conv2_protein(
                F.elu(self.conv1_protein(
                    x_dict["protein"], edge_index_dict[("protein", "ppi", "protein")])
                ),
                edge_index_dict[("protein", "ppi", "protein")],
                return_attention_weights=True,
            )

            _, attention_go_1 = self.conv1_go(
                x_dict["go"],
                edge_index_dict[("go", "relation", "go")],
                return_attention_weights=True,
            )
            _, attention_go_2 = self.conv2_go(
                F.elu(self.conv1_go(x_dict["go"], edge_index_dict[("go", "relation", "go")])),
                edge_index_dict[("go", "relation", "go")],
                return_attention_weights=True,
            )

            return [
                attention_protein_1,
                attention_go_1,
                attention_protein_2,
                attention_go_2,
            ]

