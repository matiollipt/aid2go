import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv

class BaselineHeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            rel_convs = {}
            for edge_type in metadata[1]:
                # SAGEConv supports bipartite projections using (-1, -1)
                rel_convs[edge_type] = SAGEConv((-1, -1), hidden_channels)
            conv = HeteroConv(rel_convs, aggr='mean')
            self.convs.append(conv)
            
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # SAGEConv lazy initialization expects all feature tensors to be available
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.dropout(x.relu(), p=self.dropout, training=self.training)
                      for key, x in x_dict.items()}
            
        return {key: self.lin(x) for key, x in x_dict.items()}

class LinkPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x_source, x_target, edge_label_index):
        source_idx = edge_label_index[0]
        target_idx = edge_label_index[1]
        
        source_emb = x_source[source_idx]
        target_emb = x_target[target_idx]
        
        # Dot product
        return (source_emb * target_emb).sum(dim=-1)

class BaselineModel(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers=2, dropout=0.0):
        super().__init__()
        self.encoder = BaselineHeteroGNN(metadata, hidden_channels, out_channels,
                                         num_layers, dropout)
        self.decoder = LinkPredictor()
        
    def encode(self, x_dict, edge_index_dict):
        """Return learned node embeddings before protein-GO link decoding."""
        return self.encoder(x_dict, edge_index_dict)

    def forward(self, x_dict, edge_index_dict, edge_label_index, protein_n_id=None, go_node_ids=None):
        z_dict = self.encode(x_dict, edge_index_dict)
        return self.decoder(z_dict["protein"], z_dict["go"], edge_label_index)
