import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, to_hetero

class BaselineHeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = SAGEConv((-1, -1), hidden_channels)
            self.convs.append(to_hetero(conv, metadata, aggr='mean'))
            
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            
        return x_dict

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
    def __init__(self, metadata, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.encoder = BaselineHeteroGNN(metadata, hidden_channels, out_channels, num_layers)
        self.decoder = LinkPredictor()
        
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict["protein"], z_dict["go"], edge_label_index)
