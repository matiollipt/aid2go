import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, GCNConv, global_mean_pool
from torch_geometric.data import Batch

class ProteinGraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

class DecoderMLP(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_channels // 2, 1)
        )
    def forward(self, x):
        return self.mlp(x).squeeze(-1)

class SmartModel(torch.nn.Module):
    def __init__(
        self,
        metadata,
        hidden_channels,
        out_channels,
        num_layers=2,
        heads=4,
        go_ids=None,
        go_id_to_cat=None,
        protein_graphs=None,
        protein_representation="node",
        esm_dim=1280
    ):
        super().__init__()
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.protein_representation = protein_representation
        self.protein_graphs = protein_graphs
        
        # Save GO mappings
        self.go_ids = go_ids if go_ids is not None else []
        self.go_id_to_cat = go_id_to_cat if go_id_to_cat is not None else {}
        
        # 1. Protein Graph Encoder (if representation is graph)
        if self.protein_representation == "graph":
            # Input dimension from ESM-2 residue embeddings (default 1280)
            self.protein_encoder = ProteinGraphEncoder(esm_dim, hidden_channels)
            self.proj_protein = None
        else:
            self.protein_encoder = None
            # Project mean-pooled protein embedding
            self.proj_protein = nn.LazyLinear(hidden_channels)
            
        # 2. GO projection layer (projects BERT embedding + IA dimension, e.g. 768 + 1 = 769)
        self.proj_go = nn.LazyLinear(hidden_channels)
        
        # 3. Shared Heterogeneous Graph Transformer encoder
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # HGTConv requires known input/output dimensions
            conv = HGTConv(hidden_channels, hidden_channels, metadata, heads)
            self.convs.append(conv)
            
        # 4. Three separate decoders for MF, BP, CC
        # Input to decoder is concatenated source and target representation: 2 * hidden_channels
        self.decoders = torch.nn.ModuleList([
            DecoderMLP(2 * hidden_channels),  # MF (Molecular Function)
            DecoderMLP(2 * hidden_channels),  # BP (Biological Process)
            DecoderMLP(2 * hidden_channels)   # CC (Cellular Component)
        ])

    def encode(self, x_dict, edge_index_dict, protein_n_id=None):
        device = x_dict["go"].device
        
        # 1. Compute Protein node embeddings
        if self.protein_representation == "graph" and self.protein_graphs is not None:
            # Determine which protein graphs are in this batch
            if protein_n_id is not None:
                p_indices = protein_n_id.cpu().tolist()
            else:
                p_indices = list(range(x_dict["protein"].size(0)))
                
            sub_graphs = [self.protein_graphs[idx] for idx in p_indices]
            # Batch them together
            pyg_batch = Batch.from_data_list(sub_graphs).to(device)
            h_protein = self.protein_encoder(pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch)
        else:
            # Node mode
            h_protein = self.proj_protein(x_dict["protein"]).relu()
            
        # 2. Compute GO node embeddings
        h_go = self.proj_go(x_dict["go"]).relu()
        
        h_dict = {
            "protein": h_protein,
            "go": h_go
        }
        
        # 3. Message Passing with HGT
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {key: x.relu() for key, x in h_dict.items()}
            
        return h_dict

    def decode(self, h_dict, edge_label_index, go_node_ids=None):
        device = h_dict["go"].device
        # 4. Decode edges
        source_idx = edge_label_index[0]
        target_idx = edge_label_index[1]
        
        # Retrieve target GO IDs
        if go_node_ids is not None:
            target_go_ids = [go_node_ids[idx] for idx in target_idx.cpu().tolist()]
        elif self.go_ids:
            target_go_ids = [self.go_ids[idx] for idx in target_idx.cpu().tolist()]
        else:
            # Fallback if no ID mapping is available
            target_go_ids = []
            
        # Map target GO IDs to categories (default to BP=1 if missing)
        target_cats = torch.tensor(
            [self.go_id_to_cat.get(gid, 1) for gid in target_go_ids],
            dtype=torch.long,
            device=device
        )
        if len(target_cats) == 0:
            target_cats = torch.zeros(len(target_idx), dtype=torch.long, device=device)
            
        # Prepare output logits
        out = torch.zeros(len(target_idx), device=device)
        feats = torch.cat([h_dict["protein"][source_idx], h_dict["go"][target_idx]], dim=-1)
        
        for cat_idx in range(3):
            mask = (target_cats == cat_idx)
            if mask.any():
                out[mask] = self.decoders[cat_idx](feats[mask])
                
        return out

    def forward(self, x_dict, edge_index_dict, edge_label_index, protein_n_id=None, go_node_ids=None):
        states = self.encode(x_dict, edge_index_dict, protein_n_id=protein_n_id)
        return self.decode(states, edge_label_index, go_node_ids=go_node_ids)
