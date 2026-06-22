import torch
from torch_geometric.data import HeteroData

def test_hetero_data_structure():
    data = HeteroData()
    
    # Nodes
    data["protein"].x = torch.randn(10, 16)
    data["go"].x = torch.randn(5, 16)
    
    # Edges
    # protein -> go
    edge_index_goa = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data["protein", "has_function", "go"].edge_index = edge_index_goa
    
    # go -> go
    edge_index_go = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    data["go", "is_a", "go"].edge_index = edge_index_go
    
    assert data["protein"].num_nodes == 10
    assert data["go"].num_nodes == 5
    assert data["protein", "has_function", "go"].num_edges == 2
    assert "is_a" in data.edge_types[1][1]
