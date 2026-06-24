import torch
import pytest
from torch_geometric.data import HeteroData, Data
from aid2go.modeling.smart_model import SmartModel
from aid2go.explain.explain_links import explain_protein_go_link, get_top_explanatory_nodes

def test_smart_model_node_mode():
    # Build synthetic HeteroData
    data = HeteroData()
    data["protein"].x = torch.randn(5, 1280)  # ESM-2 embedding dimension
    data["go"].x = torch.randn(3, 769)       # BERT embedding + IA dimension
    data["go"].node_id = ["GO:0001", "GO:0002", "GO:0003"]
    data["go"].ia = torch.tensor([[1.5], [2.0], [0.5]])
    
    # Edges
    data["protein", "similarity", "protein"].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    data["go", "is_a", "go"].edge_index = torch.tensor([[0, 1]], dtype=torch.long).t()
    data["protein", "has_function", "go"].edge_index = torch.tensor([[0, 0], [1, 1]], dtype=torch.long).t()
    data["go", "function_of", "protein"].edge_index = torch.tensor([[0, 0], [1, 1]], dtype=torch.long).t()
    
    # Model
    model = SmartModel(
        metadata=data.metadata(),
        hidden_channels=16,
        out_channels=16,
        num_layers=1,
        heads=2,
        go_ids=data["go"].node_id,
        go_id_to_cat={"GO:0001": 0, "GO:0002": 1, "GO:0003": 2},
        protein_representation="node"
    )
    
    # Inference
    edge_label_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    out = model(data.x_dict, data.edge_index_dict, edge_label_index)
    
    assert out.shape == (2,)
    # Output should be valid logits (not nan)
    assert not torch.isnan(out).any()

def test_smart_model_graph_mode():
    # Build synthetic HeteroData
    data = HeteroData()
    data["protein"].x = torch.randn(2, 1280)
    data["go"].x = torch.randn(3, 769)
    data["go"].node_id = ["GO:0001", "GO:0002", "GO:0003"]
    data["go"].ia = torch.tensor([[1.5], [2.0], [0.5]])
    
    # Edges
    data["protein", "similarity", "protein"].edge_index = torch.empty((2, 0), dtype=torch.long)
    data["go", "is_a", "go"].edge_index = torch.empty((2, 0), dtype=torch.long)
    data["protein", "has_function", "go"].edge_index = torch.tensor([[0, 0]], dtype=torch.long).t()
    data["go", "function_of", "protein"].edge_index = torch.tensor([[0, 0]], dtype=torch.long).t()
    
    # Attach protein contact graphs: 2 proteins, each represented as a residue graph
    p1 = Data(x=torch.randn(10, 1280), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t())
    p2 = Data(x=torch.randn(15, 1280), edge_index=torch.tensor([[0, 2]], dtype=torch.long).t())
    data.protein_graphs = [p1, p2]
    
    # Model
    model = SmartModel(
        metadata=data.metadata(),
        hidden_channels=16,
        out_channels=16,
        num_layers=1,
        heads=2,
        go_ids=data["go"].node_id,
        go_id_to_cat={"GO:0001": 0, "GO:0002": 1, "GO:0003": 2},
        protein_representation="graph",
        protein_graphs=data.protein_graphs
    )
    
    # Inference
    edge_label_index = torch.tensor([[0], [0]], dtype=torch.long)
    out = model(data.x_dict, data.edge_index_dict, edge_label_index)
    
    assert out.shape == (1,)
    assert not torch.isnan(out).any()

def test_explainability_smoke():
    data = HeteroData()
    data["protein"].x = torch.randn(3, 1280)
    data["go"].x = torch.randn(3, 769)
    data["go"].node_id = ["GO:0001", "GO:0002", "GO:0003"]
    data["go"].ia = torch.tensor([[1.0], [1.0], [1.0]])
    
    data["protein", "similarity", "protein"].edge_index = torch.tensor([[0, 1]], dtype=torch.long).t()
    data["go", "is_a", "go"].edge_index = torch.empty((2, 0), dtype=torch.long)
    data["protein", "has_function", "go"].edge_index = torch.tensor([[0, 0], [1, 1]], dtype=torch.long).t()
    data["go", "function_of", "protein"].edge_index = torch.tensor([[0, 0], [1, 1]], dtype=torch.long).t()
    
    model = SmartModel(
        metadata=data.metadata(),
        hidden_channels=16,
        out_channels=16,
        num_layers=1,
        heads=2,
        go_ids=data["go"].node_id,
        go_id_to_cat={"GO:0001": 0, "GO:0002": 1, "GO:0003": 2},
        protein_representation="node"
    )
    
    explanation = explain_protein_go_link(model, data, protein_idx=0, go_idx=0, epochs=5)
    top_nodes = get_top_explanatory_nodes(explanation, top_k_nodes=3)
    
    assert len(top_nodes) > 0
