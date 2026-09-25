import pytest
import torch
from torch_geometric.data import Data, HeteroData

from aid2go.core.serialization import load_graph, load_model_weights


def test_load_graph_with_residue_subgraphs_weights_only(tmp_path):
    graph = HeteroData()
    graph["protein"].x = torch.ones((1, 4))
    graph["go"].x = torch.ones((1, 4))
    graph["go"].node_id = ["GO:1"]
    graph["protein", "has_function", "go"].edge_index = torch.tensor([[0], [0]])
    graph.protein_graphs = [Data(x=torch.ones((2, 4)),
                                 edge_index=torch.tensor([[0, 1], [1, 0]]))]
    path = tmp_path / "graph.pt"
    torch.save(graph, path)

    loaded = load_graph(path)

    assert isinstance(loaded, HeteroData)
    assert loaded["go"].node_id == ["GO:1"]
    assert loaded.protein_graphs[0].x.shape == (2, 4)


def test_load_model_weights_requires_tensor_state_dict(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"layer.weight": torch.ones((2, 2))}, path)
    assert torch.equal(load_model_weights(path, torch.device("cpu"))["layer.weight"],
                       torch.ones((2, 2)))
    torch.save({"metadata": "not weights"}, path)
    with pytest.raises(TypeError, match="tensor state_dict"):
        load_model_weights(path, torch.device("cpu"))
