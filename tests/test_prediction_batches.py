"""Batched inference must preserve scores without repeated graph/OBO work."""
from types import SimpleNamespace
from unittest.mock import Mock
import networkx as nx
import pytest
import torch
from torch_geometric.data import HeteroData
from aid2go.core.config import AppConfig
from aid2go.infer.predict import Predictor
from aid2go.modeling.baseline_model import BaselineModel
from aid2go.modeling.degree_attention import DegreeAttentionModel
from aid2go.modeling.smart_model import SmartModel


@pytest.mark.parametrize("name,model_class", [("baseline", BaselineModel),
                         ("degree_attention", DegreeAttentionModel), ("smart", SmartModel)])
def test_inference_encodes_once_and_matches_forward(name, model_class, tmp_path, monkeypatch):
    data = HeteroData()
    data["protein"].x = torch.randn(35, 8)
    data["protein"].node_id = [f"P{i}" for i in range(35)]
    data["go"].x = torch.randn(3, 5)
    data["go"].node_id = ["GO:1", "GO:2", "GO:3"]
    data["protein", "similarity", "protein"].edge_index = torch.tensor([[0, 1], [1, 2]])
    data["protein", "has_function", "go"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data["go", "function_of", "protein"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    model = model_class(data.metadata(), hidden_channels=8, out_channels=8, num_layers=1).eval()
    indices = torch.tensor([34, 0, 12])
    pairs = torch.stack([indices.repeat_interleave(3), torch.arange(3).repeat(3)])
    with torch.no_grad():
        reference = torch.sigmoid(model(data.x_dict, data.edge_index_dict, pairs,
                                        go_node_ids=data["go"].node_id)).reshape(3, 3)
    config = AppConfig()
    config.model.name = name
    predictor = Predictor.__new__(Predictor)
    predictor.config, predictor.data = config, data
    predictor.device, predictor.model = torch.device("cpu"), model
    model.encode = Mock(wraps=model.encode)
    dag = nx.DiGraph([("GO:3", "GO:2"), ("GO:2", "GO:1")])
    load = Mock(return_value=dag)
    monkeypatch.setattr("aid2go.infer.predict.load_go_dag", load)
    obo = tmp_path / "go.obo"
    obo.touch()
    result = predictor.predict_scores(indices, apply_true_path=True, obo_path=obo)
    assert model.encode.call_count == 1
    load.assert_called_once_with(obo)
    for row, idx in enumerate(indices.tolist()):
        expected = reference[row].tolist()
        assert result[f"P{idx}"] == pytest.approx({
            "GO:1": max(expected), "GO:2": max(expected[1:]), "GO:3": expected[2],
        })
    model.encode.reset_mock()
    result = predictor.predict_scores(apply_true_path=False)
    assert len(result) == 35  # Covers more than one decoding batch.
    assert model.encode.call_count == 1


def test_true_path_requires_ontology():
    predictor = Predictor.__new__(Predictor)
    predictor.device = torch.device("cpu")
    data = HeteroData()
    data["protein"].x = torch.zeros(1, 2)
    data["go"].x = torch.zeros(1, 2)
    predictor.data = data
    with pytest.raises(FileNotFoundError, match="OBO"):
        predictor.predict_scores()
