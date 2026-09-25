import hashlib

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData

from aid2go.core.serialization import load_graph
from aid2go.data.sample import ANNOTATION, REVERSE, SIMILARITY, build_sample
from aid2go.modeling.degree_attention import DegreeCappedGATConv, DegreeAttentionModel
from aid2go.core.config import AppConfig
from aid2go.training.trainer import Trainer
from aid2go.training.trackers import NoOpTracker
from aid2go.eval.protein_holdout import evaluate_holdout
from aid2go.explain.report import save_prediction_explanations


def test_sample_has_exact_sizes_and_no_heldout_annotation_messages(tmp_path):
    source, target = tmp_path / "source", tmp_path / "sample"
    source.mkdir()
    graph = HeteroData()
    graph["protein"].x = torch.randn(30, 8)
    graph["protein"].node_id = [f"P{i:02d}" for i in range(30)]
    graph["go"].x = torch.randn(6, 5)
    graph["go"].node_id = [f"GO:{i:07d}" for i in range(1, 7)]
    pairs = [(p, p % 6) for p in range(30)] + [(p, (p + 1) % 6) for p in range(30)]
    graph[ANNOTATION].edge_index = torch.tensor(pairs, dtype=torch.long).T
    graph[REVERSE].edge_index = graph[ANNOTATION].edge_index.flip(0)
    graph[SIMILARITY].edge_index = torch.empty((2, 0), dtype=torch.long)
    graph["go", "is_a", "go"].edge_index = torch.empty((2, 0), dtype=torch.long)
    torch.save(graph, source / "hetero_data.pt")
    ids = graph["go"].node_id
    pd.DataFrame({"term": ids, "degree": list(range(6, 0, -1)),
                  "namespace": ["biological_process"] * 6}).to_csv(
                      source / "go_nodes.csv", index=False)
    pd.DataFrame(columns=["source", "relationship", "target"]).to_csv(
        source / "go_edges.tsv", sep="\t", index=False)
    (source / "go-basic.obo").write_text("format-version: 1.2\n\n" + "\n".join(
        f"[Term]\nid: {term}\nname: test\nnamespace: biological_process\n"
        for term in ids))
    hashes = [hashlib.sha256(str(i if i != 1 else 0).encode()).hexdigest()
              for i in range(30)]
    pd.DataFrame({"protein_id": graph["protein"].node_id,
                  "sequence_sha256": hashes}).to_csv(
                      source / "sequence_hashes.tsv", sep="\t", index=False)

    manifest = build_sample(source, target, proteins=20, go_terms=3, seed=42,
                            min_support=2, knn_k=3)
    full = load_graph(target / "hetero_data.pt")
    train = load_graph(target / "train_data.pt")
    val = load_graph(target / "val_data.pt")
    test = load_graph(target / "test_data.pt")
    assert full["protein"].num_nodes == 20
    assert full["go"].num_nodes == 3
    assert manifest["train_proteins"] == 16
    assert manifest["val_proteins"] == 2
    assert manifest["test_proteins"] == 2
    assert manifest["cross_split_sequence_hash_overlap"] == 0
    assert len(manifest["selected_go_training_support"]) == 3
    assert min(manifest["selected_go_training_support"].values()) >= 2
    assert all((artifact[ANNOTATION].edge_index[0] < 16).all()
               for artifact in (train, val, test))
    assert all((artifact[REVERSE].edge_index[1] < 16).all()
               for artifact in (train, val, test))
    assert len(set(hashes[graph["protein"].node_id.index(pid)]
                   for pid in full["protein"].node_id)) == 20
    assert full["go"].ia.shape == (3, 1)
    config = AppConfig()
    config.model.hidden_channels = 8
    config.model.num_layers = 1
    config.training.device = "cpu"
    config.training.epochs = 1
    trainer = Trainer(config, train, NoOpTracker(), tmp_path / "models")
    (tmp_path / "models").mkdir()
    trainer.fit(train, val)
    report_dir = tmp_path / "reports"
    result = evaluate_holdout(trainer.model, val, test, full,
                              pd.read_csv(target / "go_nodes.csv"), 16, 2,
                              trainer.device, report_dir, train_data=train,
                              history_path=tmp_path / "models" / "training_history.csv")
    assert (report_dir / "heldout_scores.npz").exists()
    assert (report_dir / "metrics.json").exists()
    assert (report_dir / "precision_recall.png").exists()
    assert (report_dir / "training_curves.png").exists()
    assert "loss_contributions" in result
    arrays = np.load(report_dir / "heldout_scores.npz")
    save_prediction_explanations(trainer.model, test, arrays["scores"],
                                 arrays["truth"], result["validation_selected_threshold"],
                                 18, report_dir / "explanations")
    assert (report_dir / "explanations" / "index.json").exists()


def test_degree_capped_attention_is_order_independent_and_normalized():
    layer = DegreeCappedGATConv(hidden_channels=8, heads=2, budget=2,
                                dropout=0, relation_seed=7)
    edges = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 0, 0]])
    first = layer.capped_edges(edges)
    second = layer.capped_edges(edges[:, torch.tensor([4, 1, 3, 0, 2])])
    assert set(map(tuple, first.T.tolist())) == set(map(tuple, second.T.tolist()))
    assert first.shape[1] == 2
    x = torch.randn(5, 8)
    layer((x, x), edges)
    selected, alpha = layer.last_attention
    assert selected.shape[1] == 2
    assert torch.allclose(alpha.sum(dim=0), torch.ones(2), atol=1e-6)


def test_degree_attention_model_scores_heterogeneous_pairs():
    graph = HeteroData()
    graph["protein"].x = torch.randn(5, 8)
    graph["go"].x = torch.randn(3, 5)
    graph[ANNOTATION].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    graph[REVERSE].edge_index = graph[ANNOTATION].edge_index.flip(0)
    graph[SIMILARITY].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    graph["go", "is_a", "go"].edge_index = torch.tensor([[0, 1], [1, 2]])
    model = DegreeAttentionModel(graph.metadata(), 8, 8, heads=2,
                                 attention_budgets={"has_function": 2})
    scores = model(graph.x_dict, graph.edge_index_dict,
                   torch.tensor([[0, 1], [0, 1]]))
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    attention = model.attention_by_relation()
    assert ANNOTATION in attention[-1]
    selected, weights = attention[-1][ANNOTATION]
    assert selected.shape[1] == weights.shape[0] > 0
