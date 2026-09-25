from aid2go.cli import app, preprocess_cmd
from aid2go.core.config import AppConfig


def test_preprocessing_lists_cached_and_completed_stages_once(tmp_path, monkeypatch, capsys):
    config = AppConfig()
    config.paths.root_dir = str(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"paths:\n  root_dir: {tmp_path}\n")
    processed = tmp_path / "data" / "processed" / config.data.dataset_version
    processed.mkdir(parents=True)
    for name in ("go_nodes.csv", "go_edges.tsv", "go_embeddings.pkl"):
        (processed / name).write_text("cached")

    outputs = {
        "goa": ("goa_filtered.tsv", "ia_values.pkl", "ia_values.tsv"),
        "proteins": ("protein_data.h5", "protein_index.pkl", "sequence_hashes.tsv"),
        "ppi": ("ppi_edge_index.pt",),
        "graph": ("hetero_data.pt",),
    }
    called = []

    def fake_runner(stage):
        def run(config, target, force=False):
            called.append(stage)
            for name in outputs[stage]:
                (target / name).write_text("done")
        return run

    monkeypatch.setattr("aid2go.data.preprocess_go.run_preprocess_go",
                        lambda *args, **kwargs: called.append("go"))
    monkeypatch.setattr("aid2go.data.preprocess_goa.run_preprocess_goa", fake_runner("goa"))
    monkeypatch.setattr("aid2go.data.preprocess_proteins.run_preprocess_proteins", fake_runner("proteins"))
    monkeypatch.setattr("aid2go.data.preprocess_ppi.run_preprocess_ppi", fake_runner("ppi"))
    monkeypatch.setattr("aid2go.data.build_hetero_graph.build_hetero_graph", fake_runner("graph"))

    preprocess_cmd.all(config_path=config_path, force=False)

    output = capsys.readouterr().out
    for label in ("GO ontology and BERT embeddings", "GOA annotations and IA",
                  "Protein sequences and ESM-2 embeddings", "PPI similarity network",
                  "PyG HeteroData graph"):
        assert output.count(label) == 1
    assert "cached" in output
    assert "Preprocessing" not in output
    assert called == ["goa", "proteins", "ppi", "graph"]


def test_pipeline_has_stage_rows_without_duplicate_pipeline_bar(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"paths:\n  root_dir: {tmp_path}\n")
    events = []

    monkeypatch.setattr(app.preprocess_cmd, "all", lambda **kwargs: events.append("preprocess"))
    monkeypatch.setattr(app.eda_cmd, "report", lambda **kwargs: events.append("eda"))

    def train(**kwargs):
        events.append("train")
        model_dir = tmp_path / "models" / "aid2go"
        model_dir.mkdir(parents=True)
        (model_dir / "best_model.pt").touch()

    monkeypatch.setattr(app.train_cmd, "run", train)
    monkeypatch.setattr(app.evaluate_cmd, "evaluate", lambda **kwargs: events.append("evaluate"))
    monkeypatch.setattr(app.predict_cmd, "predict", lambda **kwargs: events.append("predict"))

    app.pipeline(config_path=config_path, force=False)

    output = capsys.readouterr().out
    for label in ("Exploratory analysis", "Model training", "Model evaluation", "Submission export"):
        assert output.count(label) == 1
    assert "│ Pipeline " not in output
    assert events == ["preprocess", "eda", "train", "evaluate", "predict"]
