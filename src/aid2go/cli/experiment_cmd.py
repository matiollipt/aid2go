"""Reproducible sample comparison of the baseline and degree-attention models."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths

app = typer.Typer(help="Controlled protein-held-out experiments")


@app.command()
def sample(
    baseline_config: Path = typer.Option("configs/default.yaml", "--baseline-config"),
    attention_config: Path = typer.Option("configs/degree_attention.yaml", "--attention-config"),
    proteins: int = typer.Option(1000, "--proteins"),
    go_terms: int = typer.Option(100, "--go-terms"),
    seed: int = typer.Option(42, "--seed"),
    force_sample: bool = typer.Option(False, "--force-sample"),
):
    """Build a label-safe sample and save both model runs and diagnostics."""
    import numpy as np
    import pandas as pd
    from aid2go.core.seed import seed_everything
    from aid2go.core.serialization import load_graph
    from aid2go.data.sample import build_sample, _sha256
    from aid2go.eval.protein_holdout import evaluate_holdout
    from aid2go.training.trackers import NoOpTracker
    from aid2go.training.trainer import Trainer

    baseline = AppConfig.from_yaml(baseline_config)
    attention = AppConfig.from_yaml(attention_config)
    if baseline.model.name != "baseline" or attention.model.name != "degree_attention":
        raise typer.BadParameter("Expected baseline and degree_attention model configs")
    if baseline.data.dataset_version != attention.data.dataset_version:
        raise typer.BadParameter("Both configs must point to the same repaired dataset version")
    if baseline.model.hidden_channels != attention.model.hidden_channels or baseline.model.num_layers != attention.model.num_layers:
        raise typer.BadParameter("Both models must use the same hidden size and layer count")
    if (baseline.data.knn_k != attention.data.knn_k or
            baseline.data.strict_cosine_similarity != attention.data.strict_cosine_similarity):
        raise typer.BadParameter("Both models must use the same similarity graph settings")
    paths = ProjectPaths(baseline.paths.root_dir)
    source = paths.processed / baseline.data.dataset_version
    required = [source / name for name in ("hetero_data.pt", "sequence_hashes.tsv",
                                            "protein_data.h5", "go_nodes.csv", "go-basic.obo")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise typer.BadParameter("Repaired preprocessing is incomplete: " + ", ".join(missing))
    sample_version = f"{source.name}_sample_{proteins}x{go_terms}_seed{seed}"
    target = paths.processed / sample_version
    manifest_path = target / "split_manifest.json"
    if manifest_path.exists() and not force_sample:
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("schema_version") != 2
                or manifest.get("source_graph_sha256") != _sha256(source / "hetero_data.pt")
                or manifest.get("proteins") != proteins or manifest.get("go_terms") != go_terms
                or manifest.get("seed") != seed
                or manifest.get("knn_k") != baseline.data.knn_k
                or manifest.get("strict_cosine_similarity") != baseline.data.strict_cosine_similarity):
            raise typer.BadParameter("Cached sample is stale; rerun with --force-sample")
        typer.echo(f"Using cached sample {target}")
    else:
        manifest = build_sample(source, target, proteins, go_terms, seed,
                                knn_k=baseline.data.knn_k,
                                strict_cosine_similarity=baseline.data.strict_cosine_similarity)
        typer.echo(f"Saved sample and split manifest to {target}")
    source_graph = load_graph(source / "hetero_data.pt")
    sample_graph = load_graph(target / "hetero_data.pt")
    go_nodes = pd.read_csv(target / "go_nodes.csv")
    report_root = paths.reports / "experiments" / sample_version
    eda_dir = report_root / "eda"
    from aid2go.viz.report import generate_eda_report
    generate_eda_report(sample_graph, source_graph, go_nodes,
                        source / "protein_data.h5", eda_dir, seed)
    from aid2go.explain.report import save_prediction_explanations
    comparison = []
    for name, initial in (("baseline", baseline), ("degree_attention", attention)):
        config = initial.model_copy(deep=True)
        config.data.dataset_version = sample_version
        config.training.seed = seed
        config.training.split_strategy = "protein"
        config.training.loss = "bce"
        config.training.epochs = 5
        config.tracking.enabled = False
        config.tracking.experiment_name = f"{sample_version}_{name}"
        run_dir = paths.models / config.tracking.experiment_name
        run_dir.mkdir(parents=True, exist_ok=True)
        config.save_yaml(run_dir / "resolved_config.yaml")
        seed_everything(seed)
        train = load_graph(target / "train_data.pt")
        val = load_graph(target / "val_data.pt")
        test = load_graph(target / "test_data.pt")
        trainer = Trainer(config, train, NoOpTracker(), run_dir)
        trainer.fit(train, val)
        from aid2go.core.serialization import load_model_weights
        trainer.model.load_state_dict(load_model_weights(run_dir / "best_model.pt", trainer.device))
        model_report = report_root / name
        result = evaluate_holdout(trainer.model, val, test, sample_graph,
                                  go_nodes, manifest["train_proteins"],
                                  manifest["val_proteins"], trainer.device, model_report,
                                  train_data=train,
                                  history_path=run_dir / "training_history.csv")
        scores = np.load(model_report / "heldout_scores.npz")
        save_prediction_explanations(
            trainer.model, test.to(trainer.device), scores["scores"], scores["truth"],
            result["validation_selected_threshold"],
            manifest["train_proteins"] + manifest["val_proteins"],
            model_report / "explanations",
        )
        summary = {"model": name, **result["overall"]}
        comparison.append(summary)
        typer.echo(f"Saved {name} checkpoints to {run_dir} and reports to {model_report}")
    pd.DataFrame(comparison).to_csv(report_root / "comparison.tsv", sep="\t", index=False)
    (report_root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    typer.echo(f"Comparison complete: {report_root}")
