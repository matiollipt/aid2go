from pathlib import Path
import typer
from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths
from aid2go.core.serialization import load_graph, load_model_weights

app = typer.Typer(help="Held-out link evaluation subcommands")

@app.command()
def evaluate(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    checkpoint_path: Path = typer.Option(..., "--checkpoint", "-k", help="Path to model checkpoint .pt")
):
    import torch
    import pickle
    import pandas as pd
    from rich.console import Console
    from rich.table import Table
    from aid2go.data.splits import create_splits
    from aid2go.training.trainer import Trainer
    from aid2go.training.trackers import NoOpTracker
    from aid2go.eval.cafa_evaluator import CAFAEvaluator

    console = Console()
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    processed_dir = paths.get_processed_dir(config.data.dataset_version)
    
    graph_path = processed_dir / "hetero_data.pt"
    if not graph_path.exists():
        typer.echo(f"Graph data not found at {graph_path}. Run preprocess graph first.")
        raise typer.Exit(code=1)

    data = load_graph(graph_path)
    if config.training.split_strategy == "protein":
        import json
        from aid2go.eval.protein_holdout import evaluate_holdout
        manifest = json.loads((processed_dir / "split_manifest.json").read_text())
        val_data = load_graph(processed_dir / "val_data.pt")
        test_data = load_graph(processed_dir / "test_data.pt")
        train_data = load_graph(processed_dir / "train_data.pt")
        trainer = Trainer(config, test_data, NoOpTracker(), paths.models)
        trainer.model.load_state_dict(load_model_weights(checkpoint_path, trainer.device))
        report_dir = paths.reports / "experiments" / config.data.dataset_version / config.model.name.value
        evaluate_holdout(trainer.model, val_data, test_data, data,
                         pd.read_csv(processed_dir / "go_nodes.csv"),
                         manifest["train_proteins"], manifest["val_proteins"],
                         trainer.device, report_dir, train_data=train_data)
        console.print(f"Saved protein-held-out report to {report_dir}")
        return
    _, _, test_data = create_splits(data, seed=config.training.seed)
    
    # Load IA values and GO aspect map if available
    ia_path = processed_dir / "ia_values.pkl"
    ia_dict = {}
    if ia_path.exists():
        with open(ia_path, "rb") as f:
            ia_dict = pickle.load(f)

    go_nodes_path = processed_dir / "go_nodes.csv"
    go_aspect_map = {}
    if go_nodes_path.exists():
        go_df = pd.read_csv(go_nodes_path)
        go_aspect_map = dict(zip(go_df["term"], go_df["namespace"]))

    cafa_evaluator = CAFAEvaluator(ia_dict=ia_dict, go_aspect_map=go_aspect_map)

    # Setup trainer & load weights
    trainer = Trainer(config, data, NoOpTracker(), paths.models)
    trainer.model.load_state_dict(load_model_weights(checkpoint_path, trainer.device))
    
    metrics = trainer.evaluate(test_data)
    
    # Display Rich evaluation table
    table = Table(title="Held-out Protein-GO Link Evaluation", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green bold")

    for k, v in metrics.items():
        table.add_row(k.upper(), f"{v:.4f}")

    console.print(table)

    report_dir = paths.get_report_dir("evaluation")
    report_file = cafa_evaluator.generate_report(metrics, report_dir)
    console.print(f"✓ Saved held-out link evaluation report to {report_file}")
