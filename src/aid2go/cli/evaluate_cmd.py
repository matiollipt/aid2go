from pathlib import Path
import typer
from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths

app = typer.Typer(help="Certified model evaluation subcommands")

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

    data = torch.load(graph_path)
    _, _, test_data = create_splits(data)
    
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
    trainer.model.load_state_dict(torch.load(checkpoint_path, map_location=trainer.device))
    
    metrics = trainer.evaluate(test_data)
    
    # Display Rich evaluation table
    table = Table(title="Certified CAFA Model Evaluation Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green bold")

    for k, v in metrics.items():
        table.add_row(k.upper(), f"{v:.4f}")

    console.print(table)

    report_dir = paths.get_report_dir("evaluation")
    report_file = cafa_evaluator.generate_report(metrics, report_dir)
    console.print(f"✓ Saved certified evaluation report to {report_file}")
