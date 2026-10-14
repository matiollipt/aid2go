from pathlib import Path
import typer
from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths

app = typer.Typer(help="Inference & CAFA submission table export commands")

@app.command()
def predict(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    checkpoint_path: Path = typer.Option(..., "--checkpoint", "-k", help="Path to model checkpoint .pt"),
    output_path: Path = typer.Option(None, "--output", "-o", help="Path to save submission tsv"),
    apply_true_path: bool = typer.Option(True, "--true-path/--no-true-path", help="Apply True-Path Rule score propagation")
):
    from aid2go.core.serialization import load_graph
    import torch
    from aid2go.infer.predict import Predictor
    from aid2go.infer.summaries import export_cafa_submission

    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    processed_dir = paths.get_processed_dir(config.data.dataset_version)
    
    graph_path = processed_dir / "hetero_data.pt"
    if not graph_path.exists():
        typer.echo(f"Graph data not found at {graph_path}. Run preprocess graph first.")
        raise typer.Exit(code=1)

    if config.training.split_strategy == "protein":
        data = load_graph(processed_dir / "test_data.pt")
    else:
        data = load_graph(graph_path)
    obo_path = processed_dir / "go-basic.obo"

    predictor = Predictor(config, str(checkpoint_path), data)
    typer.echo("Running inference over protein nodes...")
    
    predictions = predictor.predict_scores(
        apply_true_path=apply_true_path,
        obo_path=obo_path
    )
    
    if output_path is None:
        output_path = paths.get_report_dir("submissions") / "submission.tsv"

    export_cafa_submission(predictions, output_path, term_order=predictor.go_term_order)
    typer.echo(f"Inference complete. Submission saved to {output_path}")
