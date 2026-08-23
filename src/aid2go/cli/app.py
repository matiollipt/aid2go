import typer
import sys
from pathlib import Path
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from aid2go.cli import (
    preprocess_cmd,
    train_cmd,
    explain_cmd,
    eda_cmd,
    evaluate_cmd,
    predict_cmd,
)
from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths

app = typer.Typer(
    help="Aid2Go: Protein Function Prediction Framework",
    rich_markup_mode="markdown"
)

console = Console()

app.add_typer(preprocess_cmd.app, name="preprocess", help="Data preprocessing subcommands")
app.add_typer(eda_cmd.app, name="eda", help="Exploratory Data Analysis commands")
app.add_typer(train_cmd.app, name="train", help="Model training subcommands")
app.add_typer(evaluate_cmd.app, name="evaluate", help="Model evaluation commands")
app.add_typer(predict_cmd.app, name="predict", help="Inference & submission table exporter")
app.add_typer(explain_cmd.app, name="explain", help="GNN explainer commands")

@app.command()
def status(config_path: Path = typer.Option("configs/default.yaml", "--config", "-c")):
    """Displays system status, CUDA environment, and cached dataset artifacts."""
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    processed_dir = paths.get_processed_dir(config.data.dataset_version)

    console.print(Panel.fit("[bold blue]Aid2Go Workspace & System Status[/bold blue]"))

    # Environment table
    env_table = Table(title="Environment & Hardware", show_header=True)
    env_table.add_column("Property", style="cyan")
    env_table.add_column("Value", style="magenta")

    env_table.add_row("Python Version", sys.version.split()[0])
    env_table.add_row("PyTorch Version", torch.__version__)
    cuda_available = torch.cuda.is_available()
    env_table.add_row("CUDA Available", "[green]Yes[/green]" if cuda_available else "[yellow]No (CPU mode)[/yellow]")
    if cuda_available:
        env_table.add_row("GPU Device", torch.cuda.get_device_name(0))

    console.print(env_table)

    # Artifacts table
    art_table = Table(title=f"Cached Data Artifacts (Version: {config.data.dataset_version})", show_header=True)
    art_table.add_column("Artifact", style="cyan")
    art_table.add_column("Path", style="dim")
    art_table.add_column("Status", style="bold")

    artifacts = [
        ("GO OBO File", processed_dir / "go-basic.obo"),
        ("GO Nodes", processed_dir / "go_nodes.csv"),
        ("GO Edges", processed_dir / "go_edges.tsv"),
        ("GO Embeddings", processed_dir / "go_embeddings.pkl"),
        ("GOA Filtered", processed_dir / "goa_filtered.tsv"),
        ("IA Values (PKL)", processed_dir / "ia_values.pkl"),
        ("Protein Data (HDF5)", processed_dir / "protein_data.h5"),
        ("Protein Index", processed_dir / "protein_index.pkl"),
        ("PPI Edge Index", processed_dir / "ppi_edge_index.pt"),
        ("HeteroData Graph", processed_dir / "hetero_data.pt"),
    ]

    for label, path in artifacts:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            status_str = f"[green]✔ Present ({size_mb:.2f} MB)[/green]"
        else:
            status_str = "[red]❌ Missing[/red]"
        art_table.add_row(label, str(path.name), status_str)

    console.print(art_table)

@app.command()
def pipeline(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation of all steps")
):
    """Executes the complete end-to-end pipeline: Preprocess -> EDA -> Train -> Evaluate -> Predict."""
    console.print(Panel.fit("[bold green]Starting Aid2Go End-to-End Pipeline[/bold green]"))

    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    processed_dir = paths.get_processed_dir(config.data.dataset_version)

    # 1. Preprocess
    console.print("\n[bold yellow]Step 1/5: Data Preprocessing[/bold yellow]")
    preprocess_cmd.all(config_path=config_path, force=force)

    # 2. EDA
    console.print("\n[bold yellow]Step 2/5: Exploratory Data Analysis[/bold yellow]")
    eda_cmd.report(config_path=config_path)

    # 3. Train
    console.print("\n[bold yellow]Step 3/5: Model Training[/bold yellow]")
    train_cmd.run(config_path=config_path)

    # 4. Evaluate
    best_model_path = paths.models / config.tracking.experiment_name / "best_model.pt"
    if best_model_path.exists():
        console.print("\n[bold yellow]Step 4/5: Model Evaluation[/bold yellow]")
        evaluate_cmd.evaluate(config_path=config_path, checkpoint_path=best_model_path)
    else:
        console.print("[yellow]Skipping evaluation: best_model.pt not found.[/yellow]")

    # 5. Predict
    if best_model_path.exists():
        console.print("\n[bold yellow]Step 5/5: Inference & Submission Table Generation[/bold yellow]")
        predict_cmd.predict(config_path=config_path, checkpoint_path=best_model_path)

    console.print(Panel.fit("[bold green]✅ End-to-End Pipeline Finished Successfully![/bold green]"))

if __name__ == "__main__":
    app()
