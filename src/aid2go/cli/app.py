import sys
from pathlib import Path
from typing import Optional
import typer
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
    experiment_cmd,
)
from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths
from aid2go.cli.progress import make_progress, run_stage

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
app.add_typer(experiment_cmd.app, name="experiment", help="Controlled model experiments")

@app.command()
def status(config_path: Path = typer.Option("configs/default.yaml", "--config", "-c")):
    """Displays system status, CUDA environment, and cached dataset artifacts."""
    import torch
    from aid2go.core.device import resolve_device

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
    env_table.add_row("PyTorch CUDA Build", str(torch.version.cuda or "None"))
    cuda_available = torch.cuda.is_available()
    env_table.add_row("CUDA Available", "[green]Yes[/green]" if cuda_available else "[yellow]No (CPU mode)[/yellow]")
    env_table.add_row("Configured Device", config.training.device)
    env_table.add_row("Effective Device", str(resolve_device(config.training.device)))
    if cuda_available:
        env_table.add_row("GPU Device", torch.cuda.get_device_name(0))

    console.print(env_table)

    from aid2go.data.cache import stage_spec, stage_current

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
        ("GOA Archive (configured source)", stage_spec("goa", config, processed_dir)[0]["archive"]),
        ("Fetched Protein Sequences", processed_dir / "protein_sequences.tsv"),
        ("Sequence Hashes", processed_dir / "sequence_hashes.tsv"),
        ("GOA Filtered", processed_dir / "goa_filtered.tsv"),
        ("IA Values (PKL)", processed_dir / "ia_values.pkl"),
        ("IA Values (TSV)", processed_dir / "ia_values.tsv"),
        ("Protein Data (HDF5)", processed_dir / "protein_data.h5"),
        ("Protein Index", processed_dir / "protein_index.pkl"),
        ("PPI Edge Index", processed_dir / "ppi_edge_index.pt"),
        ("HeteroData Graph", processed_dir / "hetero_data.pt"),
        ("Dataset Manifest", processed_dir / "dataset_manifest.json"),
        ("Best Model", paths.models / config.tracking.experiment_name / "best_model.pt"),
        ("Last Model", paths.models / config.tracking.experiment_name / "last_model.pt"),
        ("Training History", paths.models / config.tracking.experiment_name / "training_history.csv"),
        ("EDA Summary", paths.reports / "eda" / "summary.json"),
        ("Evaluation Report", paths.reports / "evaluation" / "cafa_evaluation_report.md"),
        ("CAFA Submission", paths.reports / "submissions" / "submission.tsv"),
    ]

    for label, path in artifacts:
        if path.is_file() and path.stat().st_size > 0:
            size_mb = path.stat().st_size / (1024 * 1024)
            status_str = f"[green]✔ Present ({size_mb:.2f} MB)[/green]"
        else:
            status_str = "[red]Empty / not a file[/red]" if path.exists() else "[red]❌ Missing[/red]"
        art_table.add_row(label, str(path.relative_to(paths.root) if path.is_relative_to(paths.root) else path), status_str)

    console.print(art_table)
    cache_table = Table(title="Preprocessing Cache Validity")
    cache_table.add_column("Stage")
    cache_table.add_column("State")
    for stage in ("go", "goa", "proteins", "ppi", "graph"):
        if config.data.dataset_version.startswith("v2"):
            state = "[green]Current fingerprints[/green]" if stage_current(stage, config, processed_dir) else "[yellow]Stale / incomplete[/yellow]"
        else:
            state = "[yellow]Legacy cache: existence checks only[/yellow]"
        cache_table.add_row(stage, state)
    console.print(cache_table)
    console.print("Present reports/checkpoints may belong to earlier runs. Fingerprints check cache inputs and parameters; they do not validate every artifact's contents.")

@app.command()
def pipeline(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation of all steps"),
    num_workers: Optional[int] = typer.Option(None, "--num-workers", "-w"),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size"),
):
    """Executes the complete end-to-end pipeline: Preprocess -> EDA -> Train -> Evaluate -> Predict."""
    console.print(Panel.fit("[bold green]Starting Aid2Go End-to-End Pipeline[/bold green]"))

    config_path = (Path(config_path) if isinstance(config_path, (str, Path))
                   else Path("configs/default.yaml"))
    force = force if isinstance(force, bool) else False
    num_workers = num_workers if isinstance(num_workers, int) else None
    chunk_size = chunk_size if isinstance(chunk_size, int) else None
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)

    with make_progress() as progress:
        preprocess_cmd.all(config_path=config_path, force=force,
                           num_workers=num_workers, chunk_size=chunk_size)
        stages = [
            ("Exploratory analysis", lambda: eda_cmd.report(config_path=config_path)),
            ("Model training", lambda: train_cmd.run(config_path=config_path)),
            ("Model evaluation", lambda: evaluate_cmd.evaluate(
                config_path=config_path, checkpoint_path=best_model_path)),
            ("Submission export", lambda: predict_cmd.predict(
                config_path=config_path, checkpoint_path=best_model_path,
                output_path=None, apply_true_path=True)),
        ]
        tasks = [progress.add_task(label, total=1, detail="pending") for label, _ in stages]
        run_stage(progress, stages[0][0], stages[0][1], task_id=tasks[0])
        run_stage(progress, stages[1][0], stages[1][1], task_id=tasks[1])
        best_model_path = paths.models / config.tracking.experiment_name / "best_model.pt"
        if best_model_path.exists():
            run_stage(progress, stages[2][0], stages[2][1], task_id=tasks[2])
            run_stage(progress, stages[3][0], stages[3][1], task_id=tasks[3])
        else:
            raise RuntimeError(f"Training did not produce a best checkpoint: {best_model_path}")

    console.print(Panel.fit("[bold green]✅ End-to-End Pipeline Finished Successfully![/bold green]"))

if __name__ == "__main__":
    app()
