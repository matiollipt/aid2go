import typer
from pathlib import Path
from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths
from aid2go.data.preprocess_go import run_preprocess_go
from aid2go.data.preprocess_goa import run_preprocess_goa
from aid2go.data.preprocess_proteins import run_preprocess_proteins
from aid2go.data.preprocess_ppi import run_preprocess_ppi
from aid2go.data.build_hetero_graph import build_hetero_graph

app = typer.Typer(help="Data preprocessing subcommands")

@app.command()
def go(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation")
):
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    run_preprocess_go(config, paths.get_processed_dir(config.data.dataset_version), force=force)

@app.command()
def goa(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation")
):
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    run_preprocess_goa(config, paths.get_processed_dir(config.data.dataset_version), force=force)

@app.command()
def proteins(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation")
):
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    run_preprocess_proteins(config, paths.get_processed_dir(config.data.dataset_version), force=force)

@app.command()
def ppi(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation")
):
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    run_preprocess_ppi(config, paths.get_processed_dir(config.data.dataset_version), force=force)

@app.command()
def graph(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation")
):
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    build_hetero_graph(config, paths.get_processed_dir(config.data.dataset_version), force=force)

@app.command()
def all(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation of all steps")
):
    """Run all preprocessing steps in sequence."""
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    processed_dir = paths.get_processed_dir(config.data.dataset_version)

    typer.echo("[1/5] Preprocessing GO subontology & BERT embeddings...")
    run_preprocess_go(config, processed_dir, force=force)

    typer.echo("[2/5] Preprocessing GOA annotations & IA calculation...")
    run_preprocess_goa(config, processed_dir, force=force)

    typer.echo("[3/5] Preprocessing protein sequences & ESM-2 embeddings...")
    run_preprocess_proteins(config, processed_dir, force=force)

    typer.echo("[4/5] Preprocessing PPI similarity network...")
    run_preprocess_ppi(config, processed_dir, force=force)

    typer.echo("[5/5] Building PyG HeteroData graph...")
    build_hetero_graph(config, processed_dir, force=force)

    typer.echo("✅ Complete preprocessing pipeline finished.")
