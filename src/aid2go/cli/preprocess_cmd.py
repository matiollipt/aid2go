import typer
from pathlib import Path
from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths
from aid2go.data.preprocess_go import run_preprocess_go
from aid2go.data.preprocess_goa import run_preprocess_goa
from aid2go.data.preprocess_proteins import run_preprocess_proteins
from aid2go.data.preprocess_ppi import run_preprocess_ppi
from aid2go.data.build_hetero_graph import build_hetero_graph

app = typer.Typer()

@app.command()
def go(config_path: Path = typer.Option("configs/default.yaml", "--config", "-c")):
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    run_preprocess_go(config, paths.get_processed_dir(config.data.dataset_version))

@app.command()
def goa(config_path: Path = typer.Option("configs/default.yaml", "--config", "-c")):
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    run_preprocess_goa(config, paths.get_processed_dir(config.data.dataset_version))

@app.command()
def proteins(config_path: Path = typer.Option("configs/default.yaml", "--config", "-c")):
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    run_preprocess_proteins(config, paths.get_processed_dir(config.data.dataset_version))

@app.command()
def ppi(config_path: Path = typer.Option("configs/default.yaml", "--config", "-c")):
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    run_preprocess_ppi(config, paths.get_processed_dir(config.data.dataset_version))

@app.command()
def graph(config_path: Path = typer.Option("configs/default.yaml", "--config", "-c")):
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    build_hetero_graph(config, paths.get_processed_dir(config.data.dataset_version))
