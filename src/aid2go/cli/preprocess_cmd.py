from pathlib import Path
from typing import Optional
import typer
from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths
from aid2go.core.logging import setup_pipeline_logging, get_logger

app = typer.Typer(help="Data preprocessing subcommands")
logger = get_logger("cli.preprocess")

@app.command()
def go(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation")
):
    """Step 1/5: Download GO OBO, filter top terms, and generate semantic BERT embeddings."""
    from aid2go.data.preprocess_go import run_preprocess_go
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    processed_dir = paths.get_processed_dir(config.data.dataset_version)
    setup_pipeline_logging(processed_dir / "pipeline.log")
    
    run_preprocess_go(config, processed_dir, force=force)

@app.command()
def goa(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation"),
    num_workers: Optional[int] = typer.Option(None, "--num-workers", "-w", help="Number of CPU worker threads"),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", help="Row chunk size for GAF streaming")
):
    """Step 2/5: Download GAF annotations, filter CAFA codes, and compute Information Accretion."""
    from aid2go.data.preprocess_goa import run_preprocess_goa
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    processed_dir = paths.get_processed_dir(config.data.dataset_version)
    setup_pipeline_logging(processed_dir / "pipeline.log")
    
    run_preprocess_goa(config, processed_dir, force=force, num_workers=num_workers, chunk_size=chunk_size)

@app.command()
def proteins(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation")
):
    """Step 3/5: Query UniProt sequences and compute ESM-2 embeddings + contact graphs into HDF5."""
    from aid2go.data.preprocess_proteins import run_preprocess_proteins
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    processed_dir = paths.get_processed_dir(config.data.dataset_version)
    setup_pipeline_logging(processed_dir / "pipeline.log")
    
    run_preprocess_proteins(config, processed_dir, force=force)

@app.command()
def ppi(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation")
):
    """Step 4/5: Build FAISS K-Nearest Neighbors protein sequence similarity network."""
    from aid2go.data.preprocess_ppi import run_preprocess_ppi
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    processed_dir = paths.get_processed_dir(config.data.dataset_version)
    setup_pipeline_logging(processed_dir / "pipeline.log")
    
    run_preprocess_ppi(config, processed_dir, force=force)

@app.command()
def graph(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation")
):
    """Step 5/5: Assemble unified PyG HeteroData bipartite graph and dataset manifest."""
    from aid2go.data.build_hetero_graph import build_hetero_graph
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    processed_dir = paths.get_processed_dir(config.data.dataset_version)
    setup_pipeline_logging(processed_dir / "pipeline.log")
    
    build_hetero_graph(config, processed_dir, force=force)

@app.command()
def all(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation of all steps"),
    num_workers: Optional[int] = typer.Option(None, "--num-workers", "-w", help="Number of CPU worker threads"),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", help="Row chunk size for GAF streaming")
):
    """Run complete end-to-end data preprocessing pipeline (Steps 1 through 5)."""
    from aid2go.data.preprocess_go import run_preprocess_go
    from aid2go.data.preprocess_goa import run_preprocess_goa
    from aid2go.data.preprocess_proteins import run_preprocess_proteins
    from aid2go.data.preprocess_ppi import run_preprocess_ppi
    from aid2go.data.build_hetero_graph import build_hetero_graph

    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    processed_dir = paths.get_processed_dir(config.data.dataset_version)
    
    log_file = processed_dir / "pipeline.log"
    setup_pipeline_logging(log_file)
    logger.info(f"Initialized Aid2Go Preprocessing Pipeline. Companion log: {log_file}")

    run_preprocess_go(config, processed_dir, force=force)
    run_preprocess_goa(config, processed_dir, force=force, num_workers=num_workers, chunk_size=chunk_size)
    run_preprocess_proteins(config, processed_dir, force=force)
    run_preprocess_ppi(config, processed_dir, force=force)
    build_hetero_graph(config, processed_dir, force=force)

    logger.info(f"🎉 Complete preprocessing pipeline finished! All dataset artifacts compiled in {processed_dir}")
