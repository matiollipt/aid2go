from pathlib import Path
import typer
from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths
from aid2go.cli.progress import make_progress, run_stage

app = typer.Typer(help="Data preprocessing subcommands")


@app.command()
def repair(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c"),
    source_version: str = typer.Option("v1", "--source-version"),
):
    """Build a corrected dataset version from validated cached GO inputs."""
    from aid2go.data.rebuild import seed_validated_go, write_repair_audit
    config = AppConfig.from_yaml(config_path)
    seed_validated_go(config, source_version)
    all(config_path=config_path, force=False)
    typer.echo(f"Repair audit saved to {write_repair_audit(config, source_version)}")


def _stage(config: AppConfig, label: str, runner, artifacts: tuple[str, ...], force: bool):
    from aid2go.data.cache import stage_current
    paths = ProjectPaths(config.paths.root_dir)
    processed_dir = paths.get_processed_dir(config.data.dataset_version)
    stage = {"GO ontology and BERT embeddings": "go",
             "GOA annotations and IA": "goa",
             "Protein sequences and ESM-2 embeddings": "proteins",
             "PPI similarity network": "ppi",
             "PyG HeteroData graph": "graph"}.get(label)
    validate = (lambda: stage_current(stage, config, processed_dir)) if stage and config.data.dataset_version.startswith("v2") else None
    stale = validate is not None and not validate()
    with make_progress() as progress:
        run_stage(progress, label, lambda: runner(config, processed_dir, force=force or stale),
                  artifacts=tuple(processed_dir / name for name in artifacts),
                  force=force or stale, cache_valid=validate)

@app.command()
def go(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation")
):
    from aid2go.data.preprocess_go import run_preprocess_go
    config = AppConfig.from_yaml(config_path)
    _stage(config, "GO ontology and BERT embeddings", run_preprocess_go,
           ("go_nodes.csv", "go_edges.tsv", "go_embeddings.pkl"), force)

@app.command()
def goa(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation")
):
    from aid2go.data.preprocess_goa import run_preprocess_goa
    config = AppConfig.from_yaml(config_path)
    try:
        artifacts = (("goa_filtered.tsv",) if config.data.dataset_version.startswith("v2")
                     else ("goa_filtered.tsv", "ia_values.pkl", "ia_values.tsv"))
        _stage(config, "GOA annotations and IA", run_preprocess_goa, artifacts, force)
    except RuntimeError as error:
        typer.secho(f"GOA preprocessing failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

@app.command()
def proteins(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation")
):
    from aid2go.data.preprocess_proteins import run_preprocess_proteins
    config = AppConfig.from_yaml(config_path)
    _stage(config, "Protein sequences and ESM-2 embeddings", run_preprocess_proteins,
           ("protein_data.h5", "protein_index.pkl", "sequence_hashes.tsv",
            "ia_values.pkl", "ia_values.tsv"), force)

@app.command()
def ppi(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation")
):
    from aid2go.data.preprocess_ppi import run_preprocess_ppi
    config = AppConfig.from_yaml(config_path)
    _stage(config, "PPI similarity network", run_preprocess_ppi,
           ("ppi_edge_index.pt",), force)

@app.command()
def graph(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation")
):
    from aid2go.data.build_hetero_graph import build_hetero_graph
    config = AppConfig.from_yaml(config_path)
    _stage(config, "PyG HeteroData graph", build_hetero_graph,
           ("hetero_data.pt",), force)

@app.command()
def all(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation of all steps")
):
    """Run all preprocessing steps in sequence."""
    from aid2go.data.preprocess_go import run_preprocess_go
    from aid2go.data.preprocess_goa import run_preprocess_goa
    from aid2go.data.preprocess_proteins import run_preprocess_proteins
    from aid2go.data.preprocess_ppi import run_preprocess_ppi
    from aid2go.data.build_hetero_graph import build_hetero_graph

    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    processed_dir = paths.get_processed_dir(config.data.dataset_version)

    stages = [
        ("GO ontology and BERT embeddings", run_preprocess_go,
         ("go_nodes.csv", "go_edges.tsv", "go_embeddings.pkl")),
        ("GOA annotations and IA", run_preprocess_goa,
         (("goa_filtered.tsv",) if config.data.dataset_version.startswith("v2")
          else ("goa_filtered.tsv", "ia_values.pkl", "ia_values.tsv"))),
        ("Protein sequences and ESM-2 embeddings", run_preprocess_proteins,
         ("protein_data.h5", "protein_index.pkl", "sequence_hashes.tsv",
          "ia_values.pkl", "ia_values.tsv")),
        ("PPI similarity network", run_preprocess_ppi, ("ppi_edge_index.pt",)),
        ("PyG HeteroData graph", build_hetero_graph, ("hetero_data.pt",)),
    ]
    with make_progress() as progress:
        from aid2go.data.cache import stage_current
        tasks = [progress.add_task(label, total=1, detail="pending")
                 for label, _, _ in stages]
        for (label, runner, artifacts), task_id in zip(stages, tasks):
            stage = {"GO ontology and BERT embeddings": "go",
                     "GOA annotations and IA": "goa",
                     "Protein sequences and ESM-2 embeddings": "proteins",
                     "PPI similarity network": "ppi",
                     "PyG HeteroData graph": "graph"}.get(label)
            validate = (lambda stage=stage: stage_current(stage, config, processed_dir)) if stage and config.data.dataset_version.startswith("v2") else None
            stale = validate is not None and not validate()
            try:
                run_stage(progress, label,
                          lambda runner=runner, stale=stale: runner(config, processed_dir, force=force or stale),
                          artifacts=tuple(processed_dir / name for name in artifacts),
                          force=force or stale, task_id=task_id, cache_valid=validate)
            except RuntimeError as error:
                typer.secho(f"{label} failed: {error}", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1) from error
