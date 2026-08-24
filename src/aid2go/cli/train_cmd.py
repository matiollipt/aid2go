from pathlib import Path
import typer
from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths

app = typer.Typer(help="Model training subcommands")

@app.command()
def run(config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file")):
    import torch
    from aid2go.data.splits import create_splits
    from aid2go.training.trainer import Trainer
    from aid2go.training.mlflow_tracker import MLFlowTracker
    from aid2go.training.trackers import NoOpTracker

    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    
    processed_dir = paths.get_processed_dir(config.data.dataset_version)
    graph_path = processed_dir / "hetero_data.pt"
    
    if not graph_path.exists():
        typer.echo(f"Graph data not found at {graph_path}. Run preprocess graph first.")
        raise typer.Exit(code=1)
        
    data = torch.load(graph_path)
    
    # Create splits
    train_data, val_data, test_data = create_splits(data)
    
    if config.tracking.enabled:
        tracker = MLFlowTracker(
            experiment_name=config.tracking.experiment_name,
            uri=config.tracking.mlflow_uri,
            run_name=config.tracking.run_name
        )
    else:
        tracker = NoOpTracker()
        
    tracker.log_params(config.model_dump())
    
    save_dir = paths.models / config.tracking.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = Trainer(config, train_data, tracker, save_dir)
    trainer.fit(train_data, val_data)
    
    tracker.end_run()
