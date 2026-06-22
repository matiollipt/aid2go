import typer
from pathlib import Path
import torch
from torch_geometric.loader import LinkNeighborLoader
from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths
from aid2go.data.splits import create_splits
from aid2go.modeling.registry import get_model_class
from aid2go.training.trainer import Trainer # Reuse evaluate method or split it out
from aid2go.training.trackers import NoOpTracker

app = typer.Typer()

@app.command()
def evaluate(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c"),
    checkpoint_path: Path = typer.Option(..., "--checkpoint", "-k")
):
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    processed_dir = paths.get_processed_dir(config.data.dataset_version)
    
    data = torch.load(processed_dir / "hetero_data.pt")
    _, _, test_data = create_splits(data)
    
    # Setup model
    tracker = NoOpTracker()
    # Trainer can be used just for evaluation context if we refactor, but for now instantiate
    # We need a dummy save dir
    trainer = Trainer(config, data, tracker, paths.models)
    trainer.model.load_state_dict(torch.load(checkpoint_path))
    
    edge_label_index = test_data["protein", "has_function", "go"].edge_label_index
    edge_label = test_data["protein", "has_function", "go"].edge_label
    
    loader = LinkNeighborLoader(
        test_data,
        num_neighbors=[10] * 2,
        edge_label_index=(("protein", "has_function", "go"), edge_label_index),
        edge_label=edge_label,
        batch_size=config.training.batch_size,
        shuffle=False,
        neg_sampling_ratio=1.0
    )
    
    metrics = trainer.evaluate(loader)
    typer.echo(f"Test Metrics: {metrics}")
