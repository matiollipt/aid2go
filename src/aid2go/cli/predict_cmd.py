import typer
from pathlib import Path
import torch
from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths
from aid2go.infer.predict import Predictor

app = typer.Typer()

@app.command()
def predict(
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c"),
    checkpoint_path: Path = typer.Option(..., "--checkpoint", "-k")
):
    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    processed_dir = paths.get_processed_dir(config.data.dataset_version)
    
    data = torch.load(processed_dir / "hetero_data.pt")
    
    predictor = Predictor(config, str(checkpoint_path), data)
    typer.echo("Running inference...")
    
    # In a real scenario, we would pass indices to predict
    result = predictor.predict_all()
    
    typer.echo("Inference complete.")