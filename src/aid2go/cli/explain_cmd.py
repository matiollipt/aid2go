from pathlib import Path
import typer
from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths

app = typer.Typer(help="GNN Explainer commands")

@app.command()
def link(
    protein_idx: int = typer.Option(0, "--protein-idx", "-p", help="Local index of protein node to explain"),
    go_idx: int = typer.Option(0, "--go-idx", "-g", help="Local index of GO node to explain"),
    config_path: Path = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config file"),
    model_path: Path = typer.Option(None, "--model-path", "-m", help="Path to trained model weights (.pt)")
):
    import torch
    from aid2go.explain.explain_links import explain_protein_go_link, get_top_explanatory_nodes
    from aid2go.training.trainer import Trainer
    from aid2go.training.trackers import NoOpTracker

    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    
    processed_dir = paths.get_processed_dir(config.data.dataset_version)
    graph_path = processed_dir / "hetero_data.pt"
    
    if not graph_path.exists():
        typer.echo(f"Graph data not found at {graph_path}. Run preprocess graph first.")
        raise typer.Exit(code=1)
        
    data = torch.load(graph_path)
    
    # Initialize trainer to easily build model with same dimensions
    trainer = Trainer(config, data, NoOpTracker(), processed_dir)
    
    # Load model weights
    if model_path is None:
        model_path = paths.models / config.tracking.experiment_name / "best_model.pt"
        if not model_path.exists():
            model_path = paths.models / config.tracking.experiment_name / "last_model.pt"
            
    if model_path.exists():
        typer.echo(f"Loading trained model weights from {model_path}...")
        trainer.model.load_state_dict(torch.load(model_path, map_location=trainer.device))
    else:
        typer.echo("Warning: No trained model weights found. Explaining an untrained model.")
        
    explanation = explain_protein_go_link(trainer.model, data, protein_idx, go_idx, epochs=50)
    top_nodes = get_top_explanatory_nodes(explanation, top_k_nodes=5)
    
    typer.echo("\n--- Explanation Results ---")
    for node_type, nodes in top_nodes.items():
        typer.echo(f"\nNode Type: {node_type}")
        for idx, importance in nodes:
            typer.echo(f"  Node Index: {idx} | Importance: {importance:.4f}")
