from pathlib import Path
import typer
from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths

app = typer.Typer(help="Exploratory Data Analysis commands")

@app.command()
def report(config_path: Path = typer.Option("configs/default.yaml", "--config", "-c")):
    import json
    import torch
    from torch_geometric.utils import degree
    from aid2go.viz.eda import plot_ppi_degree, plot_go_frequency

    config = AppConfig.from_yaml(config_path)
    paths = ProjectPaths(config.paths.root_dir)
    processed_dir = paths.get_processed_dir(config.data.dataset_version)
    
    data_path = processed_dir / "hetero_data.pt"
    if not data_path.exists():
        typer.echo("Data not found")
        raise typer.Exit(code=1)
        
    data = torch.load(data_path)
    report_dir = paths.get_report_dir("eda")
    
    # 1. Degree distribution (PPI)
    if ("protein", "interacts", "protein") in data.edge_index_dict:
        edge_index = data["protein", "interacts", "protein"].edge_index
        deg = degree(edge_index[0], num_nodes=data["protein"].num_nodes)
        plot_ppi_degree(deg.numpy(), report_dir / "ppi_degree.png")
        
    # 2. GO term frequency
    edge_index_goa = data["protein", "has_function", "go"].edge_index
    go_deg = degree(edge_index_goa[1], num_nodes=data["go"].num_nodes)
    plot_go_frequency(go_deg.numpy(), report_dir / "go_frequency.png")
    
    summary = {
        "num_proteins": data["protein"].num_nodes,
        "num_go_terms": data["go"].num_nodes,
        "num_goa_edges": edge_index_goa.size(1),
        "num_ppi_edges": data["protein", "interacts", "protein"].edge_index.size(1) if ("protein", "interacts", "protein") in data.edge_index_dict else 0
    }
    
    with open(report_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    typer.echo(f"Report saved to {report_dir}")