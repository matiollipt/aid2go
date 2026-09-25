from pathlib import Path
import typer
from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths
from aid2go.core.serialization import load_graph

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
        
    data = load_graph(data_path)
    report_dir = paths.get_report_dir("eda")
    
    # 1. Degree distribution (PPI)
    similarity_edge = ("protein", "similarity", "protein")
    if similarity_edge in data.edge_index_dict:
        edge_index = data[similarity_edge].edge_index
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
        "num_unique_goa_links": torch.unique(edge_index_goa, dim=1).size(1),
        "num_ppi_edges": data[similarity_edge].edge_index.size(1) if similarity_edge in data.edge_index_dict else 0
    }
    
    with open(report_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    typer.echo(f"Report saved to {report_dir}")
