import logging
from pathlib import Path
import torch
import pickle
import pandas as pd
from torch_geometric.data import HeteroData
from aid2go.core.config import AppConfig

logger = logging.getLogger(__name__)

def build_hetero_graph(config: AppConfig, processed_dir: Path):
    logger.info("Building HeteroData graph...")
    
    # 1. Load GO Data
    with open(processed_dir / "go_embeddings.pkl", "rb") as f:
        go_embeddings_dict = pickle.load(f)
    
    go_ids = list(go_embeddings_dict.keys())
    go_map = {gid: i for i, gid in enumerate(go_ids)}
    go_x = torch.stack([go_embeddings_dict[gid] for gid in go_ids])
    
    # 2. Load Protein Data
    with open(processed_dir / "protein_index.pkl", "rb") as f:
        protein_ids = pickle.load(f)
    
    protein_map = {pid: i for i, pid in enumerate(protein_ids)}
    
    # Load all protein tensors (optimized: could use memmap or lazy loading if huge)
    protein_tensors = []
    embeddings_dir = processed_dir / "protein_embeddings"
    for pid in protein_ids:
        protein_tensors.append(torch.load(embeddings_dir / f"{pid}.pt"))
    protein_x = torch.stack(protein_tensors)
    
    data = HeteroData()
    data["go"].x = go_x
    data["go"].node_id = go_ids
    data["protein"].x = protein_x
    data["protein"].node_id = protein_ids
    
    # 3. GO Hierarchy Edges
    go_edges_df = pd.read_csv(processed_dir / "go_edges.tsv", sep="\t")
    edge_index_go = []
    for _, row in go_edges_df.iterrows():
        if row["source"] in go_map and row["target"] in go_map:
            edge_index_go.append([go_map[row["source"]], go_map[row["target"]]])
    
    if edge_index_go:
        data["go", "is_a", "go"].edge_index = torch.tensor(edge_index_go, dtype=torch.long).t().contiguous()
    
    # 4. PPI Edges
    if (processed_dir / "ppi_edge_index.pt").exists():
        ppi_edge_index = torch.load(processed_dir / "ppi_edge_index.pt")
        if ppi_edge_index.numel() > 0:
            data["protein", "interacts", "protein"].edge_index = ppi_edge_index
            
    # 5. Protein-GO Annotations (Edges)
    goa_df = pd.read_csv(processed_dir / "goa_filtered.tsv", sep="\t")
    edge_index_goa = []
    for _, row in goa_df.iterrows():
        pid = row["DB_Object_ID"]
        gid = row["GO_ID"]
        if pid in protein_map and gid in go_map:
            edge_index_goa.append([protein_map[pid], go_map[gid]])
            
    if edge_index_goa:
        edge_index = torch.tensor(edge_index_goa, dtype=torch.long).t().contiguous()
        data["protein", "has_function", "go"].edge_index = edge_index
        
    torch.save(data, processed_dir / "hetero_data.pt")
    logger.info(f"HeteroData saved to {processed_dir / 'hetero_data.pt'}")
    return data
