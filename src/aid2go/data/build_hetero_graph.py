import logging
import json
import time
from pathlib import Path
import torch
import pickle
import h5py
import pandas as pd
from torch_geometric.data import HeteroData, Data
from aid2go.core.config import AppConfig
from aid2go.core.serialization import load_graph

logger = logging.getLogger(__name__)


def _write_dataset_manifest(config: AppConfig, processed_dir: Path, data: HeteroData) -> None:
    """Keep a readable snapshot alongside the fingerprinted graph artifact."""
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_version": config.data.dataset_version,
        "num_proteins": data["protein"].num_nodes,
        "num_go_terms": data["go"].num_nodes,
        "protein_feature_dim": data["protein"].x.shape[1],
        "go_feature_dim": data["go"].x.shape[1],
        "edge_counts": {"___".join(edge_type): int(data[edge_type].edge_index.shape[1])
                        for edge_type in data.edge_types},
        "config_snapshot": config.model_dump(mode="json"),
    }
    (processed_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

def build_hetero_graph(config: AppConfig, processed_dir: Path, force: bool = False):
    graph_path = processed_dir / "hetero_data.pt"
    from aid2go.data.cache import stage_current, record_stage
    if config.data.dataset_version.startswith("v2") and not all(
        stage_current(stage, config, processed_dir) for stage in ("goa", "proteins", "ppi")
    ):
        raise RuntimeError("Upstream v2 preprocessing is stale; run preprocess all first")
    if not force and graph_path.exists() and (
        not config.data.dataset_version.startswith("v2") or stage_current("graph", config, processed_dir)
    ):
        logger.info(f"[CACHE] HeteroData graph already exists at {graph_path}. Skipping graph construction. (Use force=True to recompute)")
        print(f"✓ [CACHE] HeteroData graph already exists at {graph_path}. Skipping graph construction.")
        data = load_graph(graph_path)
        if not (processed_dir / "dataset_manifest.json").exists():
            _write_dataset_manifest(config, processed_dir, data)
        return data

    logger.info("Building HeteroData graph...")
    
    # 1. Load GO Data
    with open(processed_dir / "go_embeddings.pkl", "rb") as f:
        go_embeddings_dict = pickle.load(f)
        
    # Load IA values
    with open(processed_dir / "ia_values.pkl", "rb") as f:
        ia_dict = pickle.load(f)
    
    go_ids = list(go_embeddings_dict.keys())
    go_map = {gid: i for i, gid in enumerate(go_ids)}
    
    # Node features: concatenate GO embedding and IA value
    go_x = torch.stack([go_embeddings_dict[gid] for gid in go_ids])
    go_ia = torch.tensor([ia_dict.get(gid, 0.0) for gid in go_ids], dtype=torch.float).unsqueeze(-1)
    
    # Node features
    go_features = torch.cat([go_x, go_ia], dim=-1)
    
    # 2. Load Protein Data
    with open(processed_dir / "protein_index.pkl", "rb") as f:
        protein_ids = pickle.load(f)
    
    protein_map = {pid: i for i, pid in enumerate(protein_ids)}
    num_proteins = len(protein_ids)
    
    # Load protein embeddings from HDF5
    protein_tensors = []
    h5_path = processed_dir / "protein_data.h5"
    with h5py.File(h5_path, "r") as h5_file:
        embeddings_group = h5_file["protein_embeddings"]
        for pid in protein_ids:
            protein_tensors.append(torch.tensor(embeddings_group[pid][:], dtype=torch.float))
            
    protein_x = torch.stack(protein_tensors)
    
    data = HeteroData()
    data["go"].x = go_features
    data["go"].node_id = go_ids
    data["go"].ia = go_ia
    
    data["protein"].x = protein_x
    data["protein"].node_id = protein_ids
    
    # If protein representation is "graph", load the structural residue graphs
    if config.data.protein_representation == "graph":
        logger.info("Loading protein residue graphs...")
        protein_graphs = []
        with h5py.File(h5_path, "r") as h5_file:
            residue_group = h5_file["residue_embeddings"]
            edges_group = h5_file["contact_edges"]
            for pid in protein_ids:
                x_residues = torch.tensor(residue_group[pid][:], dtype=torch.float)
                edge_index_residues = torch.tensor(edges_group[pid][:], dtype=torch.long)
                
                # Check for empty edges
                if edge_index_residues.numel() == 0:
                    edge_index_residues = torch.empty((2, 0), dtype=torch.long)
                    
                p_graph = Data(x=x_residues, edge_index=edge_index_residues)
                protein_graphs.append(p_graph)
        data.protein_graphs = protein_graphs
    
    # 3. GO Hierarchy Edges
    go_edges_df = pd.read_csv(processed_dir / "go_edges.tsv", sep="\t")
    edge_index_go = []
    for _, row in go_edges_df.iterrows():
        if row["source"] in go_map and row["target"] in go_map:
            edge_index_go.append([go_map[row["source"]], go_map[row["target"]]])
    
    if edge_index_go:
        data["go", "is_a", "go"].edge_index = torch.tensor(edge_index_go, dtype=torch.long).t().contiguous()
    else:
        data["go", "is_a", "go"].edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # 4. Protein-Protein Similarity Edges
    ppi_edge_path = processed_dir / "ppi_edge_index.pt"
    if ppi_edge_path.exists():
        ppi_edge_index = torch.load(ppi_edge_path, map_location="cpu", weights_only=True)
        if ppi_edge_index.numel() > 0:
            data["protein", "similarity", "protein"].edge_index = ppi_edge_index
        else:
            data["protein", "similarity", "protein"].edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        data["protein", "similarity", "protein"].edge_index = torch.empty((2, 0), dtype=torch.long)
            
    # 5. Protein-GO Annotations (Edges)
    goa_df = pd.read_csv(processed_dir / "goa_filtered.tsv", sep="\t")
    edge_index_goa = []
    for _, row in goa_df.iterrows():
        pid = row["DB_Object_ID"]
        gid = row["GO_ID"]
        if pid in protein_map and gid in go_map:
            edge_index_goa.append([protein_map[pid], go_map[gid]])
            
    if edge_index_goa:
        edge_index = torch.unique(
            torch.tensor(edge_index_goa, dtype=torch.long).t().contiguous(), dim=1
        )
        data["protein", "has_function", "go"].edge_index = edge_index
        data["go", "function_of", "protein"].edge_index = edge_index.flip(0)
    else:
        data["protein", "has_function", "go"].edge_index = torch.empty((2, 0), dtype=torch.long)
        data["go", "function_of", "protein"].edge_index = torch.empty((2, 0), dtype=torch.long)
        
    torch.save(data, processed_dir / "hetero_data.pt")
    _write_dataset_manifest(config, processed_dir, data)
    record_stage("graph", config, processed_dir)
    logger.info(f"HeteroData saved to {processed_dir / 'hetero_data.pt'}")
    return data
