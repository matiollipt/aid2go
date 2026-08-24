import json
import time
from pathlib import Path
import torch
import pickle
import h5py
import pandas as pd
from torch_geometric.data import HeteroData, Data

from aid2go.core.config import AppConfig
from aid2go.core.logging import get_logger, log_scientific_stage

logger = get_logger("build_hetero_graph")

def build_hetero_graph(config: AppConfig, processed_dir: Path, force: bool = False) -> HeteroData:
    """Executes Step 5 of data pipeline: Compiles unified PyG HeteroData Graph & Provenance Manifest."""
    log_scientific_stage(
        stage_num="5/5",
        title="PyG HeteroData Bipartite Knowledge Graph Assembly",
        description="Integrates GO semantic BERT embeddings, IA weights, ESM-2 protein embeddings, KNN homology edges, and GAF annotations into a unified PyG HeteroData graph.",
        biological_context="Protein function prediction is formulated as link prediction on a heterogeneous graph G=(V_p cup V_g, E_{p-g} cup E_{p-p} cup E_{g-g}). Message passing aggregates functional context across structural, homology, and ontology relations.",
        mathematical_concept="HeteroData encapsulates: Node Features: X_p in R^{N_p x d_p}, X_g in R^{N_g x (d_g + 1)}; Edge Indices: E_{p -> g}, E_{g -> p} (reverse), E_{p -> p} (PPI), E_{g -> g} (DAG). IA values are bound to GO nodes to dynamically weight focal loss during training."
    )

    graph_path = processed_dir / "hetero_data.pt"
    if not force and graph_path.exists():
        logger.info(f"[CACHE] HeteroData graph already exists at {graph_path}. Skipping compilation. (Use --force to overwrite)")
        return torch.load(graph_path)

    logger.info("Assembling PyG HeteroData graph from processed artifacts...")
    
    # 1. Load GO Data & IA values
    go_emb_path = processed_dir / "go_embeddings.pkl"
    ia_pkl_path = processed_dir / "ia_values.pkl"
    
    with open(go_emb_path, "rb") as f:
        go_embeddings_dict = pickle.load(f)
        
    with open(ia_pkl_path, "rb") as f:
        ia_dict = pickle.load(f)
    
    go_ids = list(go_embeddings_dict.keys())
    go_map = {gid: i for i, gid in enumerate(go_ids)}
    
    go_x = torch.stack([go_embeddings_dict[gid] for gid in go_ids])
    go_ia = torch.tensor([ia_dict.get(gid, 0.0) for gid in go_ids], dtype=torch.float).unsqueeze(-1)
    
    # Concatenate BERT embedding with scalar IA value
    go_features = torch.cat([go_x, go_ia], dim=-1)
    
    # 2. Load Protein Data
    protein_index_path = processed_dir / "protein_index.pkl"
    h5_path = processed_dir / "protein_data.h5"
    
    with open(protein_index_path, "rb") as f:
        protein_ids = pickle.load(f)
    
    protein_map = {pid: i for i, pid in enumerate(protein_ids)}
    num_proteins = len(protein_ids)
    
    protein_tensors = []
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
    
    # If protein representation is "graph", load residue graphs
    if config.data.protein_representation == "graph":
        logger.info("Loading protein residue structural graphs...")
        protein_graphs = []
        with h5py.File(h5_path, "r") as h5_file:
            residue_group = h5_file["residue_embeddings"]
            edges_group = h5_file["contact_edges"]
            for pid in protein_ids:
                x_residues = torch.tensor(residue_group[pid][:], dtype=torch.float)
                edge_index_residues = torch.tensor(edges_group[pid][:], dtype=torch.long)
                
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
        ppi_edge_index = torch.load(ppi_edge_path)
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
        edge_index = torch.tensor(edge_index_goa, dtype=torch.long).t().contiguous()
        data["protein", "has_function", "go"].edge_index = edge_index
        data["go", "function_of", "protein"].edge_index = edge_index.flip(0)
    else:
        data["protein", "has_function", "go"].edge_index = torch.empty((2, 0), dtype=torch.long)
        data["go", "function_of", "protein"].edge_index = torch.empty((2, 0), dtype=torch.long)
        
    torch.save(data, graph_path)
    
    # Generate and save dataset provenance manifest
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_version": config.data.dataset_version,
        "num_proteins": num_proteins,
        "num_go_terms": len(go_ids),
        "protein_feature_dim": protein_x.shape[1],
        "go_feature_dim": go_features.shape[1],
        "edge_counts": {
            "protein_has_function_go": data["protein", "has_function", "go"].edge_index.shape[1],
            "go_function_of_protein": data["go", "function_of", "protein"].edge_index.shape[1],
            "protein_similarity_protein": data["protein", "similarity", "protein"].edge_index.shape[1],
            "go_is_a_go": data["go", "is_a", "go"].edge_index.shape[1]
        },
        "config_snapshot": config.model_dump()
    }
    
    manifest_path = processed_dir / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    logger.info(
        f"Compiled PyG HeteroData Graph:\n"
        f"  - Protein Nodes: {num_proteins:,} (dim={protein_x.shape[1]})\n"
        f"  - GO Nodes: {len(go_ids):,} (dim={go_features.shape[1]} [BERT + IA])\n"
        f"  - Protein->GO Annotation Edges: {data['protein', 'has_function', 'go'].edge_index.shape[1]:,}\n"
        f"  - Protein->Protein PPI Edges: {data['protein', 'similarity', 'protein'].edge_index.shape[1]:,}\n"
        f"  - GO->GO Hierarchy Edges: {data['go', 'is_a', 'go'].edge_index.shape[1]:,}\n"
        f"  -> Saved {graph_path.name} & {manifest_path.name}"
    )
    logger.info("✓ Step 5/5 (HeteroData Assembly) completed successfully.")
    return data
