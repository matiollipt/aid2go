import logging
from pathlib import Path
import pickle
import h5py
import numpy as np
import torch
import faiss

from aid2go.core.config import AppConfig

logger = logging.getLogger(__name__)

def run_preprocess_ppi(config: AppConfig, processed_dir: Path, force: bool = False):
    ppi_path = processed_dir / "ppi_edge_index.pt"
    if not force and ppi_path.exists():
        logger.info(f"[CACHE] PPI edge index already exists at {ppi_path}. Skipping PPI preprocessing. (Use force=True to recompute)")
        print(f"✓ [CACHE] PPI edge index already exists at {ppi_path}. Skipping PPI preprocessing.")
        return

    protein_index_path = processed_dir / "protein_index.pkl"
    h5_path = processed_dir / "protein_data.h5"
    
    if not protein_index_path.exists() or not h5_path.exists():
        logger.error("Protein data not found. Run preprocess-proteins first.")
        return

    with open(protein_index_path, "rb") as f:
        protein_ids = pickle.load(f)
    
    num_proteins = len(protein_ids)
    if num_proteins == 0:
        logger.warning("No proteins found in the index. Creating empty edge index.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        torch.save(edge_index, processed_dir / "ppi_edge_index.pt")
        return

    logger.info(f"Loading protein embeddings from {h5_path} for KNN graph construction...")
    
    # Load all embeddings into a single numpy matrix
    embeddings_list = []
    with h5py.File(h5_path, "r") as h5_file:
        embeddings_group = h5_file["protein_embeddings"]
        for pid in protein_ids:
            embeddings_list.append(embeddings_group[pid][:])
            
    embeddings = np.stack(embeddings_list).astype(np.float32)  # [num_proteins, dim]
    dim = embeddings.shape[1]
    
    logger.info(f"Loaded embeddings shape: {embeddings.shape}. Constructing KNN graph using FAISS...")
    
    k = min(config.data.knn_k, num_proteins - 1)
    if k <= 0:
        logger.warning("KNN k is <= 0 or not enough proteins. Creating empty edge index.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        torch.save(edge_index, processed_dir / "ppi_edge_index.pt")
        return

    if config.data.strict_cosine_similarity:
        logger.info("Using strict cosine similarity (normalized inner product) for KNN search.")
        # L2 normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        feats = embeddings / norms
        index = faiss.IndexFlatIP(dim)
    else:
        logger.info("Using Euclidean L2 distance for KNN search.")
        feats = embeddings
        index = faiss.IndexFlatL2(dim)
        
    index.add(feats)
    
    # We search for k + 1 because the nearest neighbor is always the query itself
    D, I = index.search(feats, k + 1)
    
    edges = []
    for i in range(num_proteins):
        for neighbor_idx in I[i]:
            if neighbor_idx != i and neighbor_idx != -1:
                edges.append([i, neighbor_idx])
                
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    torch.save(edge_index, processed_dir / "ppi_edge_index.pt")
    logger.info(f"KNN graph construction complete. Saved {edge_index.shape[1]} edges to {processed_dir / 'ppi_edge_index.pt'}")
