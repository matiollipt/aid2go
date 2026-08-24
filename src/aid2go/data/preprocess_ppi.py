from pathlib import Path
import pickle
import h5py
import time
import numpy as np
import torch
import faiss

from aid2go.core.config import AppConfig
from aid2go.core.logging import get_logger, log_scientific_stage

logger = get_logger("preprocess_ppi")

def run_preprocess_ppi(config: AppConfig, processed_dir: Path, force: bool = False):
    """Executes Step 4 of data pipeline: FAISS K-Nearest Neighbors Protein Similarity Graph."""
    log_scientific_stage(
        stage_num="4/5",
        title="FAISS K-Nearest Neighbors (KNN) Protein Similarity Network",
        description="Constructs a sequence-similarity/homology network between proteins using FAISS dense vector indexing on ESM-2 embeddings.",
        biological_context="Homologous and functionally related proteins share evolutionary trajectories and structural patterns. In the absence of species-specific experimental PPI data, a k-NN sequence similarity network serves as a functional interaction graph.",
        mathematical_concept="Given ESM-2 embeddings X in R^{N x d}, we build a FAISS index to find the k nearest neighbors under Cosine Similarity: sim(u, v) = (x_u . x_v) / (||x_u|| * ||x_v||), inducing bidirectional edges (u, v) and (v, u)."
    )

    ppi_path = processed_dir / "ppi_edge_index.pt"
    if not force and ppi_path.exists():
        logger.info(f"[CACHE] PPI edge index already exists at {ppi_path}. Skipping computation. (Use --force to overwrite)")
        return

    protein_index_path = processed_dir / "protein_index.pkl"
    h5_path = processed_dir / "protein_data.h5"
    
    if not protein_index_path.exists() or not h5_path.exists():
        logger.error("Protein data or index not found. Run 'aid2go preprocess proteins' first.")
        raise FileNotFoundError("protein_data.h5 or protein_index.pkl missing. Run preprocess proteins first.")

    with open(protein_index_path, "rb") as f:
        protein_ids = pickle.load(f)
    
    num_proteins = len(protein_ids)
    if num_proteins == 0:
        logger.warning("No proteins found in the index. Creating empty edge index.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        torch.save(edge_index, ppi_path)
        return

    logger.info(f"Loading {num_proteins:,} protein ESM-2 embeddings from {h5_path.name}...")
    
    embeddings_list = []
    with h5py.File(h5_path, "r") as h5_file:
        embeddings_group = h5_file["protein_embeddings"]
        for pid in protein_ids:
            embeddings_list.append(embeddings_group[pid][:])
            
    embeddings = np.stack(embeddings_list).astype(np.float32)  # [num_proteins, dim]
    dim = embeddings.shape[1]
    logger.info(f"Loaded embeddings tensor of shape {embeddings.shape} (dim={dim}).")
    
    k = min(config.data.knn_k, num_proteins - 1)
    if k <= 0:
        logger.warning("KNN k is <= 0 or insufficient proteins. Creating empty edge index.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        torch.save(edge_index, ppi_path)
        return

    t0 = time.time()
    if config.data.strict_cosine_similarity:
        logger.info("Normalizing embeddings for Cosine Similarity search (FAISS IndexFlatIP)...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        feats = embeddings / norms
        index = faiss.IndexFlatIP(dim)
    else:
        logger.info("Using Euclidean L2 distance for KNN search (FAISS IndexFlatL2)...")
        feats = embeddings
        index = faiss.IndexFlatL2(dim)
        
    index.add(feats)
    logger.info(f"Built FAISS index with {index.ntotal:,} vectors in {time.time() - t0:.2f}s.")
    
    # Query k + 1 because nearest neighbor is always self
    logger.info(f"Querying {k} nearest neighbors per protein...")
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
        
    torch.save(edge_index, ppi_path)
    
    avg_degree = edge_index.shape[1] / max(num_proteins, 1)
    logger.info(
        f"FAISS KNN graph generated: {edge_index.shape[1]:,} edges across {num_proteins:,} proteins "
        f"(avg degree: {avg_degree:.2f}) -> Saved {ppi_path.name}"
    )
    logger.info("✓ Step 4/5 (FAISS KNN-PPI Network) completed successfully.")
