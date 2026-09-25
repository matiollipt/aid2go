import logging
from pathlib import Path
import pickle
import h5py
import numpy as np
import torch
import faiss

from aid2go.core.config import AppConfig
from aid2go.core.device import resolve_device

logger = logging.getLogger(__name__)


def gpu_knn(feats: np.ndarray, k: int, cosine: bool, device: torch.device) -> np.ndarray:
    """Exact, blocked KNN on CUDA; leave space for the driver and other users."""
    vectors = torch.from_numpy(feats).to(device)
    if cosine:
        vectors = torch.nn.functional.normalize(vectors, dim=1)
    norms = (vectors * vectors).sum(dim=1) if not cosine else None
    neighbors = []
    start = 0
    row_limit = len(feats)
    while start < len(feats):
        free_bytes, _ = torch.cuda.mem_get_info(device)
        # Similarity matrix and matmul workspace need headroom.
        rows = max(1, min(row_limit, len(feats) - start,
                          int(free_bytes * 0.2 / (len(feats) * 4))))
        scores = None
        try:
            scores = vectors[start:start + rows] @ vectors.T
            if not cosine:
                scores = -(norms[start:start + rows, None] + norms[None, :] - 2 * scores)
            row_index = torch.arange(rows, device=device)
            scores[row_index, start + row_index] = -torch.inf
            neighbors.append(scores.topk(k, dim=1).indices.cpu().numpy())
            start += rows
        except torch.cuda.OutOfMemoryError:
            if rows == 1:
                raise
            scores = None
            row_limit = max(1, rows // 2)
            torch.cuda.empty_cache()
            logger.warning("CUDA KNN memory exhausted; retrying with %s query rows", row_limit)
    return np.concatenate(neighbors, axis=0)

def run_preprocess_ppi(config: AppConfig, processed_dir: Path, force: bool = False):
    ppi_path = processed_dir / "ppi_edge_index.pt"
    from aid2go.data.cache import stage_current, record_stage
    if config.data.dataset_version.startswith("v2") and not stage_current("proteins", config, processed_dir):
        raise RuntimeError("Protein embeddings or final-cohort IA are stale; run preprocess proteins first")
    if not force and ppi_path.exists() and (
        not config.data.dataset_version.startswith("v2") or stage_current("ppi", config, processed_dir)
    ):
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
        record_stage("ppi", config, processed_dir)
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
    
    logger.info("Loaded embeddings shape: %s. Constructing KNN graph...", embeddings.shape)
    
    k = min(config.data.knn_k, num_proteins - 1)
    if k <= 0:
        logger.warning("KNN k is <= 0 or not enough proteins. Creating empty edge index.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        torch.save(edge_index, processed_dir / "ppi_edge_index.pt")
        record_stage("ppi", config, processed_dir)
        return

    device = resolve_device(config.training.device)
    if device.type == "cuda":
        logger.info("Using exact CUDA KNN search")
        I = gpu_knn(embeddings, k, config.data.strict_cosine_similarity, device)
    elif config.data.strict_cosine_similarity:
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
        
    if device.type == "cpu":
        index.add(feats)
        # CPU FAISS includes the query itself in its search result.
        _, I = index.search(feats, k + 1)
    
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
    record_stage("ppi", config, processed_dir)
    logger.info(f"KNN graph construction complete. Saved {edge_index.shape[1]} edges to {processed_dir / 'ppi_edge_index.pt'}")
