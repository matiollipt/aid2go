import logging
from pathlib import Path
import pandas as pd
import torch
import requests
import pickle
import re
import h5py
import numpy as np
from transformers import AutoTokenizer, EsmModel
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from aid2go.core.config import AppConfig

logger = logging.getLogger(__name__)

class ProteinSequenceDataset(Dataset):
    def __init__(self, sequences_dict, max_length):
        self.ids = list(sequences_dict.keys())
        self.sequences = sequences_dict
        self.max_length = max_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        seq_id = self.ids[idx]
        seq = self.sequences[seq_id]
        # Replace rare amino acids with X
        seq = re.sub(r"[UZOB]", "X", seq)
        # Limit sequence to max_length
        seq = seq[:self.max_length]
        return seq, seq_id

def fetch_sequences(ids: list, batch_size=100) -> dict:
    url = "https://rest.uniprot.org/uniprotkb/search"
    sequences = {}
    
    for i in range(0, len(ids), batch_size):
        chunk = ids[i:i+batch_size]
        query = " OR ".join([f"accession:{x}" for x in chunk])
        params = {
            "query": query,
            "format": "fasta",
            "size": batch_size
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            
            current_id = None
            current_seq = []
            for line in r.text.splitlines():
                if line.startswith(">"):
                    if current_id:
                        sequences[current_id] = "".join(current_seq)
                    # Example: >sp|P12345|...
                    parts = line.split("|")
                    if len(parts) > 1:
                        current_id = parts[1]
                    else:
                        current_id = None
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_id:
                sequences[current_id] = "".join(current_seq)
        except Exception as e:
            logger.error(f"Failed to fetch chunk starting at index {i}: {e}")
            
    return sequences

def parse_pdb_coords(pdb_path: Path):
    coords = []
    if not pdb_path.exists():
        return np.empty((0, 3))
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except ValueError:
                    pass
    return np.array(coords)

def build_contact_map(coords, threshold=8.0):
    if len(coords) == 0:
        return np.empty((2, 0), dtype=int)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    adj = dist < threshold
    np.fill_diagonal(adj, False)
    edge_index = np.argwhere(adj).T
    return edge_index

def build_sequence_graph(seq_length):
    edges = []
    for i in range(seq_length):
        for j in range(i + 1, min(i + 3, seq_length)):
            edges.append([i, j])
            edges.append([j, i])
    if edges:
        return np.array(edges).T
    else:
        return np.empty((2, 0), dtype=int)

def embed_proteins(sequences: dict, output_dir: Path, config: AppConfig):
    device = config.training.device
    model_name = config.data.esm_model_name
    
    logger.info(f"Loading ESM-2 model and tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()
    
    dataset = ProteinSequenceDataset(sequences, config.data.max_sequence_length)
    # batch_size=1 since protein sequences have variable lengths and GPU memory usage can be optimized
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    h5_path = output_dir / "protein_data.h5"
    
    # Open HDF5 file
    with h5py.File(h5_path, "w") as h5_file:
        embeddings_group = h5_file.create_group("protein_embeddings")
        residue_group = h5_file.create_group("residue_embeddings")
        edges_group = h5_file.create_group("contact_edges")
        
        with torch.no_grad():
            for seqs, ids in tqdm(loader, desc="Embedding Proteins with ESM-2"):
                pid = ids[0]
                seq = seqs[0]
                
                # Tokenize
                inputs = tokenizer(seq, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                hidden = outputs.last_hidden_state  # [1, seq_len + 2, emb_dim]
                
                # Exclude <cls> and <eos> tokens
                # ESM-2 tokenizer prepends <cls> and appends <eos>
                residue_embs = hidden[0, 1:-1, :].cpu().numpy()  # [seq_len, emb_dim]
                
                # Mean pool for the whole protein embedding
                mean_pooled = residue_embs.mean(axis=0)  # [emb_dim]
                
                # Save embeddings to HDF5
                embeddings_group.create_dataset(pid, data=mean_pooled)
                residue_group.create_dataset(pid, data=residue_embs)
                
                # Build structure or sequence contact graph
                # Search for local structure file (e.g. data/raw/pdb/{pid}.pdb)
                pdb_dir = Path(config.paths.data_dir) / "raw" / "pdb"
                pdb_path = pdb_dir / f"{pid}.pdb"
                
                coords = parse_pdb_coords(pdb_path)
                if len(coords) > 0 and len(coords) == len(seq):
                    edge_index = build_contact_map(coords, config.data.c_alpha_distance_threshold)
                    logger.debug(f"Built contact graph for {pid} using PDB structures.")
                else:
                    edge_index = build_sequence_graph(len(seq))
                    logger.debug(f"Built sequence chain fallback graph for {pid}.")
                    
                edges_group.create_dataset(pid, data=edge_index)
                
    # Save the index of processed proteins
    with open(output_dir / "protein_index.pkl", "wb") as f:
        pickle.dump(list(sequences.keys()), f)
    logger.info(f"Saved all protein data in HDF5 format to {h5_path}")

def run_preprocess_proteins(config: AppConfig, processed_dir: Path):
    goa_path = processed_dir / "goa_filtered.tsv"
    if not goa_path.exists():
        logger.error("GOA file not found. Run preprocess-goa first.")
        return
        
    df = pd.read_csv(goa_path, sep="\t")
    protein_ids = df["DB_Object_ID"].unique().tolist()
    logger.info(f"Found {len(protein_ids)} unique proteins in filtered GOA.")
    
    # Fetch sequences from UniProt
    sequences = fetch_sequences(protein_ids)
    logger.info(f"Fetched {len(sequences)} sequences from UniProt.")
    
    # Filter to only proteins with successfully fetched sequences
    valid_sequences = {pid: seq for pid, seq in sequences.items() if len(seq) >= config.data.min_sequence_length}
    logger.info(f"Valid sequences count: {len(valid_sequences)}")
    
    # Re-save goa_filtered to keep only valid sequences
    valid_pids = set(valid_sequences.keys())
    df_valid = df[df["DB_Object_ID"].isin(valid_pids)].copy()
    df_valid.to_csv(goa_path, sep="\t", index=False)
    
    embed_proteins(valid_sequences, processed_dir, config)
