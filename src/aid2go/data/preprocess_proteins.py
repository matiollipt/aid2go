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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn

from aid2go.core.config import AppConfig
from aid2go.core.logging import get_logger, log_scientific_stage

logger = get_logger("preprocess_proteins")

class ProteinSequenceDataset(Dataset):
    def __init__(self, sequences_dict: dict, max_length: int):
        self.ids = list(sequences_dict.keys())
        self.sequences = sequences_dict
        self.max_length = max_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        seq_id = self.ids[idx]
        seq = self.sequences[seq_id]
        # Replace rare non-canonical amino acids with X
        seq = re.sub(r"[UZOB]", "X", seq)
        seq = seq[:self.max_length]
        return seq, seq_id

def fetch_sequences(ids: list, batch_size: int = 100) -> dict:
    """
    Fetches amino acid sequences from UniProt REST API in accession chunks with progress tracking.
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    sequences = {}
    total_chunks = (len(ids) + batch_size - 1) // batch_size
    
    logger.info(f"Querying UniProt REST API for {len(ids):,} accessions in {total_chunks:,} chunks...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeRemainingColumn()
    ) as progress:
        task = progress.add_task("Fetching Sequences from UniProt", total=total_chunks)
        
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
                logger.error(f"Failed to fetch accession chunk at index {i}: {e}")
            finally:
                progress.update(task, advance=1)
                
    logger.info(f"Successfully retrieved {len(sequences):,} FASTA sequences from UniProt.")
    return sequences

def parse_pdb_coords(pdb_path: Path):
    """Extracts C-alpha atom 3D coordinates from a PDB structure file."""
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

def build_contact_map(coords, threshold: float = 8.0):
    """Builds spatial contact edge indices between C-alpha atoms within distance threshold."""
    if len(coords) == 0:
        return np.empty((2, 0), dtype=int)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    adj = dist < threshold
    np.fill_diagonal(adj, False)
    edge_index = np.argwhere(adj).T
    return edge_index

def build_sequence_graph(seq_length: int):
    """Builds a sequence-chain contact fallback graph (k-hop backbone adjacency)."""
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
    """
    Encodes protein sequences using ESM-2 transformer model and stores mean-pooled embeddings,
    residue-level tensors, and contact maps into an HDF5 archive.
    """
    device = config.training.device if (config.training.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model_name = config.data.esm_model_name
    
    logger.info(f"Loading ESM-2 transformer '{model_name}' on device '{device}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()
    
    dataset = ProteinSequenceDataset(sequences, config.data.max_sequence_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    h5_path = output_dir / "protein_data.h5"
    
    with h5py.File(h5_path, "w") as h5_file:
        embeddings_group = h5_file.create_group("protein_embeddings")
        residue_group = h5_file.create_group("residue_embeddings")
        edges_group = h5_file.create_group("contact_edges")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task("ESM-2 Protein Embedding Forward Passes", total=len(loader))
            
            with torch.no_grad():
                for seqs, ids in loader:
                    pid = ids[0]
                    seq = seqs[0]
                    
                    inputs = tokenizer(seq, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    outputs = model(**inputs)
                    hidden = outputs.last_hidden_state  # [1, seq_len + 2, emb_dim]
                    
                    # Exclude <cls> and <eos> tokens
                    residue_embs = hidden[0, 1:-1, :].cpu().numpy()  # [seq_len, emb_dim]
                    mean_pooled = residue_embs.mean(axis=0)  # [emb_dim]
                    
                    embeddings_group.create_dataset(pid, data=mean_pooled)
                    residue_group.create_dataset(pid, data=residue_embs)
                    
                    # Search for PDB or build sequence fallback contact graph
                    pdb_dir = Path(config.paths.data_dir) / "raw" / "pdb"
                    pdb_path = pdb_dir / f"{pid}.pdb"
                    
                    coords = parse_pdb_coords(pdb_path)
                    if len(coords) > 0 and len(coords) == len(seq):
                        edge_index = build_contact_map(coords, config.data.c_alpha_distance_threshold)
                    else:
                        edge_index = build_sequence_graph(len(seq))
                        
                    edges_group.create_dataset(pid, data=edge_index)
                    progress.update(task, advance=1)
                    
    # Save the index of processed proteins
    protein_index_path = output_dir / "protein_index.pkl"
    with open(protein_index_path, "wb") as f:
        pickle.dump(list(sequences.keys()), f)
        
    h5_size_mb = h5_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Saved {len(sequences):,} protein ESM-2 embeddings and contact graphs to {h5_path.name} ({h5_size_mb:.2f} MB)"
    )

def run_preprocess_proteins(config: AppConfig, processed_dir: Path, force: bool = False):
    """Executes Step 3 of data pipeline: UniProt Retrieval, Sequence Validation, ESM-2 Embeddings & HDF5 Export."""
    log_scientific_stage(
        stage_num="3/5",
        title="Protein Sequence Retrieval & ESM-2 Evolutionary Embeddings",
        description="Queries UniProt for primary amino acid sequences, validates lengths, and computes ESM-2 evolutionary representation vectors.",
        biological_context="Evolutionary Scale Modeling (ESM-2) uses a masked language modeling objective across millions of protein sequences to capture structural constraints, active site motifs, and biochemical properties directly from primary sequence.",
        mathematical_concept="Residue representations H in R^{L x d} are extracted from the 33-layer transformer. Node representations are obtained via mean pooling: h_p = (1/L) sum_{i=1}^L H_i."
    )

    h5_path = processed_dir / "protein_data.h5"
    index_path = processed_dir / "protein_index.pkl"

    if not force and h5_path.exists() and index_path.exists():
        logger.info(f"[CACHE] Protein artifacts already present in {processed_dir}. Skipping computation. (Use --force to overwrite)")
        return

    goa_path = processed_dir / "goa_filtered.tsv"
    if not goa_path.exists():
        logger.error("goa_filtered.tsv not found in processed directory. Run 'aid2go preprocess goa' first.")
        raise FileNotFoundError("goa_filtered.tsv missing. Run preprocess goa first.")
        
    df = pd.read_csv(goa_path, sep="\t")
    protein_ids = df["DB_Object_ID"].unique().tolist()
    logger.info(f"Loaded {len(protein_ids):,} unique protein IDs from {goa_path.name}")
    
    sequences = fetch_sequences(protein_ids)
    
    # Filter sequences by length constraints
    valid_sequences = {
        pid: seq for pid, seq in sequences.items()
        if len(seq) >= config.data.min_sequence_length
    }
    logger.info(
        f"Filtered valid sequences: {len(valid_sequences):,} passed min_length={config.data.min_sequence_length}"
    )
    
    if not valid_sequences:
        logger.error("No valid protein sequences retrieved! Check internet connection or UniProt accessions.")
        raise ValueError("No valid protein sequences found.")
    
    # Synchronize goa_filtered to keep only valid protein sequences
    valid_pids = set(valid_sequences.keys())
    df_valid = df[df["DB_Object_ID"].isin(valid_pids)].copy()
    df_valid.to_csv(goa_path, sep="\t", index=False)
    logger.info(f"Updated {goa_path.name} with {len(df_valid):,} annotations for verified proteins.")
    
    embed_proteins(valid_sequences, processed_dir, config)
    logger.info("✓ Step 3/5 (Protein Sequences & ESM-2 Embeddings) completed successfully.")
