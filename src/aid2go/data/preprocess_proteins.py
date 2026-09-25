import logging
from pathlib import Path
import pandas as pd
import torch
import requests
import pickle
import re
import hashlib
import time
import h5py
import numpy as np
from transformers import AutoConfig, AutoTokenizer, EsmModel
from torch.utils.data import Dataset

from aid2go.core.config import AppConfig
from aid2go.cli.progress import make_progress
from aid2go.core.device import resolve_device

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
    
    total_batches = (len(ids) + batch_size - 1) // batch_size
    with make_progress() as progress:
        task = progress.add_task("UniProt batches", total=total_batches, detail="0 sequences")
        for i in range(0, len(ids), batch_size):
            chunk = ids[i:i+batch_size]
            query = " OR ".join([f"accession:{x}" for x in chunk])
            params = {
                "query": query,
                "format": "fasta",
                "size": batch_size
            }
            for attempt in range(3):
                try:
                    r = requests.get(url, params=params, timeout=(15, 60))
                    r.raise_for_status()
                    break
                except requests.RequestException as error:
                    if attempt == 2:
                        raise RuntimeError(
                            f"UniProt batch {i // batch_size + 1}/{total_batches} failed after 3 attempts"
                        ) from error
                    time.sleep(2 ** attempt)
            current_id = None
            current_seq = []
            for line in r.text.splitlines():
                if line.startswith(">"):
                    if current_id:
                        sequences[current_id] = "".join(current_seq)
                    parts = line.split("|")
                    current_id = parts[1] if len(parts) > 1 else None
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_id:
                sequences[current_id] = "".join(current_seq)
            progress.update(task, advance=1,
                            detail=f"{len(sequences):,} sequences · {min(i + batch_size, len(ids)):,}/{len(ids):,} IDs")
            
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
    device = resolve_device(config.training.device)
    model_name = config.data.esm_model_name
    model_config = AutoConfig.from_pretrained(model_name)
    max_residues = min(config.data.max_sequence_length,
                       model_config.max_position_embeddings - 2)
    if max_residues < config.data.max_sequence_length:
        logger.warning("ESM supports at most %s residues; longer proteins will be truncated", max_residues)
    dataset = ProteinSequenceDataset(sequences, max_residues)
    order = sorted(range(len(dataset)), key=lambda i: len(dataset.sequences[dataset.ids[i]]))
    batch_limit = config.data.embedding_batch_size if device.type == "cuda" else 1
    token_budget = config.data.embedding_token_budget
    amp = device.type == "cuda" and config.training.mixed_precision
    h5_path = output_dir / "protein_data.h5"
    temp_path = output_dir / "protein_data.h5.tmp"
    pdb_dir = Path(config.paths.data_dir) / "raw" / "pdb"
    source_path = None
    if config.data.reuse_protein_data_path:
        source_path = Path(config.data.reuse_protein_data_path)
        if not source_path.is_absolute():
            source_path = Path(__file__).resolve().parents[3] / source_path
    source_file = h5py.File(source_path, "r") if source_path and source_path.exists() else None

    try:
      with h5py.File(temp_path, "w") as h5_file:
        embeddings_group = h5_file.create_group("protein_embeddings")
        residue_group = h5_file.create_group("residue_embeddings")
        edges_group = h5_file.create_group("contact_edges")
        reusable = set()
        if source_file is not None:
            old_embeddings = source_file.get("protein_embeddings")
            old_residues = source_file.get("residue_embeddings")
            old_edges = source_file.get("contact_edges")
            if all(group is not None for group in (old_embeddings, old_residues, old_edges)):
                for pid in dataset.ids:
                    sequence = dataset.sequences[pid]
                    expected_length = min(len(sequence), max_residues)
                    if (pid in old_embeddings and pid in old_residues and pid in old_edges
                            and old_embeddings[pid].shape == (model_config.hidden_size,)
                            and old_residues[pid].shape == (expected_length, model_config.hidden_size)):
                        for group_name in ("protein_embeddings", "residue_embeddings", "contact_edges"):
                            source_file.copy(source_file[group_name][pid], h5_file[group_name], name=pid)
                        reusable.add(pid)
        logger.info("Reused %s legacy protein embeddings after ID, model dimension, and length checks; "
                    "%s proteins need new ESM embeddings", len(reusable), len(dataset) - len(reusable))
        pending_order = [i for i in order if dataset.ids[i] not in reusable]
        tokenizer = model = None
        if pending_order:
            logger.info("Loading ESM-2 model and tokenizer: %s", model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = EsmModel.from_pretrained(model_name, add_pooling_layer=False).to(device)
            model.eval()
            actual_limit = min(config.data.max_sequence_length,
                               model.config.max_position_embeddings - 2)
            if actual_limit != max_residues:
                if reusable:
                    raise ValueError("Cached ESM position limit differs from loaded model")
                dataset.max_length = actual_limit
        position = 0
        with make_progress() as progress, torch.inference_mode():
            task = progress.add_task("Embedding proteins", total=len(pending_order), detail=str(device))
            while position < len(pending_order):
                count = min(batch_limit, len(pending_order) - position)
                # Length sorting reduces padding; the token budget bounds attention memory.
                longest = len(dataset[pending_order[position + count - 1]][0]) + 2
                count = min(count, max(1, token_budget // longest))
                batch = [dataset[i] for i in pending_order[position:position + count]]
                seqs, ids = zip(*batch)
                try:
                    inputs = tokenizer(list(seqs), padding=True, return_tensors="pt")
                    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
                    with torch.autocast(device_type=device.type, enabled=amp):
                        hidden = model(**inputs).last_hidden_state
                    # Pool on the accelerator and write float32 for stable downstream data.
                    lengths = [len(seq) for seq in seqs]
                    residues = [hidden[i, 1:length + 1].float().cpu().numpy()
                                for i, length in enumerate(lengths)]
                    pooled = [hidden[i, 1:length + 1].float().mean(dim=0).cpu().numpy()
                              for i, length in enumerate(lengths)]
                    del hidden, inputs
                except torch.cuda.OutOfMemoryError:
                    if device.type != "cuda" or count == 1:
                        raise
                    inputs = None
                    batch_limit = max(1, count // 2)
                    logger.warning("CUDA memory exhausted; retrying ESM with batch size %s", batch_limit)
                    torch.cuda.empty_cache()
                    continue

                for pid, seq, residue_embs, mean_pooled in zip(ids, seqs, residues, pooled):
                    embeddings_group.create_dataset(pid, data=mean_pooled)
                    residue_group.create_dataset(pid, data=residue_embs)
                    coords = parse_pdb_coords(pdb_dir / f"{pid}.pdb")
                    if len(coords) == len(seq) and len(coords) > 0:
                        edge_index = build_contact_map(coords, config.data.c_alpha_distance_threshold)
                    else:
                        edge_index = build_sequence_graph(len(seq))
                    edges_group.create_dataset(pid, data=edge_index)
                position += count
                progress.update(task, advance=count, detail=f"{position:,}/{len(pending_order):,} · {device} · batch {count}")
        h5_file.attrs["esm_model_name"] = model_name
        h5_file.attrs["legacy_reused_count"] = len(reusable)
        h5_file.attrs["legacy_validation"] = "accession, ESM dimension, residue length; legacy sequence hashes unavailable"
      temp_path.replace(h5_path)
    finally:
      if source_file is not None:
          source_file.close()
    # Save the index of processed proteins
    with open(output_dir / "protein_index.pkl", "wb") as f:
        pickle.dump(list(sequences.keys()), f)
    hashes = []
    for index, pid in enumerate(dataset.ids):
        normalized = dataset[index][0]
        hashes.append((pid, hashlib.sha256(normalized.encode("ascii")).hexdigest(), len(normalized)))
    pd.DataFrame(hashes, columns=["protein_id", "sequence_sha256", "residues"]).to_csv(
        output_dir / "sequence_hashes.tsv", sep="\t", index=False)
    logger.info(f"Saved all protein data in HDF5 format to {h5_path}")

def run_preprocess_proteins(config: AppConfig, processed_dir: Path, force: bool = False):
    h5_path = processed_dir / "protein_data.h5"
    index_path = processed_dir / "protein_index.pkl"
    hashes_path = processed_dir / "sequence_hashes.tsv"
    from aid2go.data.cache import stage_current, record_stage

    if not force and h5_path.exists() and index_path.exists() and (
        not config.data.dataset_version.startswith("v2") or
        (hashes_path.exists() and stage_current("proteins", config, processed_dir))
    ):
        logger.info(f"[CACHE] Protein artifacts already exist in {processed_dir}. Skipping protein preprocessing. (Use force=True to recompute)")
        print(f"✓ [CACHE] Protein artifacts already exist in {processed_dir}. Skipping protein preprocessing.")
        return

    goa_path = processed_dir / "goa_filtered.tsv"
    if not goa_path.exists():
        logger.error("GOA file not found. Run preprocess-goa first.")
        return
        
    df = pd.read_csv(goa_path, sep="\t")
    protein_ids = df["DB_Object_ID"].unique().tolist()
    logger.info(f"Found {len(protein_ids)} unique proteins in filtered GOA.")
    
    # Cache the fetched FASTA sequences independently of the expensive ESM pass.
    sequences_path = processed_dir / "protein_sequences.tsv"
    if sequences_path.exists() and not force:
        cached = pd.read_csv(sequences_path, sep="\t", dtype=str)
        if set(cached.protein_id) == set(protein_ids):
            sequences = dict(zip(cached.protein_id, cached.sequence))
        else:
            sequences = fetch_sequences(protein_ids)
    else:
        sequences = fetch_sequences(protein_ids)
    pd.DataFrame([(pid, sequences[pid]) for pid in protein_ids if pid in sequences],
                 columns=["protein_id", "sequence"]).to_csv(sequences_path,
                                                              sep="\t", index=False)
    logger.info(f"Fetched {len(sequences)} sequences from UniProt.")
    
    # Filter to only proteins with successfully fetched sequences
    valid_sequences = {pid: seq for pid, seq in sequences.items() if len(seq) >= config.data.min_sequence_length}
    logger.info(f"Valid sequences count: {len(valid_sequences)}")
    
    # Re-save goa_filtered to keep only valid sequences
    valid_pids = set(valid_sequences.keys())
    df_valid = df[df["DB_Object_ID"].isin(valid_pids)].copy()
    df_valid.to_csv(goa_path, sep="\t", index=False)
    
    embed_proteins(valid_sequences, processed_dir, config)
    from aid2go.data.preprocess_goa import save_ia_from_final_cohort
    save_ia_from_final_cohort(processed_dir)
    record_stage("goa", config, processed_dir)
    record_stage("proteins", config, processed_dir)
