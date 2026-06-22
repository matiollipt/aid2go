import logging
from pathlib import Path
import pandas as pd
import torch
import requests
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import pickle
import re

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
        # Replace rare amino acids
        seq = re.sub(r"[UZOB]", "X", seq)
        # Space separated for BERT
        seq = " ".join(list(seq))
        return seq, seq_id

def fetch_sequences(ids: list, batch_size=500) -> dict:
    # Minimal implementation of UniProt fetch
    # In production, use the robust one from legacy code with retries
    url = "https://rest.uniprot.org/uniprotkb/search"
    sequences = {}
    
    # Chunking
    for i in range(0, len(ids), batch_size):
        chunk = ids[i:i+batch_size]
        query = " OR ".join([f"accession:{x}" for x in chunk])
        params = {
            "query": query,
            "format": "fasta",
            "fields": "accession,sequence",
            "size": batch_size
        }
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            
            current_id = None
            current_seq = []
            for line in r.text.splitlines():
                if line.startswith(">"):
                    if current_id:
                        sequences[current_id] = "".join(current_seq)
                    # >sp|P12345|...
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
            logger.error(f"Failed to fetch chunk {i}: {e}")
            
    return sequences

def embed_proteins(sequences: dict, output_dir: Path, config: AppConfig):
    device = config.training.device
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert").to(device)
    model.eval()
    
    dataset = ProteinSequenceDataset(sequences, config.data.max_sequence_length)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    embeddings_dir = output_dir / "protein_embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings_map = {}
    
    with torch.no_grad():
        for seqs, ids in tqdm(loader, desc="Embedding Proteins"):
            inputs = tokenizer(seqs, padding=True, truncation=True, max_length=config.data.max_sequence_length, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            # CLS token or mean pooling? Legacy used mean of last hidden state (without special tokens usually, but legacy code had complex slicing)
            # Legacy code: outputs.last_hidden_state[:, :len(seq)].mean(dim=1) - this handles variable length in batch incorrectly if not careful.
            # ProtBERT CLS token is often used, or mean over non-pad tokens.
            # I will use mean over attention mask.
            
            hidden = outputs.last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden.size()).float()
            sum_hidden = torch.sum(hidden * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_hidden / sum_mask
            
            for i, pid in enumerate(ids):
                emb = mean_pooled[i].cpu()
                torch.save(emb, embeddings_dir / f"{pid}.pt")
                embeddings_map[pid] = str(embeddings_dir / f"{pid}.pt")
                
    with open(output_dir / "protein_index.pkl", "wb") as f:
        pickle.dump(list(embeddings_map.keys()), f)

def run_preprocess_proteins(config: AppConfig, processed_dir: Path):
    # Load GOA to get list of proteins
    goa_path = processed_dir / "goa_filtered.tsv"
    if not goa_path.exists():
        logger.error("GOA file not found. Run preprocess-goa first.")
        return
        
    df = pd.read_csv(goa_path, sep="\t")
    protein_ids = df["DB_Object_ID"].unique().tolist()
    logger.info(f"Found {len(protein_ids)} unique proteins in GOA.")
    
    # Check if we have sequences, if not fetch
    # (In a real scenario, we might have a local fasta source)
    
    # For now, fetch from Uniprot
    sequences = fetch_sequences(protein_ids)
    logger.info(f"Fetched {len(sequences)} sequences.")
    
    embed_proteins(sequences, processed_dir, config)
