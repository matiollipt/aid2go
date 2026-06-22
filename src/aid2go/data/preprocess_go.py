import logging
from pathlib import Path
import pandas as pd
import networkx as nx
import torch
import pickle
import requests
from obonet import read_obo
from transformers import BertTokenizer, BertModel
from tqdm.auto import tqdm
from torch_geometric.data import Data, HeteroData

from aid2go.core.config import AppConfig

logger = logging.getLogger(__name__)

def download_go_obo(url: str, output_path: Path):
    if output_path.exists():
        logger.info(f"GO OBO exists at {output_path}")
        return
    
    logger.info(f"Downloading GO OBO from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def parse_go_dag(obo_path: Path, output_dir: Path):
    logger.info("Parsing GO DAG...")
    go_dag = read_obo(str(obo_path), ignore_obsolete=True)
    
    nodes_data = []
    for term, attrs in go_dag.nodes(data=True):
        definition = attrs.get("def", "").split('"', 2)[1] if "def" in attrs else ""
        nodes_data.append({
            "term": term,
            "name": attrs.get("name", ""),
            "namespace": attrs.get("namespace", "unknown"),
            "definition": definition
        })
    
    go_df = pd.DataFrame(nodes_data)
    go_df.to_csv(output_dir / "go_nodes.csv", index=False)
    
    edges = []
    for u, v, d in go_dag.edges:
        edges.append({"source": u, "relationship": d, "target": v})
    
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(output_dir / "go_edges.tsv", sep="\t", index=False)
    
    return go_df, edges_df

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]

def embed_go_terms(go_df: pd.DataFrame, output_path: Path, device: str = "cuda"):
    if output_path.exists():
        logger.info(f"GO embeddings exist at {output_path}")
        return

    logger.info("Embedding GO terms...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()
    
    texts = go_df["definition"].fillna("").tolist()
    ids = go_df["term"].tolist()
    
    dataset = TextDataset(texts)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    embeddings = {}
    idx = 0
    with torch.no_grad():
        for batch_texts in tqdm(loader, desc="Embedding GO"):
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs).last_hidden_state.mean(dim=1)
            for emb in outputs.cpu():
                embeddings[ids[idx]] = emb
                idx += 1
                
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)

def run_preprocess_go(config: AppConfig, processed_dir: Path):
    processed_dir.mkdir(parents=True, exist_ok=True)
    obo_path = processed_dir / "go-basic.obo"
    
    download_go_obo(config.data.go_obo_url, obo_path)
    go_df, _ = parse_go_dag(obo_path, processed_dir)
    
    embed_go_terms(go_df, processed_dir / "go_embeddings.pkl", device=config.training.device)
    
    logger.info("GO Preprocessing complete.")
