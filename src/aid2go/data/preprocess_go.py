import logging
import json
import os
import tempfile
from pathlib import Path
import pandas as pd
import networkx as nx
import torch
import pickle
import requests
from obonet import read_obo
from transformers import BertTokenizer, BertModel
from aid2go.cli.progress import track_items

from aid2go.core.config import AppConfig
from aid2go.core.device import resolve_device

logger = logging.getLogger(__name__)

def download_go_obo(url: str, output_path: Path, replace: bool = False):
    if output_path.exists() and not replace:
        logger.info(f"GO OBO exists at {output_path}")
        return
    
    logger.info(f"Downloading GO OBO from {url}")
    fd, temporary = tempfile.mkstemp(prefix=".go-basic.", dir=output_path.parent)
    os.close(fd)
    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with open(temporary, "wb") as destination:
                for chunk in response.iter_content(chunk_size=8192):
                    destination.write(chunk)
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def parse_go_dag(obo_path: Path, output_dir: Path, top_n: int):
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
    
    # Calculate degree
    in_degrees = dict(go_dag.in_degree())
    out_degrees = dict(go_dag.out_degree())
    go_df["in_degree"] = go_df["term"].map(in_degrees).fillna(0).astype(int)
    go_df["out_degree"] = go_df["term"].map(out_degrees).fillna(0).astype(int)
    go_df["degree"] = go_df["in_degree"] + go_df["out_degree"]
    
    # Sort and take top N
    go_df = go_df.sort_values(by="degree", ascending=False).head(top_n).copy()
    go_df.to_csv(output_dir / "go_nodes.csv", index=False)
    logger.info(f"Saved top {len(go_df)} GO terms to {output_dir / 'go_nodes.csv'}")
    
    selected_terms = set(go_df["term"])
    
    edges = []
    for u, v, d in go_dag.edges:
        if u in selected_terms and v in selected_terms:
            edges.append({"source": u, "relationship": d, "target": v})
    
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(output_dir / "go_edges.tsv", sep="\t", index=False)
    logger.info(f"Saved {len(edges_df)} GO edges to {output_dir / 'go_edges.tsv'}")
    
    return go_df, edges_df

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]

def embed_go_terms(go_df: pd.DataFrame, output_path: Path, device: str = "cuda"):
    logger.info("Embedding GO terms...")
    device = resolve_device(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()
    
    texts = go_df["definition"].fillna("").tolist()
    ids = go_df["term"].tolist()
    
    dataset = TextDataset(texts)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    embeddings = {}
    idx = 0
    with torch.inference_mode():
        for batch_texts in track_items(loader, "Embedding GO", "batches"):
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(**inputs).last_hidden_state.mean(dim=1)
            for emb in outputs.float().cpu():
                embeddings[ids[idx]] = emb
                idx += 1
                
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info(f"Saved {len(embeddings)} GO embeddings to {output_path}")

def run_preprocess_go(config: AppConfig, processed_dir: Path, force: bool = False):
    processed_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = processed_dir / "go_nodes.csv"
    edges_path = processed_dir / "go_edges.tsv"
    emb_path = processed_dir / "go_embeddings.pkl"
    from aid2go.data.cache import stage_current, record_stage, manifest_path
    
    if not force and nodes_path.exists() and edges_path.exists() and emb_path.exists() and (
        not config.data.dataset_version.startswith("v2") or stage_current("go", config, processed_dir)
    ):
        logger.info(f"[CACHE] GO artifacts already exist in {processed_dir}. Skipping GO preprocessing. (Use force=True to recompute)")
        print(f"✓ [CACHE] GO artifacts already exist in {processed_dir}. Skipping GO preprocessing.")
        return

    obo_path = processed_dir / "go-basic.obo"
    old_manifest = manifest_path(processed_dir, "go")
    previous_url = None
    if old_manifest.exists():
        try:
            previous_url = json.loads(old_manifest.read_text()).get("parameters", {}).get("go_obo_url")
        except (OSError, json.JSONDecodeError):
            pass
    download_go_obo(config.data.go_obo_url, obo_path,
                    replace=previous_url is not None and previous_url != config.data.go_obo_url)
    go_df, _ = parse_go_dag(obo_path, processed_dir, top_n=config.data.top_n_go_terms)
    
    embed_go_terms(go_df, emb_path, device=config.training.device)
    if config.data.dataset_version.startswith("v2"):
        record_stage("go", config, processed_dir)
    logger.info("GO Preprocessing complete.")
