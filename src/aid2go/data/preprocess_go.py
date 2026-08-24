from pathlib import Path
import pandas as pd
import networkx as nx
import torch
import pickle
from obonet import read_obo
from transformers import BertTokenizer, BertModel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn

from aid2go.core.config import AppConfig
from aid2go.core.logging import get_logger, log_scientific_stage
from aid2go.data.downloads import download_file_resumable

logger = get_logger("preprocess_go")

def download_go_obo(url: str, output_path: Path, force: bool = False):
    """Downloads the Gene Ontology OBO file with resumable HTTP range support."""
    download_file_resumable(
        url=url,
        target_path=output_path,
        force=force,
        desc="GO OBO Definition (go-basic.obo)"
    )

def parse_go_dag(obo_path: Path, output_dir: Path, top_n: int):
    """
    Parses the Gene Ontology Directed Acyclic Graph (DAG), calculates graph degrees,
    and filters to the top N terms to establish canonical semantic anchors.
    """
    logger.info(f"Parsing GO DAG structure from {obo_path}...")
    go_dag = read_obo(str(obo_path), ignore_obsolete=True)
    
    total_obo_terms = go_dag.number_of_nodes()
    total_obo_edges = go_dag.number_of_edges()
    logger.info(f"Parsed GO DAG: {total_obo_terms:,} total active terms, {total_obo_edges:,} total relations.")

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
    
    # Calculate degree connectivity (in-degree + out-degree in DAG)
    in_degrees = dict(go_dag.in_degree())
    out_degrees = dict(go_dag.out_degree())
    go_df["in_degree"] = go_df["term"].map(in_degrees).fillna(0).astype(int)
    go_df["out_degree"] = go_df["term"].map(out_degrees).fillna(0).astype(int)
    go_df["degree"] = go_df["in_degree"] + go_df["out_degree"]
    
    # Filter to top N terms by degree
    go_df = go_df.sort_values(by="degree", ascending=False).head(top_n).copy()
    go_nodes_path = output_dir / "go_nodes.csv"
    go_df.to_csv(go_nodes_path, index=False)
    
    ns_counts = go_df["namespace"].value_counts().to_dict()
    logger.info(
        f"Filtered top {len(go_df):,} GO terms by graph degree. "
        f"Breakdown by Subontology: {ns_counts}. Saved -> {go_nodes_path.name}"
    )
    
    selected_terms = set(go_df["term"])
    edges = []
    for u, v, d in go_dag.edges:
        if u in selected_terms and v in selected_terms:
            edges.append({"source": u, "relationship": d, "target": v})
    
    edges_df = pd.DataFrame(edges)
    go_edges_path = output_dir / "go_edges.tsv"
    edges_df.to_csv(go_edges_path, sep="\t", index=False)
    logger.info(f"Extracted {len(edges_df):,} structural GO DAG edges between selected terms -> {go_edges_path.name}")
    
    return go_df, edges_df

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]

def embed_go_terms(go_df: pd.DataFrame, output_path: Path, device: str = "cpu", batch_size: int = 32):
    """
    Computes dense semantic embeddings of GO term textual definitions using BERT.
    """
    logger.info(f"Initializing BERT tokenizer and model for {len(go_df):,} GO definitions on device '{device}'...")
    
    target_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(target_device)
    model.eval()
    
    # Fallback to name if definition is empty
    texts = [
        row["definition"] if (isinstance(row["definition"], str) and row["definition"].strip()) else row["name"]
        for _, row in go_df.iterrows()
    ]
    ids = go_df["term"].tolist()
    
    dataset = TextDataset(texts)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    embeddings = {}
    idx = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeRemainingColumn()
    ) as progress:
        task = progress.add_task("Generating BERT Embeddings for GO Terms", total=len(loader))
        with torch.no_grad():
            for batch_texts in loader:
                inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(target_device)
                outputs = model(**inputs).last_hidden_state.mean(dim=1)
                for emb in outputs.cpu():
                    embeddings[ids[idx]] = emb
                    idx += 1
                progress.update(task, advance=1)
                
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)
    
    logger.info(f"Generated and saved {len(embeddings):,} GO term BERT embeddings (dim={list(embeddings.values())[0].shape[-1]}) -> {output_path.name}")

def run_preprocess_go(config: AppConfig, processed_dir: Path, force: bool = False):
    """Executes Step 1 of data pipeline: GO DAG Parsing, Subontology Filtering, and BERT Embedding."""
    log_scientific_stage(
        stage_num="1/5",
        title="Gene Ontology (GO) DAG Parsing & Semantic BERT Embeddings",
        description="Downloads canonical GO definitions, filters to top connected terms, and encodes functional descriptions with BERT.",
        biological_context="The Gene Ontology is a structured, hierarchical vocabulary categorizing gene products across 3 orthogonal aspects: Molecular Function (MF), Biological Process (BP), and Cellular Component (CC). Filtering by topological degree isolates general functional hubs.",
        mathematical_concept="GO terms form a Directed Acyclic Graph (DAG) G=(V, E). Nodes are initialized with mean-pooled dense 768-dim BERT vectors h_v = MeanPool(BERT(Definition_v)), preserving natural language semantic relatedness."
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = processed_dir / "go_nodes.csv"
    edges_path = processed_dir / "go_edges.tsv"
    emb_path = processed_dir / "go_embeddings.pkl"
    
    if not force and nodes_path.exists() and edges_path.exists() and emb_path.exists():
        logger.info(f"[CACHE] GO DAG artifacts already present in {processed_dir}. Skipping computation. (Use --force to overwrite)")
        return

    obo_path = processed_dir / "go-basic.obo"
    download_go_obo(config.data.go_obo_url, obo_path, force=force)
    go_df, _ = parse_go_dag(obo_path, processed_dir, top_n=config.data.top_n_go_terms)
    
    embed_go_terms(go_df, emb_path, device=config.training.device)
    logger.info("✓ Step 1/5 (GO DAG & Embeddings) completed successfully.")
