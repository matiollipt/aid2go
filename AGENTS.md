# AID2GO Project & Agent Guide

This document provides a comprehensive overview of the refactored `aid2go` package, outlines the scientific and technical domain nuances of protein function prediction, and defines future engineering directions.

---

## 1. Scientific Domain Nuances: Protein Function & Gene Ontology (GO)

To work effectively on this codebase, agents must understand the biological context of the prediction task:

### The Gene Ontology (GO)
The Gene Ontology is a structured, controlled vocabulary representing gene product properties across three independent categories (aspects):
1. **Molecular Function (MF)**: Elemental activities performed by a gene product (e.g., `catalytic activity`, `protein binding`).
2. **Biological Process (BP)**: Broad molecular pathways or programs (e.g., `cell division`, `mitochondrion organization`).
3. **Cellular Component (CC)**: The subcellular localization where a protein operates (e.g., `nucleus`, `cytoplasm`).

### The Hierarchical DAG Structure
GO terms are organized as a Directed Acyclic Graph (DAG) where nodes represent terms, and directed edges represent relations (e.g., `is_a`, `part_of`).
* **The True-Path Rule**: A fundamental rule in GO annotations. If a protein is annotated with a GO term, it is implicitly annotated with all its ancestral terms up to the root. Consequently, child terms represent more specific functions, while parent terms represent general functions.

### Information Accretion (IA)
Because parent terms are extremely common, predicting them is easy but carries little information. The specificity of a term is measured by its **Information Content (IC)** or **Information Accretion (IA)**.
* Mathematically, IA computes the surprise of observing a term given that all its parents are observed:
  $$\text{IA}(t) = -\log_2 \left( \frac{P(t \text{ is annotated})}{P(\text{all parents of } t \text{ are annotated})} \right)$$
* In `aid2go`, we calculate IA dynamically across the dataset using propagation through the subontology DAG. Terms with high IA values represent rare, highly specific functions. We use these IA values to weight the focal loss during GNN training, ensuring the model focuses on learning specific annotations.

### Dataset Scaling for Prototyping
For rapid local development, the dataset is subset specifically to maintain density:
* **GO Terms (Top 1000 by degree)**: We filter the GO DAG to the 1000 terms with the highest connections (in-degree + out-degree) to capture general semantic anchors.
* **Proteins (Top 5000 by annotation density)**: We filter annotations to the top 5000 proteins that contain the most annotations within the selected GO terms. This preserves a dense, connected bipartite graph for prototyping.

---

## 2. Refactored Architecture Overview

The codebase is structured under the `src/aid2go/` directory to adhere to modular software engineering standards:

```text
src/aid2go/
├─ core/                  # Configurations, paths, and reproducibility seeds
│  ├─ config.py           # Pydantic configuration schemas (AppConfig, DataConfig, etc.)
│  ├─ paths.py            # Local workspace path helpers
│  └─ types.py            # Enum type declarations (ModelName, DatasetSplit)
├─ data/                  # Preprocessing pipelines and graph construction
│  ├─ preprocess_go.py    # Downloads OBO, filters top 1000 GO terms, generates BERT embeddings
│  ├─ preprocess_goa.py   # Filters GAF annotations, subsets to 5000 proteins, computes IA values
│  ├─ preprocess_proteins.py # Fetches sequences, generates ESM-2 embeddings, builds contact graphs (HDF5)
│  ├─ preprocess_ppi.py   # Computes FAISS-based KNN graph over protein embeddings
│  └─ build_hetero_graph.py # Compiles PyG HeteroData (concatenating IA features & reverse edges)
├─ modeling/              # GNN models and losses
│  ├─ SmartModel.py       # Shared HGT encoder, ProteinGraphEncoder, and 3 category-specific decoders
│  ├─ baseline_model.py   # Simple HeteroConv-based GraphSAGE model
│  ├─ losses.py           # WeightedFocalLoss scaled by GO node IA weights
│  └─ registry.py         # Model registration lookup
├─ training/              # Training loops and trackers
│  └─ trainer.py          # Unified trainer with automatic full-graph training fallback
├─ explain/               # Explainability pipelines
│  └─ explain_links.py    # GNNExplainer module using node feature importance
├─ cli/                   # Typer CLI subcommands
│  ├─ app.py              # CLI entrypoint
│  ├─ preprocess_cmd.py   # Preprocessing CLI actions
│  ├─ train_cmd.py        # Model training CLI action
│  └─ explain_cmd.py      # Explainer CLI action
```

---

## 3. Key Technical Details

### Protein Embeddings & Graphs
* **ESM-2 Embeddings**: Protein sequences are encoded using the `facebook/esm2_t33_650M_UR50D` transformer model.
* **HDF5 Storage**: All protein embeddings (mean-pooled vectors and per-residue tensors) are saved in a unified HDF5 file `protein_data.h5` to prevent disk thrashing from thousands of small `.pt` files.
* **Hierarchical Encoding**: When `protein_representation` is set to `"graph"`, we load residue-level embeddings along with a structure-based contact graph (where edges connect C-alpha atoms within 8.0Å). A `ProteinGraphEncoder` (using GCNs) pools the residue graphs dynamically during model evaluation.

### FAISS KNN Graph
Instead of hardcoding external Protein-Protein Interaction (PPI) databases which are species-specific, `aid2go` builds a sequence-similarity network by computing a K-Nearest Neighbors (KNN) graph on the ESM-2 embeddings using FAISS. The `strict_cosine_similarity` option normalizes embeddings before applying `IndexFlatIP`.

### Explainer
Explainability is powered by PyTorch Geometric's `Explainer` and `GNNExplainer`. To prevent crashes inside HGT layers during backward passes:
* Empty edge index tensors are padded with a dummy self-loop.
* The explainer uses `edge_mask_type=None` and `node_mask_type='attributes'`, focusing on node-level importance to discover which neighboring nodes drive predictions.

---

## 4. Future Directions & Technical Roadmap

Agents continuing work on `aid2go` should focus on the following enhancements:

### 1) Structured Post-Processing for Hierarchical Constraints
Currently, GNN decoders predict edge logits independently. Future work should implement an explicit **True-Path Rule enforcement** layer or post-processing scorer:
* Ensure that the predicted probability of a child GO term cannot exceed the predicted probability of its parent GO terms:
  $$P(\text{child}) \le P(\text{parent})$$
* Implement Bayesian network propagation or isotonic regression to enforce hierarchical monotonicity.

### 2) Advanced Spatial Protein GNNs
The current `ProteinGraphEncoder` utilizes a basic 2-layer GCN on C-alpha contact edges.
* Upgrade this to a spatial GNN (such as EGNN, SchNet, or GVP) that can consume coordinate distance matrices directly as edge features, retaining geometric invariance.

### 3) Large-Scale Distributed Data Loader
For scaling the package beyond prototype limits (e.g. training on the full CAFA6 dataset with millions of proteins and 45,000 GO terms):
* Replace the fallback trainer list loaders with a disk-backed heterogeneous neighbor loader (e.g., using `PyG`'s `GraphSAINT` or partition-based loaders).
* Integrate chunked HDF5 reading during mini-batch sampling to ensure GPU memory does not overflow.

### 4) Multimodal Embedding Ensemble
* Integrate additional dimensions into node initialization:
  * For GO nodes, combine BERT definition embeddings with Graph Neural Network structural embeddings of the GO DAG itself.
  * For protein nodes, combine ESM-2 embeddings with text representations of scientific publications extracted from PubMed (e.g., via PubMedBERT).
