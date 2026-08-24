# AID2GO Project & Agent Guide

This document provides a comprehensive overview of the `aid2go` package, outlines the scientific and technical domain nuances of protein function prediction, and defines future engineering directions.

---

## 1. Scientific Domain Nuances: Protein Function & Gene Ontology (GO)

To work effectively on this codebase, agents must understand the biological context of the prediction task:

### The Gene Ontology (GO)
The Gene Ontology is a structured, controlled vocabulary representing gene product properties across three independent categories (aspects):
1. **Molecular Function (MF)**: Elemental activities performed by a gene product (e.g., `catalytic activity`, `protein binding`).
2. **Biological Process (BP)**: Broad molecular pathways or programs (e.g., `cell division`, `mitochondrion organization`).
3. **Cellular Component (CC)**: The subcellular localization where a protein operates (e.g., `nucleus`, `cytoplasm`).

### The Hierarchical DAG Structure & True-Path Rule
GO terms are organized as a Directed Acyclic Graph (DAG) where nodes represent terms, and directed edges represent relations (e.g., `is_a`, `part_of`).
* **The True-Path Rule**: If a protein is annotated with a GO term, it is implicitly annotated with all its ancestral terms up to the root. Consequently, child terms represent more specific functions, while parent terms represent general functions.
* **Score Monotonicity**: In inference, predicted probabilities must satisfy $P(\text{parent}) \ge P(\text{child})$. In `aid2go`, this is enforced through topological propagation in `aid2go.infer.postprocess.enforce_true_path_rule`.

### Information Accretion (IA) & Certified CAFA Evaluation
Because parent terms are common, predicting them carries little information. Specificity is quantified via **Information Accretion (IA)**:
$$\text{IA}(t) = -\log_2 \left( \frac{P(t \text{ is annotated})}{P(\text{all parents of } t \text{ are annotated})} \right)$$
* **Loss Weighting**: In training, IA values dynamically weight the focal loss in `WeightedFocalLoss`, ensuring the model focuses on rare, highly specific annotations.
* **$S_{\max}$ Metric**: In evaluation, IA weights are used to compute Remaining Uncertainty ($RU$) and Misinformation ($MI$) to determine the semantic distance $S_{\max} = \min_{\tau} \sqrt{RU(\tau)^2 + MI(\tau)^2}$ alongside $F_{\max}$ (*Radivojac et al., 2013*; *Zhou et al., 2019*).

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
│  ├─ preprocess_go.py    # Downloads OBO, filters top 1000 GO terms, generates BERT embeddings (with caching)
│  ├─ preprocess_goa.py   # Filters GAF annotations, subsets to 5000 proteins, computes IA values (with caching)
│  ├─ preprocess_proteins.py # Fetches sequences, generates ESM-2 embeddings, builds contact graphs (HDF5)
│  ├─ preprocess_ppi.py   # Computes FAISS-based KNN graph over protein embeddings
│  └─ build_hetero_graph.py # Compiles PyG HeteroData (concatenating IA features & reverse edges)
├─ modeling/              # GNN models and losses
│  ├─ smart_model.py      # Shared HGT encoder, ProteinGraphEncoder, and 3 category-specific decoders
│  ├─ baseline_model.py   # Simple HeteroConv-based GraphSAGE model
│  ├─ losses.py           # WeightedFocalLoss scaled by GO node IA weights
│  └─ registry.py         # Model registration lookup
├─ training/              # Training loops and trackers
│  └─ trainer.py          # Unified trainer with automatic full-graph training fallback & evaluation
├─ eval/                  # Certified CAFA evaluation suite
│  ├─ cafa_metrics.py     # Fmax, Smax (IA weighted), and AUPR calculation routines
│  └─ cafa_evaluator.py   # Subontology (BP, CC, MF) breakdown and report generation
├─ infer/                 # Inference, post-processing & competition export
│  ├─ predict.py          # Pairwise link scoring engine
│  ├─ postprocess.py      # True-Path Rule propagation & threshold filtering
│  └─ summaries.py        # Official 3-column CAFA submission table exporter
├─ explain/               # Explainability pipelines
│  └─ explain_links.py    # GNNExplainer module using node feature importance
├─ viz/                   # Exploratory Data Analysis & visualizations
│  └─ eda.py              # Degree distributions, GO frequencies, summary charts
└─ cli/                   # Typer & Rich CLI subcommands
   ├─ app.py              # CLI entrypoint with `status` and `pipeline` runners
   ├─ preprocess_cmd.py   # Preprocessing CLI actions with `--force` caching control
   ├─ train_cmd.py        # Model training CLI action
   ├─ evaluate_cmd.py     # Certified evaluation CLI action with Rich table display
   ├─ predict_cmd.py      # Prediction & CAFA submission exporter CLI action
   ├─ eda_cmd.py          # Dataset EDA CLI action
   └─ explain_cmd.py      # GNNExplainer CLI action
```

---

## 3. Key Technical Details

### Resource & File Caching Layer
All preprocessing pipelines (`preprocess_go`, `preprocess_goa`, `preprocess_proteins`, `preprocess_ppi`, `build_hetero_graph`) verify the presence of existing target artifacts. Expensive operations (such as multi-gigabyte GAF parsing or ESM-2 forward passes) are skipped unless `--force` (`-f`) is passed to the CLI or `force=True` is provided programmatically.

### Protein Embeddings & Graphs
* **ESM-2 Embeddings**: Protein sequences are encoded using the `facebook/esm2_t33_650M_UR50D` transformer model.
* **HDF5 Storage**: All protein embeddings (mean-pooled vectors and per-residue tensors) are saved in a unified HDF5 file `protein_data.h5` to prevent disk thrashing.
* **Hierarchical Encoding**: When `protein_representation` is set to `"graph"`, we load residue-level embeddings along with a structure-based contact graph (edges connecting C-alpha atoms within 8.0Å). A `ProteinGraphEncoder` (using GCNs) pools the residue graphs dynamically during model evaluation.

### FAISS KNN Graph
Instead of hardcoding external Protein-Protein Interaction (PPI) databases which are species-specific, `aid2go` builds a sequence-similarity network by computing a K-Nearest Neighbors (KNN) graph on the ESM-2 embeddings using FAISS (`IndexFlatIP` with normalized vectors or `IndexFlatL2`).

### Certified CAFA Evaluation & Submission Exporter
* **Evaluation**: `CAFAEvaluator` computes $F_{\max}$, $S_{\max}$ (IA weighted), and Micro/Macro AUPR globally and partitioned across subontologies (BPO, CCO, MFO).
* **Submission**: `export_cafa_submission` generates official CAFA 3-column TSV files (`Protein_ID \t GO_ID \t Score` formatted to 3 decimal places).

---

## 4. Future Directions & Technical Roadmap

Agents continuing work on `aid2go` should focus on the following enhancements:

### 1) Advanced Spatial Protein GNNs
The current `ProteinGraphEncoder` utilizes a basic 2-layer GCN on C-alpha contact edges.
* Upgrade this to a spatial GNN (such as EGNN, SchNet, or GVP) that can consume coordinate distance matrices directly as edge features, retaining geometric invariance.

### 2) Large-Scale Distributed Mini-Batch Sampling
For scaling beyond prototype limits to full-genome datasets (millions of proteins, 45,000 GO terms):
* Implement partition-based heterogeneous loaders (`GraphSAINT`, `ClusterLoader`).
* Integrate chunked HDF5 reads during batch generation to prevent memory pressure.

### 3) Multimodal Embedding Ensemble
* Integrate additional dimensions into node initialization:
  * For GO nodes, combine BERT definition embeddings with Graph Neural Network structural embeddings of the GO DAG itself.
  * For protein nodes, combine ESM-2 embeddings with text representations of scientific publications extracted from PubMed (e.g., via PubMedBERT).
