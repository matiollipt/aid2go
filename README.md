# Aid2Go: Protein Function Prediction Framework

Aid2Go is a production-grade framework for protein function prediction using Heterogeneous Graph Neural Networks (GNNs). It predicts Gene Ontology (GO) terms across all three biological subontologies (Biological Process, Molecular Function, and Cellular Component) by learning jointly over:

- **Protein Structural & Sequence Embeddings** (ESM-2 / Contact Graphs)
- **Protein-Protein Interaction & Similarity Networks** (FAISS KNN on ESM-2 embeddings)
- **Gene Ontology DAG Hierarchy** (`is_a` & `part_of` relations)
- **Protein-GO Functional Annotations** (GAF filtered by experimental evidence)
- **Information Accretion ($IA$) node weighting & True-Path rule constraints**

---

## Scientific Background & Computational Approach

### The protein annotation problem

Modern sequencing produces protein sequences far faster than they can be characterized in the laboratory. Public databases hold hundreds of millions of sequences, but only a small fraction have experimentally validated functions. **Automated protein function prediction (AFP)** closes this gap: given a protein, it infers what the protein does, where it acts, and which processes it takes part in. These predictions matter for several areas of life science:

- **Functional genomics**: annotating newly sequenced genomes and metagenomes, and finding the roles of uncharacterized ("dark") proteins.
- **Molecular biology and biomedicine**: prioritizing disease-associated variants and candidate genes, interpreting omics experiments (for example by pathway and term enrichment), and proposing drug targets or repurposing opportunities.
- **Biotechnology**: mining enzymes and pathways for engineering, and guiding which proteins to test experimentally.

The community benchmark for this task is the Critical Assessment of Functional Annotation (CAFA), which Aid2Go's metrics and submission format follow.

### Why the label space makes this hard

Protein functions are described with the **Gene Ontology (GO)**. Predicting GO terms is not an ordinary classification problem, for four reasons:

1. **The labels form a graph, not a flat list.** GO is a **directed acyclic graph (DAG)** of tens of thousands of terms. It has **three roots**, one for each independent sub-ontology: _Molecular Function_ (MF), _Biological Process_ (BP) and _Cellular Component_ (CC). Terms are linked by `is_a` and `part_of` relations (go-basic also has further relation types). A term can have several parents, so the structure is a DAG and not a tree. Relations can also connect terms in different branches.
2. **The True-Path rule couples the labels.** Annotating a protein with a term implies annotation with all of its ancestors up to the root. The labels are therefore strongly dependent, and a valid prediction must obey $P(\text{parent}) \ge P(\text{child})$.
3. **The task is multi-label, imbalanced, and has a long tail.** A protein usually carries tens of terms. General terms such as _cellular process_ are very common, while the specific and most informative terms have few or no examples.
4. **The annotations are incomplete.** A missing annotation is usually "not yet studied", not "absent". Evidence quality also varies, from direct experiments to computational inference.

Treating each GO term as an independent output ignores all of this structure. Such a model cannot share statistical strength between related terms, and it can output scores that contradict the hierarchy.

### The graph approach

Whenever data are made of entities and the relations between them, a **graph** is the natural representation. Here the entities are proteins and GO terms, and the relations are sequence similarity between proteins, the hierarchy between terms, and the "protein has function" links between the two kinds of node. Aid2Go builds one **heterogeneous graph** with these two node types and several edge types. Protein function prediction then becomes **link prediction**: given a protein node and a GO node, estimate whether an edge should connect them.

### Why graph neural networks fit

Graph neural networks (GNNs) are well suited to this setting because the structure of the problem matches how they work:

- **Node features**: every node starts with a vector describing it. Proteins carry ESM-2 sequence embeddings, and GO terms carry BERT embeddings of their textual definitions. The model starts from rich biological and semantic information, not from arbitrary identifiers.
- **Message passing**: in each layer a node collects messages from its neighbours and uses them to update its own representation. Stacking layers lets information travel several hops. A protein learns from similar proteins and from the terms they are annotated with. A GO term learns from its parents, children and annotated proteins. Both node types thus receive information from the neighbours they are connected to.
- **Relation awareness**: heterogeneous models (the HGT encoder and the HeteroConv-based GraphSAGE used here) apply different transformations to each node and edge type, so a similarity link between proteins is not treated like a `part_of` link between terms.
- **Matching inductive bias**: the label hierarchy is part of the input graph, not only a constraint applied afterwards. Related terms get related representations, which helps rare, specific terms borrow evidence from their better-annotated neighbours.
- **A natural fit with the data**: the dataset already consists of connected components and labels, so no artificial reshaping is needed. The annotations are the edges to be predicted, and the surrounding graph gives the context to predict them.

### How the five components fit into the architecture

| #   | Component                                                             | Role in the graph                                                                                                                                                                                                                                                                                                                                                                                                                                          | Where it enters Aid2Go                                                                |
| :-- | :-------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| 1   | **Protein sequence & structural embeddings** (ESM-2 / contact graphs) | **Protein node features.** Mean-pooled 1280-dim ESM-2 vectors encode evolutionary, biochemical and structural signal. With `protein_representation: "graph"`, residue embeddings and a C-alpha contact graph (edges within 8.0 Å) pass through a `ProteinGraphEncoder` (GCN) that pools them into a single protein vector.                                                                                                                                 | `preprocess_proteins.py`, `ProteinGraphEncoder` in `modeling/smart_model.py`          |
| 2   | **Similarity network** (FAISS KNN on ESM-2 embeddings)                | **Protein–protein edges.** Each protein is linked to its nearest neighbours in embedding space, so message passing can transfer annotations between proteins that are likely to be functionally related ("guilt by association"). This is a sequence-similarity network, not an experimental interaction database, and it is species-agnostic.                                                                                                             | `preprocess_ppi.py` → `ppi_edge_index.pt`                                             |
| 3   | **GO DAG hierarchy** (`is_a` & `part_of`)                             | **GO–GO edges.** The ontology gives the label side its structure. Terms exchange information with their parents and children, and BERT definition embeddings serve as their initial features. The three roots (BPO, CCO, MFO) are different regions of the same graph.                                                                                                                                                                                     | `preprocess_go.py` → `go_edges.tsv`, `go_embeddings.pkl`                              |
| 4   | **Protein–GO annotations** (GAF filtered by experimental evidence)    | **Protein–GO edges, which are the supervision.** Only experimentally supported evidence codes are kept (`EXP`, `IDA`, `IPI`, `IMP`, `IGI`, `IEP`, `TAS`, `IC`), so the labels are high-confidence. Known edges provide information to the graph and train the model, and prediction scores the pairs that are not yet linked.                                                                                                                              | `preprocess_goa.py` → `goa_filtered.tsv`, `build_hetero_graph.py`                     |
| 5   | **Information Accretion ($IA$) weighting & True-Path constraints**    | **Node weights and structural constraints.** $IA$ measures how informative a term is, so common root-near terms count little and rare, specific terms count a lot. It is added to GO node features, weights the focal loss (`WeightedFocalLoss`) against the long tail, and defines the IA-weighted $S_{\max}$ metric. At inference, True-Path post-processing propagates scores up the DAG so predictions satisfy $P(\text{parent}) \ge P(\text{child})$. | `ia_values.pkl`, `modeling/losses.py`, `infer/postprocess.py`, `eval/cafa_metrics.py` |

Taken together, items 1 and 3 supply the **node features and structure** (what each protein and term is), items 2 and 4 supply the **edges** along which messages and labels flow, and item 5 makes the **learning and output** respect the biology: it focuses training on informative terms and keeps the final predictions consistent with the ontology.

---

## Features

- **End-to-End Pipeline**: Preprocessing, EDA, GNN training, True-Path post-processing, CAFA-style metrics, and three-column submission generation.
- **Smart Resource & Artifact Caching**: Automatically skips expensive operations (GAF downloads/parsing, ESM-2 transformer forward passes, FAISS index construction) if processed artifacts already exist, with `--force` overrides.
- **Streamlined Rich & Typer CLI**: Rich terminal tables, spinners, progress bars, hardware diagnostic inspector (`aid2go status`), and a one-shot pipeline runner (`aid2go pipeline`).
- **Internal Evaluation**: Keeps the earlier sampled-link check and adds a whole-protein holdout with $F_{\max}$, IA-weighted $S_{\max}$, and AUPR. Neither split is a time-delayed CAFA benchmark.
- **Hierarchical Monotonicity Post-Processing**: Enforces the True-Path rule $P(\text{parent}) \ge P(\text{child})$ via topological score propagation through the GO DAG.

---

## Data Sources & Fetched Files

| Source / Artifact                                               | Download Link & Source                                                                        | Nature of Data                                                                                                     | Purpose in Aid2Go Pipeline                                                                                                                           | Processing Method                                                                                                                                                                                                                                                             |
| :-------------------------------------------------------------- | :-------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Gene Ontology OBO** (`go-basic.obo`)                          | [OBO Foundry GO](https://purl.obolibrary.org/obo/go/go-basic.obo)                             | Structured ontology DAG containing GO terms, definitions, and `is_a`/`part_of` relations across BPO, CCO, and MFO. | Defines the ontology graph structure and text descriptions for GO node features.                                                                     | Parsed via `obonet` into a `networkx.DiGraph`. Top 1000 terms by degree are filtered (`go_nodes.csv`), DAG edges extracted (`go_edges.tsv`), and definitions embedded into 768-dim vectors using BERT (`go_embeddings.pkl`).                                                  |
| **Gene Ontology Annotations (GOA)** (`goa_uniprot_gcrp.gaf.gz`) | [EBI GOA UniProt](https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_gcrp.gaf.gz) | Gene Association File (GAF 2.2) linking UniProt accession IDs to GO terms with experimental evidence codes.        | Provides ground truth bipartite annotation edges (`protein` $\leftrightarrow$ `go`) and empirical term frequencies for Information Accretion ($IA$). | Streamed in chunks; filtered to high-confidence experimental codes (`EXP`, `IDA`, `IPI`, `IMP`, `IGI`, `IEP`, `TAS`, `IC`) and top proteins (`goa_filtered.tsv`). Propagated through the GO DAG to calculate per-term Information Accretion ($IA$) weights (`ia_values.pkl`). |
| **UniProtKB Protein Sequences**                                 | [UniProt REST API](https://rest.uniprot.org/uniprotkb/search)                                 | Amino acid FASTA sequences for annotated UniProtKB entries.                                                        | Source sequences used to compute evolutionary embeddings and contact graphs.                                                                         | Batched REST queries fetch FASTA sequences. Sequences are filtered by length ($\ge 50$ aa) and tokenized for downstream transformer encoders.                                                                                                                                 |
| **ESM-2 Pretrained Transformer**                                | [HuggingFace Hub](https://huggingface.co/facebook/esm2_t33_650M_UR50D)                        | 650M-parameter protein language model (`esm2_t33_650M_UR50D`) trained on UniRef50.                                 | Deep contextualized sequence representations capturing evolutionary, biochemical, and structural features.                                           | Batched forward passes extract residue embeddings. Mean-pooled vectors (1280-dim) and per-residue tensors are saved in a unified HDF5 archive (`protein_data.h5`).                                                                                                            |
| **BERT Definition Encoder**                                     | [HuggingFace Hub](https://huggingface.co/bert-base-uncased)                                   | 110M-parameter English NLP transformer model (`bert-base-uncased`).                                                | Semantic language model encoding textual definitions and descriptions of GO terms into dense vectors.                                                | Tokenizes GO definition strings and mean-pools token hidden states to produce 768-dim GO initial node features.                                                                                                                                                               |
| **AlphaFold DB / PDB Coordinates** _(Optional)_                 | [AlphaFold Database](https://alphafold.ebi.ac.uk/)                                            | Atomic coordinate files (.pdb / .cif) of predicted 3D protein structures.                                          | Spatial structural coordinates for residue contact graph construction.                                                                               | Parsed C-alpha ($C_\alpha$) coordinate vectors; distance matrices ($d \le 8.0$Å) generate sparse contact adjacency maps saved in HDF5 (`contact_edges`). Falls back to sequence adjacency if PDB is unavailable.                                                              |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/maitiollipt/aid2go.git
cd aid2go

# Install dependencies using uv or pip
uv sync
# or
pip install -e .
```

---

## Quickstart

### 1. Check System Status & Cached Artifacts

Inspect your hardware (CUDA GPU, PyTorch version) and existing processed artifacts:

```bash
aid2go status
```

### 2. Repair derived data and compare the two models

The default config keeps the 5,000-protein, 1,000-term limits and points to `v2_unique_pairs`. The old `v1` directory and checkpoints remain available.

```bash
aid2go preprocess repair --config configs/default.yaml --source-version v1
aid2go experiment sample --baseline-config configs/default.yaml --attention-config configs/degree_attention.yaml --proteins 1000 --go-terms 100 --seed 42
```

The sample command saves the split manifest under `data/processed/v2_unique_pairs_sample_1000x100_seed42/`, model checkpoints under `models/`, and EDA, evaluation, comparison, and explanation reports under `reports/experiments/`. It selects GO targets using training proteins only and uses the same split and BCE loss for both models. Existing model results should not be compared with the new protein-holdout results as if they used the same protocol.

See the [full pipeline and model inspection diagrams](docs/full_pipeline_diagram.md) for the data flow, model branches, evaluation, and direct links to the saved outputs.

### 3. Run the End-to-End Pipeline

Run all pipeline steps (Preprocess $\to$ EDA $\to$ Train $\to$ Evaluate $\to$ Export CAFA Submission):

```bash
aid2go pipeline --config configs/default.yaml
```

---

## CLI Reference

### `aid2go status`

Displays hardware diagnostics, CUDA GPU availability, dataset version, and cached file status.

### `aid2go preprocess`

Modular data preparation commands with automatic artifact caching:

- `aid2go preprocess all [--force]`: Run all preprocessing stages sequentially.
- `aid2go preprocess go [--force]`: Download GO OBO, filter top terms, and compute BERT definition embeddings.
- `aid2go preprocess goa [--force]`: Filter GAF annotations by evidence codes and unique protein–GO pairs. For v2, `preprocess proteins` computes Information Accretion ($IA$) after sequence validation.

GOA preprocessing validates the cached gzip archive and uses a disk-backed unique-pair table so repeated evidence rows do not inflate annotation counts or protein ranks. It calculates IA again after sequence-invalid proteins are removed. The repaired version records input fingerprints for GOA, protein embeddings, similarity edges, and graph construction; changed inputs invalidate their downstream cache.

- `aid2go preprocess proteins [--force]`: Retrieve sequences from UniProt and compute ESM-2 embeddings stored in HDF5 (`protein_data.h5`).
- `aid2go preprocess ppi [--force]`: Build sequence-similarity KNN network using FAISS.
- `aid2go preprocess graph [--force]`: Assemble PyG `HeteroData` graph.

### `aid2go eda`

Generate exploratory data analysis summary statistics and distribution plots:

```bash
aid2go eda report --config configs/default.yaml
```

### `aid2go train`

Train baseline, degree-attention, or smart heterogeneous GNN models:

```bash
aid2go train run --config configs/default.yaml
```

### `aid2go evaluate`

Evaluate model checkpoints on held-out data with CAFA-style metrics ($F_{\max}$, $S_{\max}$, AUPR per aspect). The default config still uses sampled-link holdout; resolved sample configs use whole-protein holdout:

```bash
aid2go evaluate --config configs/default.yaml --checkpoint models/aid2go_v2_default/best_model.pt
```

### `aid2go predict`

Run link scoring, apply True-Path rule propagation, and export official 3-column CAFA submission tables (`Protein_ID \t GO_ID \t Score`):

```bash
aid2go predict --config configs/default.yaml --checkpoint models/aid2go_v2_default/best_model.pt --output reports/submissions/submission.tsv
```

### `aid2go explain`

Explain specific protein-GO predictions using GNNExplainer node attribution:

```bash
aid2go explain link --protein-idx 0 --go-idx 5 --config configs/default.yaml
```

---

## Directory Structure

### Project Layout

```text
aid2go/
├── configs/                  # YAML experiment configurations (default, degree_attention, smart)
├── src/aid2go/
│   ├── cli/                  # Typer & Rich CLI subcommands (app, preprocess, train, eval, predict, eda, explain)
│   ├── core/                 # AppConfig schemas, paths, and type definitions
│   ├── data/                 # Data preprocessing pipelines & HeteroData graph builder
│   ├── eval/                 # Internal evaluation and CAFA-style metric sweeps
│   ├── explain/              # GNNExplainer link attribution
│   ├── infer/                # Predictor, True-Path postprocessing, and CAFA submission exporter
│   ├── modeling/             # Smart (HGT), GraphSAGE, and capped-attention models
│   ├── training/             # Trainer, checkpointing, and MLflow trackers
│   └── viz/                  # EDA & visualization helpers
├── tests/                    # Comprehensive pytest test suite
└── reports/                  # Generated plots, metrics, and CAFA submission files
```

### Data & Processed Artifacts Layout

```text
aid2go/
└── data/
    ├── raw/                               # Raw downloads (FASTA, raw OBO, PDB structures)
    │   └── pdb/                           # Optional local AlphaFold / PDB coordinate files ({uniprot_id}.pdb)
    │
    └── processed/
        ├── v1/                            # Preserved previous derived data
        ├── v2_unique_pairs/               # Repaired derived data with unique GOA links
        └── v2_unique_pairs_sample_1000x100_seed42/ # Shared sample and protein split
```

The repaired dataset contains the same kinds of files as `v1`, plus `protein_sequences.tsv`, `sequence_hashes.tsv`, and preprocessing input manifests. Because the old HDF5 file did not store sequence hashes, the default repair regenerates all ESM embeddings from newly fetched sequences. An explicit `data.reuse_protein_data_path` can reuse legacy vectors after accession, dimension, and residue-length checks, but that weaker validation should be reported with any results.

The preserved `v1` layout is:

```text
data/processed/v1/
            ├── go-basic.obo               # Gene Ontology OBO file
            ├── go_nodes.csv               # Top GO terms with degree metadata
            ├── go_edges.tsv               # GO DAG relationship edges
            ├── go_embeddings.pkl          # 768-dim BERT definition embeddings
            ├── goa_uniprot_gcrp.gaf.gz    # Downloaded GAF annotation file
            ├── goa_filtered.tsv           # Filtered annotations by evidence code (EXP, IDA, etc.)
            ├── ia_values.pkl              # Information Accretion (IA) dictionary
            ├── ia_values.tsv              # Information Accretion TSV table
            ├── protein_data.h5            # HDF5 archive (ESM-2 embeddings & contact graphs)
            ├── protein_index.pkl          # Protein ID index list
            ├── ppi_edge_index.pt          # FAISS KNN sequence-similarity graph
            └── hetero_data.pt             # Compiled PyG HeteroData multi-relational graph
```

---

## Testing

Run the full pytest suite:

```bash
uv run pytest
```
