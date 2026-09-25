# Data Pipeline & Caching Architecture

Aid2Go is designed to process multi-gigabyte genomic datasets efficiently while preventing unnecessary re-computation during iterative model exploration.

---

## 1. Pipeline Flowchart & Artifact Lineage

The Aid2Go data pipeline operates as a directed dependency graph:

```
[go-basic.obo] ───► preprocess go ──────► go_nodes.csv
                                   ──────► go_edges.tsv
                                   ──────► go_embeddings.pkl
                                                  │
[goa_uniprot_gcrp.gaf.gz] ─► preprocess goa ──────┼──► unique-pair goa_filtered.tsv
                                                  │         │
[UniProt REST / FASTA] ──► preprocess proteins ───┼──► protein_data.h5 (ESM-2 embeddings)
                                                  │   protein_index.pkl, sequence_hashes.tsv
                                                  └──► final-cohort ia_values.pkl, ia_values.tsv
                                                  │         │
                                   preprocess ppi ◄─────────┘
                                         │
                                         ▼
                                   ppi_edge_index.pt
                                         │
                                         ▼
                                build_hetero_graph
                                         │
                                         ▼
                                   hetero_data.pt (PyG HeteroData)
```

---

## 2. Resource Management & Caching Strategy

Operations such as parsing multi-gigabyte GAF annotation files or performing forward passes through large ESM-2 transformer models (`esm2_t33_650M_UR50D`) require significant compute time and memory.

### Caching Rules & Fast Resume Matrix

| Pipeline Command | Target Artifacts Checked | Action when Present (`force=False`) |
| :--- | :--- | :--- |
| `aid2go preprocess go` | `go_nodes.csv`, `go_edges.tsv`, `go_embeddings.pkl` | Skips OBO download, DAG filtering, and BERT tokenization. |
| `aid2go preprocess goa` | `goa_filtered.tsv` for v2 | Skips GAF streaming and unique-pair filtering. |
| `aid2go preprocess proteins` | `protein_data.h5`, `protein_index.pkl`, `sequence_hashes.tsv`, `ia_values.pkl`, `ia_values.tsv` | Skips UniProt REST queries, ESM-2 forward passes, and final-cohort IA when input fingerprints match. |
| `aid2go preprocess ppi` | `ppi_edge_index.pt` | Skips FAISS index building and nearest-neighbor search. |
| `aid2go preprocess graph` | `hetero_data.pt` | Returns existing compiled PyG `HeteroData` graph. |

### GPU execution

`training.device: auto` selects CUDA when PyTorch can access a GPU and otherwise
uses CPU. An explicit `cpu` setting remains available for debugging. Check the
configured and effective devices with `aid2go status`; a CUDA-enabled PyTorch
build alone does not establish that a working driver and GPU are available.

In `v2_unique_pairs`, GOA selection counts distinct protein-GO pairs with a
temporary SQLite table. Repeated evidence records do not inflate rankings.
GOA, protein, similarity, and graph stages save input fingerprints; modifying
an upstream artifact or relevant setting causes the downstream stage to run
again. IA is refreshed after the final sequence-valid protein cohort is known.
The repaired data is written to a new version, leaving `v1` untouched.

ESM-2 embedding batches are grouped by sequence length. `data.embedding_batch_size`
(default 32) and `data.embedding_token_budget` (default 8192 padded tokens)
bound each batch. CUDA inference uses mixed precision by default and halves the
batch after a CUDA out-of-memory error. A single sequence that cannot fit still
fails clearly. The HDF5 file is replaced only after the embedding pass succeeds.
The model's maximum sequence length is enforced before creating residue graphs.

Training uses CUDA mixed precision when `training.mixed_precision` is true,
gradient accumulation, pinned host memory for sampled batches, and a resident
GPU graph for full-graph training. Exact KNN search runs in CUDA tensor blocks;
FAISS remains the CPU fallback. Parsing, tokenization, HDF5 writes, graph
assembly, and scikit-learn evaluation metrics remain CPU tasks. GPU memory use
varies with sequence length and the free memory available; filling every byte
would cause out-of-memory failures and is not a useful optimization target.

### Baseline training checks

The default baseline trains a two-layer heterogeneous GraphSAGE encoder and
dot-product link decoder on precomputed ESM-2 protein features and BERT-plus-IA
GO features. The split removes duplicate protein-GO pairs, uses disjoint
training supervision, and supplies equal numbers of sampled positive and
negative links. When PyG sampling extensions are absent, each epoch has one
full-graph optimizer update, so the default five epochs can finish in seconds.
The trainer prints the link counts, per-epoch loss and validation scores, and
optimizer-update counts. It also writes `training_history.csv` beside the
model checkpoints. Previously saved checkpoints and evaluation scores from the
duplicate, positive-only split require retraining and reevaluation.

### Progress and checkpoint loading

The pipeline shows one persistent row for each stage. Completed cached stages
are marked `cached`, completed work is marked `done`, and work counters disappear
after their stage finishes. A missing output artifact marks its stage as failed.

PyTorch loads model checkpoints with `weights_only=True` and checks that they
contain tensor state dictionaries. The PyG graph artifact also loads with
`weights_only=True`, using a scoped allowlist for its graph container classes.
Hugging Face may report unused pretraining-head weights when `BertModel` or
`EsmModel` loads a checkpoint trained with extra heads. For the local BERT and
ESM checkpoints, the encoder weights load without missing parameters; ESM's
unused pooler is disabled. Missing encoder weights would require investigation.

### Forcing Re-computation
To bypass the cache and recompute any step from scratch, pass the `--force` (`-f`) flag:
```bash
# Force recompute GOA and IA values only
aid2go preprocess goa --force

# Force recompute the entire pipeline
aid2go preprocess all --force
```

---

## 3. Storage Formats & High-Throughput I/O

- **HDF5 (`protein_data.h5`)**:
  - `protein_embeddings/`: 1280-dimensional mean-pooled vectors per protein ID.
  - `residue_embeddings/`: $[L \times 1280]$ per-residue tensors.
  - `contact_edges/`: $[2 \times E]$ structural contact or sequence proximity edges.
  - *Advantage*: Eliminates inode exhaustion, enables memory-mapped chunked slicing, and avoids storing hundreds of thousands of individual `.pt` files.
- **FAISS Indexing**:
  - Uses blocked exact CUDA tensor search when available, or a CPU FAISS flat index for Euclidean or cosine KNN retrieval.
- **PyTorch Geometric `HeteroData` (`hetero_data.pt`)**:
  - Self-contained, serializable multi-relational graph structure containing node attributes, edge indices, and subontology metadata.
