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
[goa_uniprot_all.gaf.gz] ─► preprocess goa ───────┼──► goa_filtered.tsv
                                           ───────┼──► ia_values.pkl, ia_values.tsv
                                                  │         │
[UniProt REST / FASTA] ──► preprocess proteins ───┼──► protein_data.h5 (ESM-2 embeddings)
                                                  │   protein_index.pkl
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
| `aid2go preprocess goa` | `goa_filtered.tsv`, `ia_values.pkl`, `ia_values.tsv` | Skips GAF streaming, chunk filtering, and DAG $IA$ propagation. |
| `aid2go preprocess proteins` | `protein_data.h5`, `protein_index.pkl` | Skips UniProt REST queries and ESM-2 forward passes. |
| `aid2go preprocess ppi` | `ppi_edge_index.pt` | Skips FAISS index building and nearest-neighbor search. |
| `aid2go preprocess graph` | `hetero_data.pt` | Returns existing compiled PyG `HeteroData` graph. |

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
  - Builds $L_2$-normalized inner-product index `IndexFlatIP` on GPU or multi-threaded CPU for instant $O(N \log k)$ KNN retrieval.
- **PyTorch Geometric `HeteroData` (`hetero_data.pt`)**:
  - Self-contained, serializable multi-relational graph structure containing node attributes, edge indices, and subontology metadata.
