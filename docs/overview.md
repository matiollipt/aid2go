# Aid2Go Documentation

Welcome to the official documentation for **Aid2Go**, a high-performance, modular Python framework for automated protein function prediction using Heterogeneous Graph Neural Networks (GNNs).

---

## Documentation Navigation

- **[Scientific Foundations & CAFA Benchmark](cafa_and_science.md)**: Deep dive into the molecular genetics domain, Gene Ontology (GO), the True-Path Rule, ESM-2 protein language modeling, FAISS sequence-similarity networks, and the CAFA competition structure.
- **[Evaluation Metrics & Statistical Foundations](evaluation_and_statistics.md)**: Mathematical formulations and scientific rationales for certified CAFA metrics ($F_{\max}$, Information Accretion weighted $S_{\max}$, Remaining Uncertainty $RU$, Misinformation $MI$, and AUPR).
- **[API Reference](api_reference.md)**: Comprehensive technical documentation of all Python modules, classes, functions, and CLI subcommands in `aid2go`.
- **[Data Pipeline & Caching Architecture](pipeline_and_caching.md)**: Details on the deterministic ETL pipeline, HDF5 storage, FAISS KNN graph construction, PyG HeteroData compilation, and smart artifact caching.

---

## High-Level Architecture

Aid2Go frames protein function prediction as a multi-relational link prediction task on a heterogeneous graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:

```
                    ┌──────────────────────────────────────────────┐
                    │               Gene Ontology DAG              │
                    │               Nodes: 45,000+ GO              │
                    │           Features: BERT Definition          │
                    │           Edges: is_a / part_of              │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           │ (has_function / function_of)
                                           │ Annotation Edges (GAF)
                                           ▼
┌──────────────────────────────────────────┴───────────────────────┐
│                          Protein Network                         │
│                    Nodes: UniProt / CAFA Sequences               │
│                  Features: ESM-2 650M Mean Embeddings            │
│                  Structure: C-alpha Contact Residue Graphs       │
│                  Edges: FAISS KNN Sequence-Similarity            │
└──────────────────────────────────────────────────────────────────┘
```

The system trains a two-stream Heterogeneous Graph Transformer (HGT) / Attention network to project proteins and GO concepts into a shared embedding space, predicting functional associations while strictly respecting hierarchical ontology constraints.
