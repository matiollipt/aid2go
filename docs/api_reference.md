# API Reference

This document provides technical documentation for the core classes, functions, and modules in the `aid2go` package.

---

## 1. Core (`aid2go.core`)

### `AppConfig` (`aid2go.core.config.AppConfig`)
Pydantic configuration schema encapsulating all hyper-parameters and file paths.
- `paths`: `PathsConfig` defining workspace directories.
- `data`: `DataConfig` defining dataset version, top-N GO terms, ESM-2 model name, FAISS parameters, and sequence thresholds.
- `model`: `ModelConfig` defining model architecture (`smart` or `baseline`), hidden dimensions, layer counts, and decoders.
- `training`: `TrainingConfig` defining epochs, learning rates, weight decay, device (`cuda` or `cpu`), and batch sizes.
- `tracking`: `TrackingConfig` defining MLflow tracking URI, run names, and logging flags.

### `ProjectPaths` (`aid2go.core.paths.ProjectPaths`)
Helper class providing deterministic directory resolution:
- `get_raw_dir() -> Path`: Resolves and creates `data/raw/`.
- `get_processed_dir(version: str) -> Path`: Resolves and creates `data/processed/<version>/`.
- `get_report_dir(report_name: str) -> Path`: Resolves and creates `reports/<report_name>/`.

---

## 2. Data Pipelines (`aid2go.data`)

### `run_preprocess_go(config: AppConfig, processed_dir: Path, force: bool = False)`
Parses the Gene Ontology OBO file, computes node degrees, filters top $N$ terms, extracts ontology hierarchy edges, and generates 768-dimensional BERT textual definition embeddings stored in `go_embeddings.pkl`. Automatically skips processing if target artifacts exist unless `force=True`.

### `run_preprocess_goa(config: AppConfig, processed_dir: Path, force: bool = False)`
Parses Gene Ontology Annotation (GAF) files by evidence code into disk-backed unique protein–GO pairs, ranks proteins by distinct annotations, and saves `goa_filtered.tsv`. For v2, IA is deferred until sequence validation.

### `run_preprocess_proteins(config: AppConfig, processed_dir: Path, force: bool = False)`
Fetches protein amino acid sequences from UniProtKB via REST API, tokenizes sequences, and computes 1280-dimensional ESM-2 transformer embeddings. Constructs C-alpha spatial contact graphs (or sequence chain fallbacks) and saves tensors to `protein_data.h5`. The v2 path also saves sequence hashes and computes IA from the final sequence-valid cohort.

### `run_preprocess_ppi(config: AppConfig, processed_dir: Path, force: bool = False)`
Loads protein embeddings from `protein_data.h5`, builds exact nearest neighbors with L2 distance or normalized cosine similarity according to the config, and exports `ppi_edge_index.pt`.

### `build_hetero_graph(config: AppConfig, processed_dir: Path, force: bool = False) -> HeteroData`
Assembles all processed components into a unified PyTorch Geometric `HeteroData` graph:
- Node Types: `protein` (ESM-2 features), `go` (BERT embeddings + concatenated $IA$ weight).
- Edge Types:
  - `("go", "is_a", "go")`: Ontology DAG hierarchy.
  - `("protein", "similarity", "protein")`: FAISS KNN interaction network.
  - `("protein", "has_function", "go")`: Experimental functional annotations.
  - `("go", "function_of", "protein")`: Reverse functional annotations.
- Exports `hetero_data.pt`.

---

## 3. Modeling & Losses (`aid2go.modeling`)

### `SmartModel` (`aid2go.modeling.smart_model.SmartModel`)
High-capacity GNN architecture combining:
- **Shared Heterogeneous Graph Transformer (HGT)** encoder over multi-relational graphs.
- **`ProteinGraphEncoder`**: 2-layer GCN pooling structural residue contact graphs.
- **Category-Specific Decoders**: 3 independent MLP heads for **Biological Process**, **Cellular Component**, and **Molecular Function** subontologies.

### `WeightedFocalLoss` (`aid2go.modeling.losses.WeightedFocalLoss`)
Focal loss scaled dynamically by Gene Ontology node Information Accretion ($IA$) weights:
$$\mathcal{L} = -\alpha_t (1 - p_t)^\gamma \cdot w_{IA} \cdot \log(p_t)$$

---

## 4. Evaluation Suite (`aid2go.eval`)

### `CAFAEvaluator` (`aid2go.eval.cafa_evaluator.CAFAEvaluator`)
Certified evaluation suite for CAFA competition benchmarks:
- `evaluate(y_true, y_score, go_terms=None) -> Dict[str, float]`: Computes global and aspect-partitioned (BPO, CCO, MFO) $F_{\max}$, $S_{\max}$, and AUPR.
- `generate_report(metrics, output_dir) -> Path`: Exports Markdown (`cafa_evaluation_report.md`) and JSON summary tables.

### Metric Functions (`aid2go.eval.cafa_metrics`)
- `calculate_fmax(y_true, y_score, thresholds=None) -> float`: Evaluates multi-threshold protein-centric $F_{\max}$.
- `calculate_smax(y_true, y_score, ia_weights, thresholds=None) -> float`: Evaluates $IA$-weighted semantic distance $S_{\max}$.
- `calculate_aupr(y_true, y_score) -> float`: Computes area under the precision-recall curve.

---

## 5. Inference & Post-Processing (`aid2go.infer`)

### `Predictor` (`aid2go.infer.predict.Predictor`)
- `predict_scores(protein_indices=None, apply_true_path=True, obo_path=None) -> Dict[str, Dict[str, float]]`: Generates continuous link probabilities across all protein-GO pairs, optionally applying True-Path monotonicity propagation.

### `enforce_true_path_rule` (`aid2go.infer.postprocess.enforce_true_path_rule`)
Propagates prediction probabilities upward through the GO DAG ensuring $P(\text{parent}) \ge \max_{c \in \text{children}} P(c)$.

### `export_cafa_submission` (`aid2go.infer.summaries.export_cafa_submission`)
Formats predictions into official 3-column tab-separated CAFA submission files:
```text
Protein_ID \t GO_ID \t Score (3 decimal places)
```

---

## 6. Explainability (`aid2go.explain`)

### `explain_protein_go_link(model, data, protein_idx, go_idx, epochs=50)`
Employs PyTorch Geometric's `Explainer` and `GNNExplainer` to identify influential node features and neighboring proteins/terms driving specific functional predictions.
