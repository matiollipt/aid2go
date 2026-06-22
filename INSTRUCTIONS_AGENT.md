# AID2GO PROJECT

## 1) Current module inventory (what exists, and what it’s for)

### A) End-to-end scripts / entrypoints

* **`experiments.py`** — *Notebook-style* end-to-end pipeline: loads GO embeddings + GO edges + IA, loads per-protein ProtBERT tensors, builds a `HeteroData` graph (`protein` ↔ `go`), runs `RandomLinkSplit`, trains `Aid2GO_SmartModel` with `FocalLoss` via `Trainer`. It currently **does not add PPI edges** (comment says “missing PPI edges”). 
* **`main.py`** — Typer CLI with `train` and `evaluate`. The `train` command builds a **baseline** hetero link model (`HeteroLinkModel`), uses `Trainer`, and supports **tracking** via `MLflowTracker` / Tensorboard / NoOp. The `evaluate` command uses `CafaEvaluator`. 

### B) Core utilities (config, paths, types, reproducibility)

* **`config.py`** — Pydantic configs (`TrainingConfig`, `ModelConfig`, `Config.from_yaml`). 
* **`paths.py`** — Centralized data directories (`Paths`), env override via `CAFA6_DATA_ROOT`, plus `mkdirs()`. 
* **`seed.py`** — `seed_everything()` reproducibility helper. 
* **`types.py`** — type aliases (`Batch = HeteroData`, etc.). 

### C) Data acquisition + preprocessing

**GO / GOA**

* **`goa.py`** — Filters GOA GAF by evidence codes (chunked), optionally removes obsolete terms (downloads GO OBO), and can emit “CAFA-like” annotation TSV. 
* **`go.py`** — GO DAG download + dataframe export, edge extraction, text embedding with BERT, GO graph building (hetero/multi-edge or homo with edge types), plus clustering/plots and subgraph generation utilities. 
* **`ia.py`** — Computes Information Accretion (IA), includes ontology cleaning, aspect sub-ontology, and term propagation logic. 

**Proteins / sequences / embeddings / PPI**

* **`utils.py`** — FTP download helper, FASTA loader, misc utilities (layout calc, UniProt mapping download). 
* **`proteins.py`** — ProtBERT embedding pipeline (`embed_sequences_from_dict`), datasets for embedding tensors, PPI dataset builder with degree distribution plot, LCC extraction, and UniProt REST FASTA fetch helpers. 
* **`ppi.py`** — Minimal helper to convert a PPI edge dataframe into an edge index tensor. 

### D) Graph splitting / loaders

* **`splits.py`** — `random_link_split()` wrapper around `torch_geometric.transforms.RandomLinkSplit` for the hetero edge type (`protein`, `function`, `go`). 

### E) Modeling (baseline + “smart” model)

* **`hetero_gnn.py`** — Baseline `HeteroLinkModel`: linear projection + node embeddings + GraphSAGE (via `to_hetero`) + dot-product link classifier. 
* **`smart_model.py`** — `Aid2GO_SmartModel`: two-stream GAT (protein stream expects PPI edges if present; GO stream uses IA as importance); uses edge attributes in GATConv (`edge_dim=1`) and an MLP head on concatenated protein+GO features for link logits. 
* **`encoders.py`** — Simple MLP encoder block. 

### F) Losses

* **`losses.py`** — default `BCEWithLogitsLoss`. 
* **`smart_losses.py`** — `FocalLoss` for imbalance. 

### G) Training + tracking

* **`base.py`** — `Tracker` protocol + `NoOpTracker`. 
* **`mlflow_tracker.py`** — local MLflow tracker saving runs to disk, logs params/metrics/artifacts. 
* (`Trainer` is referenced by both `experiments.py` and `main.py` but its file isn’t in the uploaded set; it’s the right place to centralize metric logging + checkpointing.)

### H) Evaluation + inference + reporting + viz

* **`cafa_metrics.py`** — micro-AUPR + Fmax sweep utilities. 
* **`cafa_evaluator.py`** — `CafaEvaluator` applies (optional) true-path propagation using GO graph and IA utilities, outputs `CafaReport`. 
* **`scorer.py`** — batched inference: scores protein×GO candidates by building a temporary heterograph and running neighbor-sampled loaders. 
* **`summaries.py`** — prediction summarization + top-k per protein saving. 
* **`protein_go.py`** — interactive Plotly sunburst for a protein’s top predicted GO terms (currently doesn’t reconstruct the real GO parent-child hierarchy). 

---

## 2) What’s “scattered” today (the main friction points)

1. **Two parallel pipelines**:

* `experiments.py` (Smart model) hardcodes paths/config and does not use tracking. 
* `main.py train` (Baseline model) already has config + MLflow tracking + checkpoint save, but duplicates data loading/graph building logic. 

2. **Data artifacts are produced in multiple places/formats**:

* GO embeddings are pickles with naming that varies by column, GO edges are TSVs, IA is a text/TSV, protein embeddings are per-protein `.pt` files, etc. (`go.py`, `ia.py`, `proteins.py`, `experiments.py` all touch these).

3. **EDA exists, but not as a coherent “report pipeline”**:

* Degree plots are produced inside `PPIDataset`, clustering plots in `go.py`, and protein GO viz is elsewhere.

4. **Explainability hooks are not formalized**:

* `Aid2GO_SmartModel` is attention-based and IA-weighted, which is great for interpretation, but nothing standard exports attention/attribution per predicted link. 

---

## 3) Refactor plan: streamline into 4 “pipelines” + one shared graph builder

### Target capability checklist (what the refactor should enable)

* **Preprocessing**: deterministic generation of (a) GO graph + embeddings + IA, (b) protein embeddings, (c) PPI edges, (d) annotation edges.
* **EDA**: one command generates dataset stats + plots + HTML reports and logs them to MLflow.
* **Training**: one training loop for *both* baseline and smart models, with consistent metrics, MLflow params/metrics, and model checkpoints logged as artifacts.
* **Explainability**: given `(protein_id, go_id)` (or top-k per protein), generate an explanation artifact (subgraph + ranked neighbors/edges + attention/attribution), saved + logged.

### Proposed folder structure (minimal changes, maximum reuse)

Keep the current `cafa6/` idea, but make responsibilities explicit:

**`cafa6/core/`**

* `config.py`, `paths.py`, `seed.py`, logging utilities. (Already present.)

**`cafa6/data/`**

* `preprocess_go.py` (wrap what’s in `go.py` + IA computation)
* `preprocess_proteins.py` (wrap what’s in `proteins.py`) 
* `preprocess_ppi.py` (wrap `ppi.py` + `get_lcc` / filters)
* `build_hetero_graph.py` (**new single source of truth**) that:

  * loads artifacts (GO embeddings, GO edges, IA, protein embeddings, PPI edges, annotation edges),
  * builds `HeteroData` with *consistent node ordering and id↔index maps*,
  * attaches optional node/edge attributes (e.g., `go.ia`).

**`cafa6/viz/`**

* `eda.py` — consolidated “report builders” (degree distributions, GO frequency/IA histograms, embedding projections, split diagnostics). Reuse plotting already inside `proteins.py`/`go.py`, but **don’t plot as a side-effect of dataset building**.
* `protein_go.py` — upgrade to build a true GO hierarchy view (use GO DAG parents) instead of parent="".

**`cafa6/modeling/`**

* keep `hetero_gnn.py`, `smart_model.py`, `losses.py`, `smart_losses.py`, `encoders.py`.
* add a thin `registry.py` (string → model class) so CLI can choose `baseline|smart`.

**`cafa6/training/`**

* `trainer.py` becomes the *only* place that:

  * computes train/val metrics,
  * logs to `Tracker`,
  * checkpoints best model,
  * optionally logs confusion/PR curves as artifacts.
    (`main.py` already passes a tracker; `experiments.py` should too.)

**`cafa6/eval/`**

* keep `cafa_metrics.py`, `cafa_evaluator.py`.

**`cafa6/infer/`**

* move `scorer.py` and `summaries.py` here; add `postprocess.py` (true-path expansion + thresholding per aspect).

**`cafa6/explain/`** *(new)*

* `explain_links.py`:

  * “attention export” path for the Smart GAT (modify `SmartFocusedGAT` to optionally return attention weights),
  * “model-agnostic” path using `torch_geometric.explain.Explainer` for link explanations (subgraph + edge mask),
  * emit HTML/JSON artifacts per protein or per link.

### CLI plan (single coherent UX)

Extend Typer CLI (`main.py`) into subcommands:

* `cafa6 preprocess go` → produces `go_edges.tsv`, `go_embeddings.pkl`, `ia.tsv`
* `cafa6 preprocess proteins` → produces per-protein `.pt` and a manifest of failed/too-long sequences
* `cafa6 preprocess ppi` → produces `ppi_edges.pt` (+ optional filtered LCC)
* `cafa6 eda` → generates `reports/` (png + html) + logs artifacts to MLflow
* `cafa6 train --model smart|baseline` → unified training, MLflow params/metrics, saves best checkpoint
* `cafa6 predict` → uses `scorer.py` to score, then `summaries.py` to top-k and optional CAFA formatting
* `cafa6 explain --uniprot P12345 --topk 30` → explain + viz outputs

---

## 4) Concrete “streamlining” steps (ordered to minimize disruption)

### Step 1 — Deduplicate graph construction

* Extract the shared parts of `experiments.py` graph building into `cafa6/data/build_hetero_graph.py`, then make:

  * `experiments.py` become a **thin** “run smart experiment” wrapper calling library code. 
  * `main.py train` call the same builder instead of re-implementing GO/protein loading. 

**Outcome:** one canonical place to define node ordering, mappings, and which edge types exist.

### Step 2 — Unify config + paths

* Replace the inline `Config` class in `experiments.py` with the Pydantic config + `Paths` (already exists).
* Add explicit “artifact paths” in config (where GO embeddings live, which text field was embedded, which ProtLM was used, etc.).

**Outcome:** reproducible runs; no more “data vs data2” surprises.

### Step 3 — Make EDA a first-class pipeline (and MLflow artifact source)

* Move plotting side-effects out of dataset constructors (e.g., the degree histogram inside `PPIDataset`) into `viz/eda.py`. 
* `eda` command writes all plots to `reports/` and logs them via `MLflowTracker.log_artifact()`.

**Outcome:** EDA is consistent, repeatable, and tracked per dataset version.

### Step 4 — Training: unify metrics + checkpointing + tracking

* Ensure `Trainer` logs:

  * train/val loss
  * **AUPR + Fmax** (you already have CAFA metrics utilities) instead of accuracy-only (accuracy is especially misleading with negative sampling).
* Log **model weights** as MLflow artifacts:

  * best checkpoint (`best_model.pt`)
  * optionally last checkpoint, optimizer state
* Have both baseline and smart models train via the same `Trainer` interface.

**Outcome:** consistent training dashboards and comparable experiments across model variants. (And `experiments.py` finally tracks runs like the CLI already does.)

### Step 5 — Explainability (pragmatic first version)

Implement two explain modes:

1. **Smart-model attention report (fastest win)**
   Modify `SmartFocusedGAT` / `Aid2GO_SmartModel` to optionally return:

   * top contributing GO neighbors (ontology edges) by attention,
   * top contributing protein neighbors (PPI) by attention, when PPI edges are present.

2. **Model-agnostic subgraph explanation (more general)**
   Use PyG explainer to return an edge mask over the sampled computation graph for a specific `(protein, go)` link; export:

   * a small subgraph (nodes/edges)
   * ranked edges
   * an HTML network view
   * attach GO term names/namespace using GO DAG utilities.

**Outcome:** every prediction can ship with “why” artifacts, logged alongside the run.

---

If you want, I can turn this plan into a **proposed new package tree + a mapping table (“move X → Y”)** so refactoring becomes mostly mechanical (and you can keep both `baseline` and `smart` runnable at every commit).

---

## 1) Automatic folder + file creation (dirs + touch)

### Option A — one bash script (recommended)

```bash
cat > create_aid2go_skeleton.sh << 'BASH'
#!/usr/bin/env bash
set -euo pipefail

ROOT="aid2go"
SRC="$ROOT/src/aid2go"
TESTS="$ROOT/tests"
SCRIPTS="$ROOT/scripts"
CONFIGS="$ROOT/configs"
NOTEBOOKS="$ROOT/notebooks"
REPORTS="$ROOT/reports"
DATA="$ROOT/data"
DOCS="$ROOT/docs"

mkdir -p \
  "$SRC/core" \
  "$SRC/data" \
  "$SRC/viz" \
  "$SRC/modeling" \
  "$SRC/training" \
  "$SRC/eval" \
  "$SRC/infer" \
  "$SRC/explain" \
  "$SRC/cli" \
  "$TESTS" \
  "$SCRIPTS" \
  "$CONFIGS" \
  "$NOTEBOOKS" \
  "$REPORTS" \
  "$DATA/raw" \
  "$DATA/interim" \
  "$DATA/processed" \
  "$DOCS"

# Root-level project files
touch \
  "$ROOT/README.md" \
  "$ROOT/pyproject.toml" \
  "$ROOT/.gitignore" \
  "$ROOT/Makefile" \
  "$ROOT/.env.example" \
  "$ROOT/LICENSE" \
  "$ROOT/CHANGELOG.md"

# Package init files
touch \
  "$SRC/__init__.py" \
  "$SRC/py.typed" \
  "$SRC/core/__init__.py" \
  "$SRC/data/__init__.py" \
  "$SRC/viz/__init__.py" \
  "$SRC/modeling/__init__.py" \
  "$SRC/training/__init__.py" \
  "$SRC/eval/__init__.py" \
  "$SRC/infer/__init__.py" \
  "$SRC/explain/__init__.py" \
  "$SRC/cli/__init__.py"

# Core
touch \
  "$SRC/core/config.py" \
  "$SRC/core/paths.py" \
  "$SRC/core/seed.py" \
  "$SRC/core/types.py" \
  "$SRC/core/logging.py" \
  "$SRC/core/errors.py"

# Data pipeline
touch \
  "$SRC/data/artifacts.py" \
  "$SRC/data/preprocess_go.py" \
  "$SRC/data/preprocess_goa.py" \
  "$SRC/data/preprocess_proteins.py" \
  "$SRC/data/preprocess_ppi.py" \
  "$SRC/data/build_hetero_graph.py" \
  "$SRC/data/splits.py" \
  "$SRC/data/datasets.py" \
  "$SRC/data/parsers.py"

# Viz / EDA
touch \
  "$SRC/viz/eda.py" \
  "$SRC/viz/go_viz.py" \
  "$SRC/viz/protein_go.py"

# Modeling
touch \
  "$SRC/modeling/encoders.py" \
  "$SRC/modeling/baseline_model.py" \
  "$SRC/modeling/smart_model.py" \
  "$SRC/modeling/losses.py" \
  "$SRC/modeling/registry.py"

# Training + tracking
touch \
  "$SRC/training/trainer.py" \
  "$SRC/training/metrics.py" \
  "$SRC/training/trackers.py" \
  "$SRC/training/mlflow_tracker.py" \
  "$SRC/training/checkpoints.py"

# Eval + CAFA
touch \
  "$SRC/eval/cafa_metrics.py" \
  "$SRC/eval/cafa_evaluator.py" \
  "$SRC/eval/scorer.py" \
  "$SRC/eval/reporting.py"

# Inference / postprocess
touch \
  "$SRC/infer/predict.py" \
  "$SRC/infer/summaries.py" \
  "$SRC/infer/postprocess.py"

# Explainability
touch \
  "$SRC/explain/explain_links.py" \
  "$SRC/explain/attention_export.py" \
  "$SRC/explain/subgraph_export.py"

# CLI
touch \
  "$SRC/cli/app.py" \
  "$SRC/cli/preprocess_cmd.py" \
  "$SRC/cli/eda_cmd.py" \
  "$SRC/cli/train_cmd.py" \
  "$SRC/cli/predict_cmd.py" \
  "$SRC/cli/evaluate_cmd.py" \
  "$SRC/cli/explain_cmd.py"

# Scripts (shell helpers)
touch \
  "$SCRIPTS/run_preprocess_all.sh" \
  "$SCRIPTS/run_eda.sh" \
  "$SCRIPTS/run_train_baseline.sh" \
  "$SCRIPTS/run_train_smart.sh" \
  "$SCRIPTS/run_predict.sh" \
  "$SCRIPTS/run_explain.sh"

# Tests
touch \
  "$TESTS/test_parsers.py" \
  "$TESTS/test_graph_builder.py" \
  "$TESTS/test_splits.py" \
  "$TESTS/test_training_smoke.py" \
  "$TESTS/test_inference_smoke.py"

# Minimal config examples
touch \
  "$CONFIGS/default.yaml" \
  "$CONFIGS/baseline.yaml" \
  "$CONFIGS/smart.yaml"

echo "✅ Created skeleton at: $ROOT/"
BASH

chmod +x create_aid2go_skeleton.sh
./create_aid2go_skeleton.sh
```


