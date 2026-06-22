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
