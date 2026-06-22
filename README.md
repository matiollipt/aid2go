# Aid2Go

Aid2Go is a Python package for protein function prediction using Heterogeneous Graph Neural Networks.
It predicts Gene Ontology (GO) terms for proteins by leveraging a graph of:
- Protein-Protein Interactions (PPI)
- GO Hierarchy (is_a relationships)
- Protein-GO Annotations (experimental evidence)

## Installation

```bash
pip install -e .
```

## Quickstart

1. **Configuration**: Edit `configs/default.yaml` to set your paths and parameters.
2. **Preprocessing**:
   ```bash
   # Download and process GO Ontology
   aid2go preprocess go
   
   # Download and filter GO Annotations (GOA)
   aid2go preprocess goa
   
   # Fetch and embed Protein Sequences (ProtBERT)
   aid2go preprocess proteins
   
   # Process PPI network
   aid2go preprocess ppi
   
   # Build the Heterogeneous Graph
   aid2go preprocess graph
   ```
3. **Training**:
   ```bash
   aid2go train run --config configs/default.yaml
   ```
4. **Evaluation**:
   ```bash
   aid2go evaluate --checkpoint models/aid2go_default/best_model.pt
   ```

## CLI Commands

The CLI is the main entrypoint:
- `aid2go preprocess [go|goa|proteins|ppi|graph]`: Data preparation steps.
- `aid2go train run`: Train a model.
- `aid2go predict`: Run inference.
- `aid2go eda report`: Generate data statistics.

## Data Structure

Data is organized in `data/`:
- `data/raw`: Raw downloads.
- `data/processed/<version>`: Processed artifacts (embeddings, graph objects).

## Models

- **Baseline**: GraphSAGE-based HeteroGNN.
- **Smart**: GAT-based HeteroGNN with focal loss.

Select the model in `configs/default.yaml`:
```yaml
model:
  name: "smart"
```
