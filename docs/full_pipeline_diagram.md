# Aid2Go pipeline and model inspection map

These diagrams describe the implemented `v2_unique_pairs` pipeline and the saved
`1,000 × 100`, seed 42 comparison. Arrows show data dependencies. The legacy
`v1` artifacts remain separate.

## 1. Sources to the repaired heterogeneous graph

```mermaid
flowchart TD
    subgraph sources[Sources and pretrained encoders]
        OBO["GO basic OBO<br/>term definitions and ontology edges"]
        GAF["GOA UniProt GAF archive<br/>protein to GO evidence rows"]
        UNI["UniProtKB REST<br/>protein FASTA sequences"]
        PDB["Optional local PDB files<br/>C-alpha coordinates"]
        BERT["bert-base-uncased"]
        ESM["ESM-2 650M"]
    end

    subgraph repair[Repaired preprocessing: data/processed/v2_unique_pairs]
        GO["Parse OBO; rank GO terms by ontology degree<br/>keep up to 1,000 terms and selected edges"]
        GOTEXT["Encode GO definitions<br/>768-dimensional BERT vectors"]
        GOA["Validate gzip; filter evidence and selected terms<br/>SQLite unique protein-GO pairs; rank proteins<br/>by distinct term count, then protein ID; keep up to 5,000"]
        SEQ["Fetch and validate sequences<br/>retain sequence-valid proteins and their pairs"]
        PROT["Encode sequences with ESM-2<br/>1,280-dimensional mean-pooled vectors<br/>residue vectors and contact or sequence edges"]
        IA["Propagate final-cohort annotations over GO<br/>calculate information accretion after sequence filtering"]
        KNN["Exact nearest neighbors of ESM vectors<br/>k = 10; Euclidean distance by default"]
        HET["Assemble PyG HeteroData<br/>protein features: ESM vectors<br/>GO features: BERT vectors plus IA<br/>edges: ontology, similarity, annotation and reverse"]
        CACHE["Input fingerprints and dataset manifest<br/>reject stale downstream caches"]
    end

    OBO --> GO --> GOTEXT
    BERT --> GOTEXT
    GAF --> GOA
    GO --> GOA
    GOA --> SEQ
    UNI --> SEQ --> PROT
    ESM --> PROT
    PDB -.-> PROT
    SEQ --> IA
    GO --> IA
    PROT --> KNN
    GOTEXT --> HET
    IA --> HET
    KNN --> HET
    SEQ --> HET
    GO --> HET
    HET --> CACHE

    HET --> GRAPH["hetero_data.pt<br/>4,997 proteins; 1,000 GO terms in saved v2 run"]
    CACHE --> GRAPH
```

The main artifacts are `go_nodes.csv`, `go_edges.tsv`, `go_embeddings.pkl`,
`goa_filtered.tsv`, `protein_sequences.tsv`, `sequence_hashes.tsv`,
`protein_data.h5`, `ia_values.tsv`, `ppi_edge_index.pt`, and `hetero_data.pt`.
The repair audit reports 39,575 unique annotation pairs in the saved v2 graph.
In graph assembly, selected ontology edges are stored under the PyG relation
name `is_a`; the input edge table also records each original relationship.
Optional PDB coordinates are read from local files; this preprocessing path does
not fetch structures.

## 2. Shared sample to training, evaluation, and saved outputs

```mermaid
flowchart TD
    GRAPH["Repaired v2 heterogeneous graph<br/>and sequence hashes"]

    subgraph sample[Shared sample: 1,000 proteins × 100 GO terms, seed 42]
        PS["Select 1,000 distinct sequence hashes<br/>randomly, without annotation labels"]
        SPLIT["Protein holdout<br/>800 train / 100 validation / 100 test"]
        TERMS["Choose 100 GO terms with at least 5<br/>training-protein annotations; sort by<br/>ontology degree, then GO ID"]
        SKNN["Recompute k = 10 similarity neighbors<br/>inside the 1,000-protein sample"]
        SIA["Recompute GO IA from training proteins only"]
        MSG["Training message graph: subset of train<br/>annotation edges and their reverse edges<br/>No held-out annotation edges"]
        LABEL["Training labels: remaining positive train pairs<br/>and equal-count sampled negatives<br/>Validation/test: every held-out protein × all 100 terms"]
        MAN["split_manifest.json and train/val/test_data.pt<br/>selected IDs, counts, sequence separation"]
    end

    subgraph models[Two runs on the same split: 5 epochs, BCE, seed 42, 64 hidden channels, 2 layers]
        BASE["Baseline<br/>relation-wise GraphSAGE + mean aggregation<br/>dot-product protein-GO decoder"]
        ATTN["Degree attention<br/>relation-wise GATv2; deterministic incoming<br/>neighbor caps; normalized attention per receiver<br/>dot-product protein-GO decoder"]
    end

    subgraph assessment[Internal protein-holdout assessment]
        SCORE["Score all 100 test proteins × 100 terms<br/>sigmoid probabilities; use validation<br/>predictions to select explanation threshold"]
        MET["Overall and BP / CC / MF metrics<br/>AUPR, Fmax, IA-weighted Smax<br/>bands by annotation count, incoming<br/>similarity degree, and GO training frequency"]
        EXP["Representative test predictions<br/>protein and GO IDs, scores, neighbor context,<br/>gradient × input, attention if available,<br/>score change after edge removal"]
        EDA["EDA: frequencies, degrees, selection bias,<br/>embedding validity/norms, cosine, PCA,<br/>fixed-seed UMAP, neighbor label agreement"]
    end

    GRAPH --> PS --> SPLIT --> TERMS --> SKNN --> SIA --> MSG --> MAN
    MSG --> LABEL --> MAN
    MAN --> BASE
    MAN --> ATTN
    BASE --> SCORE
    ATTN --> SCORE
    SCORE --> MET
    SCORE --> EXP
    GRAPH --> EDA
    MAN --> EDA

    BASE --> BFILES["models/..._baseline/<br/>best_model.pt, last_model.pt,<br/>training_history.csv, resolved_config.yaml"]
    ATTN --> AFILES["models/..._degree_attention/<br/>best_model.pt, last_model.pt,<br/>training_history.csv, resolved_config.yaml"]
    MET --> REPORTS["reports/experiments/.../<br/>per-model metrics.json, report.md,<br/>heldout_scores.npz, PR and training plots;<br/>comparison.tsv and comparison.json"]
    EXP --> XFILES["per-model explanations/<br/>JSON records and PNG figures"]
    EDA --> EFILES["eda/<br/>PNG figures, TSV diagnostics, summary.json"]
```

The held-out annotation and reverse edges are absent from message passing.
Validation and test message graphs include the sampled similarity edges, so a
held-out protein can receive sequence-neighbor context. Attention weights and
edge-removal score changes are model diagnostics, not biological causal evidence.
Observed annotations are incomplete; an unobserved pair is not a proven
nonfunction. These are **internal protein-holdout results, not a CAFA benchmark**.

### Open the implementation and results

| Inspect | Code or saved output |
|---|---|
| GO, GOA, protein, and graph preparation | [`preprocess_go.py`](../src/aid2go/data/preprocess_go.py), [`preprocess_goa.py`](../src/aid2go/data/preprocess_goa.py), [`preprocess_proteins.py`](../src/aid2go/data/preprocess_proteins.py), [`build_hetero_graph.py`](../src/aid2go/data/build_hetero_graph.py) |
| Sample and split logic | [`sample.py`](../src/aid2go/data/sample.py), [`split_manifest.json`](../data/processed/v2_unique_pairs_sample_1000x100_seed42/split_manifest.json) |
| Model layers and training | [`baseline_model.py`](../src/aid2go/modeling/baseline_model.py), [`degree_attention.py`](../src/aid2go/modeling/degree_attention.py), [`trainer.py`](../src/aid2go/training/trainer.py) |
| Evaluation and explanations | [`protein_holdout.py`](../src/aid2go/eval/protein_holdout.py), [`explain/report.py`](../src/aid2go/explain/report.py) |
| Comparison | [`comparison.tsv`](../reports/experiments/v2_unique_pairs_sample_1000x100_seed42/comparison.tsv), [`comparison.json`](../reports/experiments/v2_unique_pairs_sample_1000x100_seed42/comparison.json) |
| Per-model metrics | [baseline](../reports/experiments/v2_unique_pairs_sample_1000x100_seed42/baseline/metrics.json), [degree attention](../reports/experiments/v2_unique_pairs_sample_1000x100_seed42/degree_attention/metrics.json) |
| EDA index | [`summary.json`](../reports/experiments/v2_unique_pairs_sample_1000x100_seed42/eda/summary.json) |
| Saved configs | [baseline](../models/v2_unique_pairs_sample_1000x100_seed42_baseline/resolved_config.yaml), [degree attention](../models/v2_unique_pairs_sample_1000x100_seed42_degree_attention/resolved_config.yaml) |

For the saved test set, baseline AUPR/Fmax/Smax were
`0.2927 / 0.4022 / 2.9913`; degree attention yielded
`0.2850 / 0.3286 / 3.1266`. The detailed reports contain the aspect-specific
and stratified results. These numbers describe this single fixed-seed run.

The ordinary `aid2go pipeline --config configs/default.yaml` keeps the
5,000-protein/1,000-term limits and the existing **link-holdout** route. It runs
preprocessing, EDA, training, evaluation, and true-path-adjusted prediction
export. The controlled comparison above is produced by
`aid2go experiment sample --baseline-config configs/default.yaml --attention-config configs/degree_attention.yaml --proteins 1000 --go-terms 100 --seed 42`.
