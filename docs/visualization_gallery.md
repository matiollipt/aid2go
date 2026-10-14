# Offline visualization gallery

Run `aid2go eda gallery --help` for the configuration, output directory, seed,
and projection-cache options. With compatible processed and trained artifacts,
run `aid2go eda gallery` to write `reports/visualization-gallery/index.html`.
Open that file in a browser; its scripts, styles, data, and network renderer
are embedded, so viewing does not require a server or internet connection.

The builder requires the processed heterogeneous graph, GO nodes and ontology,
IA values, the configured best checkpoint and training history, and the saved
1000-term/100-protein seed-42 sample experiment reports for the baseline and
degree-attention comparisons. It reads existing artifacts; changing dataset
limits in a configuration does not rebuild those artifacts. Graphviz `dot`
must be available when generating the gallery's ontology examples.

## Association and neighborhood exploration

- The ground-truth and prediction panels each have a searchable protein picker
  and a bounded, scrollable list of GO terms. Prediction scores have threshold
  and display-limit controls. Protein selections stay synchronized.
- The ESM similarity networks start at the selected protein's direct neighbors.
  Select one to three hops, a node budget, and a minimum similarity. Edge and
  node caps are reported so a capped view is not mistaken for the complete
  neighborhood. These edges represent embedding similarity.
- Search the analyzed GO vocabulary by identifier or name to redraw its GO
  neighborhood. Parent and child hop controls include ontology context beyond
  the analyzed vocabulary. Parents appear above children; arrows follow the
  ontology's child-to-parent direction. Context terms are visually distinguished.
- Click graph nodes to inspect or reroot; use zoom and fit to explore the view.

The gallery also includes degree and IA distributions, t-SNE and UMAP panels,
training curves, ROC/precision-recall diagnostics, semantic metric sweeps, and
saved prediction explanations. Full-graph exploratory predictions use the
annotation graph and must not be interpreted as held-out performance. The
sample experiment panels identify their evaluation scope; the CAFA-style
internal metrics are not official competition certification.

## Reproducibility and packaging

Alongside the HTML, the builder saves `data.json`, a source manifest, projection
caches, and prediction score caches with input fingerprints. Generated reports
and model/data artifacts remain outside version control.

The network renderer is Cytoscape.js 3.34.3, bundled under its MIT license.
`src/aid2go/viz/vendor/cytoscape-version.json` records the upstream package and
integrity metadata. Templates, the renderer, metadata, and license are included
in the Python package. No CDN script is required by the generated page.
