import os
import sys
import argparse
import obonet
import numpy as np
import pandas as pd
import networkx as nx
from goatools.obo_parser import GODag
from collections import Counter
from scipy.sparse import dok_matrix
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/ia_processing.log"), logging.StreamHandler()],
)


def clean_ontology_edges(ontology):
    """
    Remove all ontology edges except types "is_a" and "part_of" and ensure there are no inter-ontology edges.
    :param ontology: Ontology structure (networkx DiGraph or MultiDiGraph).
    :return: Cleaned ontology graph.
    """
    logging.info("Cleaning ontology edges...")

    # Copy ontology to avoid modifying the original object
    ontology_clean = ontology.copy()

    # Keep only "is_a" and "part_of" edges
    remove_edges = [
        (i, j, k) for i, j, k in ontology.edges if not (k == "is_a" or k == "part_of")
    ]
    ontology_clean.remove_edges_from(remove_edges)

    # Verify and remove cross-ontology edges
    crossont_edges = [
        (i, j, k)
        for i, j, k in ontology.edges
        if ontology_clean.nodes[i]["namespace"] != ontology_clean.nodes[j]["namespace"]
    ]
    if crossont_edges:
        logging.warning(f"Removing {len(crossont_edges)} cross-ontology edges.")
        ontology_clean.remove_edges_from(crossont_edges)

    logging.info("Ontology edges cleaned.")
    return ontology_clean


def fetch_aspect(ontology, root):
    """
    Return a subgraph of an ontology starting at a specified root node.

    :param ontology: Ontology structure (networkx DiGraph or MultiDiGraph).
    :param root: Root node (GO term) to start the subgraph.
    :return: Subgraph of the ontology corresponding to the specified root.
    """
    logging.info(f"Fetching aspect subgraph for root: {root}")
    namespace = ontology.nodes[root]["namespace"]
    aspect_nodes = [
        n for n, v in ontology.nodes(data=True) if v["namespace"] == namespace
    ]
    subont_ = ontology.subgraph(aspect_nodes)
    logging.info(f"Subgraph for {root} fetched with {len(subont_)} nodes.")
    return subont_


def propagate_terms(terms_df, subontologies):
    """
    Propagate terms in a DataFrame according to the structure in subontologies.

    :param terms_df: DataFrame of annotated terms (columns: 'EntryID', 'term', 'aspect').
    :param subontologies: Dictionary of ontology aspects (networkx DiGraphs or MultiDiGraphs).
    :return: DataFrame with propagated terms.
    """
    logging.info("Propagating terms based on subontology structure...")

    # Precompute ancestors for efficiency
    subont_terms = {
        aspect: set(terms_df[terms_df.aspect == aspect].term.values)
        for aspect in subontologies.keys()
    }
    ancestor_lookup = {
        aspect: {
            t: nx.descendants(subont, t) for t in subont_terms[aspect] if t in subont
        }
        for aspect, subont in subontologies.items()
    }

    propagated_terms = []
    for (protein, aspect), entry_df in terms_df.groupby(["EntryID", "aspect"]):
        protein_terms = set().union(
            *[list(ancestor_lookup[aspect][t]) + [t] for t in set(entry_df.term.values)]
        )
        propagated_terms += [
            {"EntryID": protein, "term": t, "aspect": aspect} for t in protein_terms
        ]

    logging.info("Terms propagation completed.")
    return pd.DataFrame(propagated_terms)


def term_counts(terms_df, term_indices):
    """
    Count the number of instances of each term.

    :param terms_df: DataFrame of propagated terms (columns: 'EntryID', 'term', 'aspect').
    :param term_indices: Dictionary mapping terms to indices.
    :return: Sparse matrix of term counts.
    """
    logging.info("Counting term instances...")
    num_proteins = len(terms_df.groupby("EntryID"))
    S = dok_matrix((num_proteins + 1, len(term_indices)), dtype=np.int32)
    S[-1, :] = 1  # Add a dummy protein

    for i, (protein, protdf) in enumerate(terms_df.groupby("EntryID")):
        row_count = {term_indices[t]: c for t, c in Counter(protdf["term"]).items()}
        for col, count in row_count.items():
            S[i, col] = count

    logging.info("Term counting completed.")
    return S


def parse_inputs(argv):
    """
    Parse command-line arguments.

    :param argv: List of command-line arguments.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute Information Accretion of GO annotations. Note: If annotations in input file have been propagated to ontology roots, the input ontology graph should be the same as the one used to propagate terms."
    )

    parser.add_argument("--annot", "-a", required=True, help="Path to annotation file.")
    parser.add_argument(
        "--graph",
        "-g",
        default=None,
        help="Path to OBO ontology graph file if local. If empty, the current OBO structure will be downloaded.",
    )
    parser.add_argument(
        "--outfile",
        "-o",
        default="IA.txt",
        help="Path to save computed IA for each term in the GO. Defaults to ./IA.txt",
    )
    parser.add_argument(
        "--prop",
        "-p",
        action="store_true",
        help="Flag to propagate terms in the annotation file according to the ontology graph.",
    )

    return parser.parse_args(argv)


def calc_ia(term, count_matrix, ontology, terms_index):
    """
    Compute Information Accretion (IA) for a given term.

    :param term: GO term to compute IA for.
    :param count_matrix: Sparse matrix of term counts.
    :param ontology: Ontology graph.
    :param terms_index: Dictionary mapping terms to column indices in the count matrix.
    :return: Information Accretion value for the term.
    """
    parents = nx.descendants_at_distance(ontology, term, 1)

    # Count proteins with the term
    prots_with_term = count_matrix[:, terms_index[term]].sum()

    # Count proteins with all parents
    num_parents = len(parents)
    prots_with_parents = (
        count_matrix[:, [terms_index[p] for p in parents]].sum(1) == num_parents
    ).sum()

    if prots_with_term == prots_with_parents:
        return 0

    return -np.log2(prots_with_term / prots_with_parents)


# if __name__ == "__main__":
#     args = parse_inputs(sys.argv[1:])

#     # Load ontology graph and annotated terms
#     if args.graph is None:
#         logging.info("Downloading GO OBO file...")
#         ontology_graph = download_file(
#             "http://purl.obolibrary.org/obo/go/go-basic.obo",
#             os.path.join(data_location, "go-basic.obo"),
#         )
#     else:
#         ontology_graph = clean_ontology_edges(obonet.read_obo(args.graph))

#     annotation_df = pd.read_csv(args.annot, sep="\t")

#     # Get the three subontologies
#     roots = {"BPO": "GO:0008150", "CCO": "GO:0005575", "MFO": "GO:0003674"}
#     subontologies = {
#         aspect: fetch_aspect(ontology_graph, roots[aspect]) for aspect in roots
#     }

#     if args.prop:
#         logging.info("Propagating terms...")
#         annotation_df = propagate_terms(annotation_df, subontologies)

#     # Count term instances
#     logging.info("Counting terms...")
#     aspect_counts = {}
#     aspect_terms = {}
#     term_idx = {}
#     for aspect, subont in subontologies.items():
#         aspect_terms[aspect] = sorted(subont.nodes)  # Ensure consistent order
#         term_idx[aspect] = {t: i for i, t in enumerate(aspect_terms[aspect])}
#         aspect_counts[aspect] = term_counts(
#             annotation_df[annotation_df.aspect == aspect], term_idx[aspect]
#         )

#         assert aspect_counts[aspect].sum() == len(
#             annotation_df[annotation_df.aspect == aspect]
#         ) + len(aspect_terms[aspect])

#     sp_matrix = {aspect: dok.tocsc() for aspect, dok in aspect_counts.items()}

#     # Compute IA
#     logging.info("Computing Information Accretion...")
#     aspect_ia = {
#         aspect: {t: 0 for t in aspect_terms[aspect]} for aspect in aspect_terms.keys()
#     }
#     for aspect, subontology in subontologies.items():
#         for term in aspect_ia[aspect].keys():
#             aspect_ia[aspect][term] = calc_ia(
#                 term, sp_matrix[aspect], subontology, term_idx[aspect]
#             )

#     ia_df = pd.concat(
#         [
#             pd.DataFrame.from_dict(
#                 {
#                     "term": aspect_ia[aspect].keys(),
#                     "ia": aspect_ia[aspect].values(),
#                     "aspect": aspect,
#                 }
#             )
#             for aspect in subontologies.keys()
#         ]
#     )

#     # Validate and save results
#     assert ia_df["ia"].min() >= 0

#     logging.info(f"Saving IA results to {args.outfile}...")
#     ia_df[["term", "ia"]].to_csv(args.outfile, header=None, sep="\t", index=False)
#     logging.info("IA computation completed successfully.")
