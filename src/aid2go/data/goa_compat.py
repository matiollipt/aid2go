"""Legacy propagated-table helpers retained for existing callers.

The repaired v2 pipeline uses the disk-backed unique-pair stream and computes
IA only after the sequence-valid cohort is fixed. These helpers remain useful
for small standalone tables and for the earlier API.
"""

from __future__ import annotations

import os

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed


def get_default_num_workers(requested: int | None = None) -> int:
    if isinstance(requested, int) and requested > 0:
        return requested
    return max(1, min(os.cpu_count() or 4, 32))


def _propagate_batch(records: list[dict], ancestor_lookup: dict) -> tuple[list, list, list]:
    entries, terms, aspects = [], [], []
    for row in records:
        present = set(row["term"])
        lookup = ancestor_lookup.get(row["aspect"], {})
        for term in row["term"]:
            present.update(lookup.get(term, ()))
        for term in present:
            entries.append(row["EntryID"])
            terms.append(term)
            aspects.append(row["aspect"])
    return entries, terms, aspects


def propagate_terms(terms_df: pd.DataFrame, subontologies: dict,
                    num_workers: int | None = None) -> pd.DataFrame:
    ancestor_lookup = {}
    for aspect, ontology in subontologies.items():
        observed = set(terms_df.loc[terms_df.aspect == aspect, "term"])
        ancestor_lookup[aspect] = {term: set(nx.descendants(ontology, term))
                                   for term in observed if term in ontology}
    grouped = terms_df.groupby(["EntryID", "aspect"])["term"].apply(set).reset_index()
    records = grouped.to_dict("records")
    if not records:
        return pd.DataFrame(columns=["EntryID", "term", "aspect"])
    workers = min(get_default_num_workers(num_workers), len(records))
    chunks = np.array_split(records, min(len(records), workers * 4))
    results = Parallel(n_jobs=workers)(
        delayed(_propagate_batch)(chunk.tolist(), ancestor_lookup) for chunk in chunks)
    entries, terms, aspects = [], [], []
    for batch_entries, batch_terms, batch_aspects in results:
        entries.extend(batch_entries)
        terms.extend(batch_terms)
        aspects.extend(batch_aspects)
    return pd.DataFrame({"EntryID": entries, "term": terms, "aspect": aspects})


def term_counts(terms_df: pd.DataFrame, term_indices: dict) -> sp.csr_matrix:
    valid = terms_df.loc[terms_df.term.isin(term_indices), ["EntryID", "term"]].drop_duplicates()
    if valid.empty:
        return sp.csr_matrix(np.ones((1, len(term_indices)), dtype=np.int32))
    proteins = {protein: index for index, protein in enumerate(valid.EntryID.unique())}
    row = valid.EntryID.map(proteins).to_numpy()
    column = valid.term.map(term_indices).to_numpy()
    matrix = sp.coo_matrix((np.ones(len(valid), dtype=np.int32), (row, column)),
                           shape=(len(proteins), len(term_indices))).tocsr()
    dummy = sp.csr_matrix(np.ones((1, len(term_indices)), dtype=np.int32))
    return sp.vstack((matrix, dummy), format="csr")


def calc_ia(term: str, count_matrix: sp.spmatrix, ontology, terms_index: dict) -> float:
    if term not in terms_index:
        return 0.0
    parents = list(ontology.successors(term))
    if not parents or any(parent not in terms_index for parent in parents):
        return 0.0
    child_count = float(count_matrix[:, terms_index[term]].sum())
    parent_count = float((count_matrix[:, [terms_index[parent] for parent in parents]]
                          .sum(axis=1) == len(parents)).sum())
    if parent_count <= 0 or child_count >= parent_count:
        return 0.0
    return -float(np.log2(child_count / parent_count))
