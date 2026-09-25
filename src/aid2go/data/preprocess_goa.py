"""Stream GOA annotations and calculate information accretion."""

from __future__ import annotations

import gzip
import logging
import math
import os
import pickle
import shutil
import sqlite3
import subprocess
import tempfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import obonet
import pandas as pd
import requests

from aid2go.cli.progress import make_progress
from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths
from aid2go.data.goa_compat import (get_default_num_workers, propagate_terms,
                                    term_counts, calc_ia)

logger = logging.getLogger(__name__)
SCAN_LINES = 100_000


def download_file(url: str, output_path: Path):
    resume_at = 0
    if output_path.exists():
        try:
            verify_gzip(output_path)
            logger.info("Using verified GOA download: %s", output_path)
            return
        except (EOFError, gzip.BadGzipFile, OSError, zlib.error) as error:
            logger.warning("Cached GOA is incomplete or corrupt: %s (%s). Downloading replacement.", output_path, error)
            if isinstance(error, EOFError):
                resume_at = output_path.stat().st_size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    os.close(fd)
    try:
        try:
            headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
            with requests.get(url, headers=headers, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {resume_at}-"):
                        raise RuntimeError(f"Unexpected GOA Content-Range: {content_range}")
                    with open(output_path, "rb") as source, open(temporary, "wb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                    initial = resume_at
                else:
                    initial = 0
                remaining = int(response.headers.get("Content-Length", 0)) or None
                total = initial + remaining if remaining is not None else None
                with make_progress() as progress:
                    task = progress.add_task("Download GOA", total=total, completed=initial,
                                             detail=f"{initial:,} bytes")
                    with open(temporary, "ab" if initial else "wb") as destination:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                destination.write(chunk)
                                progress.update(task, advance=len(chunk), detail=f"{destination.tell():,} bytes")
            verify_gzip(Path(temporary))
        except (requests.RequestException, EOFError, gzip.BadGzipFile, OSError, zlib.error, RuntimeError) as error:
            reason = (
                f"Cannot reach {url}: {type(error).__name__}"
                if isinstance(error, requests.RequestException)
                else f"Archive is incomplete or corrupt: {error}"
            )
            raise RuntimeError(
                f"GOA download could not be validated. {reason}. "
                f"The existing file at {output_path} was preserved. "
                "Check network access and rerun 'aid2go preprocess goa'."
            ) from error
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify_gzip(path: Path):
    """Read through the trailer, checking CRC and end marker before trusting a cache file."""
    with make_progress() as progress:
        task = progress.add_task("Verify GOA archive", total=path.stat().st_size, detail="0 bytes")
        try:
            with open(path, "rb") as raw:
                with gzip.GzipFile(fileobj=raw) as archive:
                    while archive.read(1024 * 1024):
                        progress.update(task, completed=raw.tell(),
                                        detail=f"{raw.tell():,} compressed bytes")
        except (EOFError, gzip.BadGzipFile, OSError, zlib.error):
            progress.update(task, description="[red]Verify GOA archive FAILED[/red]",
                            detail="incomplete or corrupt")
            raise
        progress.update(task, completed=path.stat().st_size,
                        detail=f"{path.stat().st_size:,} compressed bytes")


def _scan_gaf(path, codes, selected_terms, progress, task, num_workers=None):
    chunks = rows = matches = 0
    pigz = shutil.which("pigz") if path.suffix == ".gz" else None
    worker_count = get_default_num_workers(num_workers)
    with open(path, "rb") as raw:
        process = (subprocess.Popen([pigz, "-dc", "-p", str(worker_count), str(path)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                   if pigz else None)
        stream = process.stdout if process else (
            gzip.GzipFile(fileobj=raw) if path.suffix == ".gz" else raw)
        try:
            for line in stream:
                if line.startswith(b"!"):
                    continue
                rows += 1
                fields = line.rstrip(b"\r\n").split(b"\t")
                if len(fields) >= 7:
                    evidence = fields[6].decode("utf-8")
                    term = fields[4].decode("utf-8")
                    if evidence in codes and (selected_terms is None or term in selected_terms):
                        matches += 1
                        yield fields[1].decode("utf-8"), term
                if rows % SCAN_LINES == 0:
                    chunks += 1
                    progress.update(task, completed=raw.tell() if not process else None,
                                    detail=f"{chunks:,} chunks · {rows:,} rows · {matches:,} matches")
        finally:
            if stream is not raw:
                stream.close()
            if process:
                code = process.wait()
                error = process.stderr.read().decode("utf-8", errors="replace")
                process.stderr.close()
                if code:
                    raise RuntimeError(f"pigz failed while reading {path}: {error.strip()}")
    if rows % SCAN_LINES:
        chunks += 1
    progress.update(task, completed=path.stat().st_size,
                    detail=f"{chunks:,} chunks · {rows:,} rows · {matches:,} matches")


def filter_goa(gaf_path: Path, output_path: Path, evidence_codes: list[str],
               selected_go_terms: set[str] | None = None, max_proteins: int | None = None,
               chunk_size: int | None = None, num_workers: int | None = None):
    """Select proteins by distinct terms and write one row per protein-GO pair.

    A temporary disk-backed table keeps memory bounded even for the complete
    GOA archive. Multiple evidence records for the same pair do not increase
    a protein's annotation-density rank.
    """
    gaf_path, output_path = Path(gaf_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    size = gaf_path.stat().st_size
    codes = set(evidence_codes)
    fd, database_path = tempfile.mkstemp(prefix=".goa_pairs.", suffix=".sqlite", dir=output_path.parent)
    os.close(fd)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    os.close(fd)
    try:
        with sqlite3.connect(database_path) as db, make_progress() as progress:
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("CREATE TABLE pairs (protein TEXT NOT NULL, term TEXT NOT NULL, "
                       "PRIMARY KEY (protein, term)) WITHOUT ROWID")
            task = progress.add_task("GOA unique pairs", total=size, detail="0 rows")
            batch = []
            batch_limit = min(max(1, chunk_size or 50_000), 50_000)
            for pair in _scan_gaf(gaf_path, codes, selected_go_terms, progress, task,
                                  num_workers=num_workers):
                batch.append(pair)
                if len(batch) >= batch_limit:
                    db.executemany("INSERT OR IGNORE INTO pairs VALUES (?, ?)", batch)
                    db.commit()
                    batch.clear()
            if batch:
                db.executemany("INSERT OR IGNORE INTO pairs VALUES (?, ?)", batch)
                db.commit()
            unique_pairs = db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
            db.execute("CREATE TABLE selected (protein TEXT PRIMARY KEY) WITHOUT ROWID")
            limit = " LIMIT ?" if max_proteins is not None else ""
            query = ("SELECT protein FROM pairs GROUP BY protein "
                     "ORDER BY COUNT(*) DESC, protein ASC" + limit)
            parameters = (max_proteins,) if max_proteins is not None else ()
            selected = [row[0] for row in db.execute(query, parameters)]
            db.executemany("INSERT INTO selected VALUES (?)", ((pid,) for pid in selected))
            selected_pairs = 0
            with open(temporary, "w", encoding="utf-8") as destination:
                destination.write("DB_Object_ID\tGO_ID\n")
                for protein, term in db.execute(
                    "SELECT pairs.protein, pairs.term FROM pairs JOIN selected USING (protein) "
                    "ORDER BY pairs.protein, pairs.term"
                ):
                    destination.write(f"{protein}\t{term}\n")
                    selected_pairs += 1
            os.replace(temporary, output_path)
        logger.info("Saved %s unique pairs for %s proteins (from %s matched pairs) to %s",
                    selected_pairs,
                    len(selected), unique_pairs, output_path)
    finally:
        for path in (database_path, temporary):
            if os.path.exists(path):
                os.unlink(path)
    return pd.read_csv(output_path, sep="\t")


def clean_ontology_edges(ontology):
    cleaned = ontology.copy()
    cleaned.remove_edges_from(
        (source, target, key) for source, target, key in list(cleaned.edges(keys=True))
        if key not in {"is_a", "part_of"}
        or cleaned.nodes[source].get("namespace") != cleaned.nodes[target].get("namespace"))
    return cleaned


def calculate_ia(goa_path: Path, ontology, selected_terms: set[str], progress):
    """Calculate IA from unique propagated terms per protein, with one pseudocount."""
    parents = {term: set(ontology.successors(term))
               for term in selected_terms if term in ontology}
    needed = selected_terms | set().union(*parents.values()) if parents else selected_terms
    grouped = defaultdict(set)
    with open(goa_path, encoding="utf-8") as source:
        total = max(0, sum(1 for _ in source) - 1)
    task = progress.add_task("IA annotations", total=total, detail="0 proteins")
    with open(goa_path, encoding="utf-8") as source:
        next(source)
        for index, line in enumerate(source, 1):
            protein, term = line.rstrip("\n").split("\t")
            if term in ontology:
                grouped[protein].add(term)
            if index % 10_000 == 0:
                progress.update(task, completed=index, detail=f"{len(grouped):,} proteins")
    progress.update(task, completed=total, detail=f"{len(grouped):,} proteins")
    ancestor_cache = {}
    child_counts, parent_counts = Counter(), Counter()
    task = progress.add_task("IA proteins", total=len(grouped), detail="0 terms")
    for index, terms in enumerate(grouped.values(), 1):
        propagated = set()
        for term in terms:
            if term not in ancestor_cache:
                ancestor_cache[term] = (nx.descendants(ontology, term) | {term}) & needed
            propagated.update(ancestor_cache[term])
        child_counts.update(propagated & selected_terms)
        for term, direct_parents in parents.items():
            if direct_parents and direct_parents <= propagated:
                parent_counts[term] += 1
        if index % 100 == 0:
            progress.update(task, completed=index,
                            detail=f"{index:,}/{len(grouped):,} proteins · {len(child_counts):,} terms")
    progress.update(task, completed=len(grouped),
                    detail=f"{len(grouped):,} proteins · {len(child_counts):,} terms")
    return {
        term: -math.log2((child_counts[term] + 1) / (parent_counts[term] + 1))
        if parents.get(term) and parent_counts[term] >= child_counts[term] else 0.0
        for term in selected_terms
    }


def run_preprocess_goa(config: AppConfig, processed_dir: Path, force: bool = False,
                       num_workers: int | None = None, chunk_size: int | None = None):
    processed_dir.mkdir(parents=True, exist_ok=True)
    filtered = processed_dir / "goa_filtered.tsv"
    ia_pkl = processed_dir / "ia_values.pkl"
    ia_tsv = processed_dir / "ia_values.tsv"
    from aid2go.data.cache import stage_current, record_stage
    expected = (filtered,) if config.data.dataset_version.startswith("v2") else (filtered, ia_pkl, ia_tsv)
    if not force and all(path.exists() for path in expected) and (
        not config.data.dataset_version.startswith("v2") or stage_current("goa", config, processed_dir)
    ):
        print(f"✓ [CACHE] GOA & IA artifacts already exist in {processed_dir}. Skipping GOA preprocessing.")
        return
    nodes, obo = processed_dir / "go_nodes.csv", processed_dir / "go-basic.obo"
    if not nodes.exists() or not obo.exists():
        raise FileNotFoundError("GO nodes or ontology missing. Run preprocess go first.")
    selected_terms = set(pd.read_csv(nodes, usecols=["term"])["term"])
    gaf = processed_dir / config.data.goa_url.rsplit("/", 1)[-1]
    if config.data.goa_archive_path:
        shared = Path(config.data.goa_archive_path)
        if not shared.is_absolute():
            shared = ProjectPaths(config.paths.root_dir).root / shared
        if shared.exists():
            try:
                verify_gzip(shared)
                gaf = shared
            except (EOFError, gzip.BadGzipFile, OSError, zlib.error):
                logger.warning("Shared GOA archive failed validation; obtaining a new copy")
                download_file(config.data.goa_url, gaf)
        else:
            download_file(config.data.goa_url, gaf)
    else:
        download_file(config.data.goa_url, gaf)
    codes = [code for group in config.data.evidence_codes.values() for code in group]
    if not codes:
        codes = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC"]
    filter_goa(gaf, filtered, codes, selected_terms, config.data.max_proteins,
               chunk_size=chunk_size or config.data.gaf_chunk_size,
               num_workers=num_workers if isinstance(num_workers, int) else config.data.num_workers)
    if config.data.dataset_version.startswith("v2"):
        # IA belongs to the sequence-valid cohort, which is only known after
        # UniProt retrieval in preprocess_proteins.
        ia_pkl.unlink(missing_ok=True)
        ia_tsv.unlink(missing_ok=True)
    else:
        save_ia_from_final_cohort(processed_dir)
    record_stage("goa", config, processed_dir)
    logger.info("GOA unique-pair preprocessing complete.")


def save_ia_from_final_cohort(processed_dir: Path):
    """Refresh IA whenever the final protein/annotation cohort changes."""
    filtered = processed_dir / "goa_filtered.tsv"
    ia_pkl = processed_dir / "ia_values.pkl"
    ia_tsv = processed_dir / "ia_values.tsv"
    nodes, obo = processed_dir / "go_nodes.csv", processed_dir / "go-basic.obo"
    selected_terms = set(pd.read_csv(nodes, usecols=["term"])["term"])
    ontology = clean_ontology_edges(obonet.read_obo(str(obo), ignore_obsolete=True))
    with make_progress() as progress:
        ia = calculate_ia(filtered, ontology, selected_terms, progress)
    fd, temporary = tempfile.mkstemp(prefix=".ia_values.", dir=processed_dir)
    os.close(fd)
    try:
        with open(temporary, "wb") as destination:
            pickle.dump(ia, destination)
        os.replace(temporary, ia_pkl)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    fd, temporary = tempfile.mkstemp(prefix=".ia_values.", dir=processed_dir)
    os.close(fd)
    try:
        pd.DataFrame(sorted(ia.items()), columns=["term", "ia"]).to_csv(
            temporary, sep="\t", index=False)
        os.replace(temporary, ia_tsv)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    logger.info("Saved %s IA values", len(ia))
