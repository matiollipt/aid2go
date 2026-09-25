"""Atomic HTTP downloads with optional continuation of a partial file."""

from __future__ import annotations

import os
from pathlib import Path

import requests


def download_file_resumable(url: str, target_path: Path, force: bool = False,
                            desc: str | None = None) -> Path:
    """Download to a sibling ``.part`` file, then replace the target atomically.

    The caller must validate domain-specific content after downloading. A
    server that ignores Range requests is handled by starting from byte zero.
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        return target
    part = target.with_name(target.name + ".part")
    if force:
        part.unlink(missing_ok=True)
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        append = response.status_code == 206 and offset > 0
        if append:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise ValueError(f"Unexpected Content-Range: {content_range}")
        with open(part, "ab" if append else "wb") as destination:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    destination.write(chunk)
    os.replace(part, target)
    return target
