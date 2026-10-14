"""One calm, shared progress display for nested CLI stages."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.progress import (
    BarColumn, Progress,
    TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)

_active_progress: ContextVar[Progress | None] = ContextVar("aid2go_progress", default=None)


@contextmanager
def make_progress():
    """Nested stages add tasks to the active display instead of starting another Live."""
    active = _active_progress.get()
    if active is not None:
        yield active
        return

    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=18),
        TaskProgressColumn(),
        TextColumn("{task.fields[detail]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(),
        refresh_per_second=2,
        transient=False,
    )
    token = _active_progress.set(progress)
    try:
        with progress:
            yield progress
    finally:
        _active_progress.reset(token)


def track_items(items: Iterable, description: str, unit: str = "items") -> Iterator:
    """Show one temporary task for a loop, including when called inside a pipeline."""
    try:
        total = len(items)
    except TypeError:
        total = None
    with make_progress() as progress:
        task = progress.add_task(description, total=total, detail=f"0 {unit}")
        completed = 0
        try:
            for item in items:
                yield item
                completed += 1
                progress.update(task, completed=completed, detail=f"{completed:,} {unit}")
        finally:
            progress.remove_task(task)


def run_stage(progress: Progress, description: str, runner: Callable[[], object], *,
              artifacts: tuple[Path, ...] = (), force: bool = False,
              task_id: int | None = None,
              cache_valid: Callable[[], bool] | None = None) -> None:
    """Keep one visible row for a stage, including cached and failed stages."""
    if task_id is None:
        task_id = progress.add_task(description, total=1, detail="pending")
    cached = (bool(artifacts) and not force and all(path.exists() for path in artifacts)
              and (cache_valid is None or cache_valid()))
    if cached:
        progress.update(task_id, completed=1, detail="[cyan]cached[/cyan]")
        return
    progress.reset(task_id, total=1, start=True, detail="running")
    existing_tasks = {task.id for task in progress.tasks}
    try:
        runner()
        missing = [path for path in artifacts if not path.exists()]
        if missing:
            raise RuntimeError(f"{description} did not create: {', '.join(map(str, missing))}")
    except Exception:
        progress.stop_task(task_id)
        progress.update(task_id, detail="[red]failed[/red]")
        raise
    finally:
        # Work counters can be numerous (GOA scans, embeddings); only stage
        # outcomes stay on screen after each stage finishes.
        for task in list(progress.tasks):
            if task.id not in existing_tasks:
                progress.remove_task(task.id)
    progress.stop_task(task_id)
    elapsed = progress.tasks[progress.task_ids.index(task_id)].elapsed or 0
    progress.update(task_id, completed=1, detail=f"[green]done ({elapsed:.1f}s)[/green]")
