from pathlib import Path

import pytest

from aid2go.cli.progress import make_progress, run_stage, track_items
from rich.progress import SpinnerColumn


def test_nested_progress_reuses_one_live_display():
    with make_progress() as outer:
        assert not any(isinstance(column, SpinnerColumn) for column in outer.columns)
        parent = outer.add_task("Pipeline", total=2, detail="preprocessing")
        with make_progress() as inner:
            assert inner is outer
            child = inner.add_task("GOA count", total=1, detail="one chunk")
            inner.advance(child)
        assert next(task for task in outer.tasks if task.id == parent).completed == 0
        assert next(task for task in outer.tasks if task.id == child).completed == 1
        assert list(track_items(range(3), "Processing", "chunks")) == [0, 1, 2]
        assert len(outer.tasks) == 2  # The temporary loop task is removed.
        outer.advance(parent)


def test_cached_stage_stays_visible_without_running(tmp_path):
    artifact = tmp_path / "done.txt"
    artifact.write_text("complete")
    calls = []
    with make_progress() as progress:
        run_stage(progress, "Cached work", lambda: calls.append(1), artifacts=(artifact,))
        assert len(progress.tasks) == 1
        assert progress.tasks[0].completed == 1
        assert "cached" in progress.tasks[0].fields["detail"]
    assert calls == []


def test_stage_missing_artifact_is_failed(tmp_path):
    with make_progress() as progress:
        with pytest.raises(RuntimeError, match="did not create"):
            run_stage(progress, "Broken work", lambda: None,
                      artifacts=(Path(tmp_path / "absent"),))
        assert "failed" in progress.tasks[0].fields["detail"]
