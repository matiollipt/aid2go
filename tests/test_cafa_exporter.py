import pytest
from pathlib import Path
import pandas as pd
from aid2go.infer.summaries import export_cafa_submission

def test_export_cafa_submission(tmp_path):
    predictions = {
        "P12345": {"GO:0008150": 0.9523, "GO:0003674": 0.4321},
        "P67890": {"GO:0005575": 0.8888}
    }
    out_file = tmp_path / "submission.tsv"

    export_cafa_submission(predictions, out_file, min_threshold=0.01)

    assert out_file.exists()
    lines = out_file.read_text().strip().split("\n")
    assert len(lines) == 3

    # Verify tab format: Protein_ID \t GO_ID \t Score formatted to 3 decimal places
    first_row = lines[0].split("\t")
    assert first_row[0] == "P12345"
    assert first_row[1] == "GO:0008150"
    assert first_row[2] == "0.952"


def test_top_k_keeps_ancestors_before_equal_score_children(tmp_path):
    predictions = {"P": {"GO:child": 0.8, "GO:parent": 0.8, "GO:root": 0.8}}
    target = tmp_path / "submission.tsv"
    export_cafa_submission(predictions, target, top_k=2,
                           term_order={"GO:root": 0, "GO:parent": 1, "GO:child": 2})
    assert [line.split("\t")[1] for line in target.read_text().splitlines()] == ["GO:root", "GO:parent"]
