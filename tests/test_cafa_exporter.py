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
