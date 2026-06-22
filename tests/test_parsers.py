from aid2go.data.preprocess_goa import filter_goa
import pandas as pd

def test_gaf_filter(tmp_path):
    # Create dummy GAF
    gaf_content = [
        "!header",
        "UniProtKB\tP12345\t\t\tGO:0001\t\tEXP\t\tF\t\t\t\t\t\t\t",
        "UniProtKB\tP67890\t\t\tGO:0002\t\tIEA\t\tF\t\t\t\t\t\t\t"
    ]
    gaf_file = tmp_path / "test.gaf"
    with open(gaf_file, "w") as f:
        f.write("\n".join(gaf_content))
        
    out_file = tmp_path / "filtered.tsv"
    filter_goa(gaf_file, out_file, ["EXP"])
    
    df = pd.read_csv(out_file, sep="\t")
    assert len(df) == 1
    assert df.iloc[0]["DB_Object_ID"] == "P12345"
