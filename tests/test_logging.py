import logging
from pathlib import Path
from aid2go.core.logging import setup_pipeline_logging, get_logger, log_scientific_stage

def test_setup_pipeline_logging(tmp_path: Path):
    log_file = tmp_path / "test_pipeline.log"
    logger = setup_pipeline_logging(log_file=log_file, level=logging.DEBUG)

    logger.info("Testing structured info log message")
    logger.debug("Testing debug level detail")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Testing structured info log message" in content
    assert "Testing debug level detail" in content
    assert "[INFO]" in content or "INFO" in content

def test_log_scientific_stage(tmp_path: Path):
    log_file = tmp_path / "companion.log"
    setup_pipeline_logging(log_file=log_file, level=logging.INFO)

    log_scientific_stage(
        stage_num="2/5",
        title="GOA & Information Accretion Test",
        description="Calculate IA values for terms",
        biological_context="Information gain over DAG parents",
        mathematical_concept="IA(t) = -log2(P(t)/P(parents))"
    )

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "GOA & Information Accretion Test" in content
    assert "IA(t) = -log2(P(t)/P(parents))" in content
