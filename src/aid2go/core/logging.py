import logging
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

console = Console()

def setup_pipeline_logging(
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
    quiet_third_party: bool = True
) -> logging.Logger:
    """
    Configures structured logging for Aid2Go pipelines.
    
    Attaches a RichHandler for colorful, human-readable terminal output and an optional
    FileHandler for companion dataset provenance logging (e.g. data/processed/v1/pipeline.log).
    """
    logger = logging.getLogger("aid2go")
    logger.setLevel(level)
    
    # Avoid duplicate handlers if setup is called multiple times
    logger.handlers.clear()

    # Rich terminal handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        markup=True
    )
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)

    # Companion file logging
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.debug(f"Attached companion file logger at {log_path.resolve()}")

    if quiet_third_party:
        for lib in ["urllib3", "requests", "transformers", "torch", "matplotlib"]:
            logging.getLogger(lib).setLevel(logging.WARNING)

    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Returns a namespaced logger under 'aid2go'."""
    if name is None or name == "aid2go":
        return logging.getLogger("aid2go")
    if name.startswith("aid2go."):
        return logging.getLogger(name)
    return logging.getLogger(f"aid2go.{name}")

def log_scientific_stage(
    stage_num: str,
    title: str,
    description: str,
    biological_context: str,
    mathematical_concept: Optional[str] = None
):
    """
    Displays a scientific context panel in the terminal and records domain rationale to logs.
    
    Explains the biological significance, mathematical grounding, and functional role of the
    current stage towards protein function prediction.
    """
    logger = get_logger("pipeline")
    
    panel_content = Text()
    panel_content.append(f"{title}\n\n", style="bold cyan")
    panel_content.append("🎯 Objective:\n", style="bold yellow")
    panel_content.append(f"  {description}\n\n")
    panel_content.append("🧬 Biological Context:\n", style="bold green")
    panel_content.append(f"  {biological_context}\n")
    
    if mathematical_concept:
        panel_content.append("\n📐 Mathematical & Computational Foundation:\n", style="bold magenta")
        panel_content.append(f"  {mathematical_concept}\n")

    console.print(Panel(panel_content, title=f"[bold white]Step {stage_num}[/bold white]", border_style="bright_blue"))
    
    logger.info(f"========== [Step {stage_num}] {title} ==========")
    logger.info(f"Objective: {description}")
    logger.info(f"Biological Context: {biological_context}")
    if mathematical_concept:
        logger.info(f"Mathematical Foundation: {mathematical_concept}")
