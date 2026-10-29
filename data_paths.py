"""Repository-relative input locations and separate generated-output folders."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPOSITORY_ROOT / "Data"
OUTPUT_ROOT = REPOSITORY_ROOT / "output"


def output_dir(*parts: str) -> Path:
    """Create and return a generated-output folder outside Data."""
    path = OUTPUT_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


# The preserved audit-analysis module imports these two names. Its input files
# retain their original dataset layout inside Data/audit.
EXP_RAW_DATA = DATA_ROOT / "audit" / "exp_raw_data"


def intermediate_dir(dataset: str, *parts: str, create: bool = True) -> Path:
    path = DATA_ROOT / "audit" / "intermediate_analysis_data" / dataset
    path = path.joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
