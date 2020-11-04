"""Utils functions."""
from pathlib import Path


def ensure_dir(file_path):
    """Create directory to save file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def add_emodel(filename, emodel):
    """Add emodel suffix to filename."""
    return str(Path(filename).with_suffix("")) + "_" + emodel + str(Path(filename).suffix)
