"""Utils functions."""
from pathlib import Path

from bluepyemodel.access_point.local import LocalAccessPoint


def ensure_dir(file_path):
    """Create directory to save file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def get_database(api_config):
    """Fetch emodel AP."""
    if api_config.api == "local":
        return LocalAccessPoint(
            emodel="cADpyr_L5TPC",
            emodel_dir=api_config.emodel_dir,
            final_path=api_config.final_path,
            legacy_dir_structure=True,
            with_seeds=True,
        )
    raise NotImplementedError(f"api {api_config.api} is not implemented")
