"""Morphology related functions"""
import logging
import pathlib

logger = logging.getLogger(__name__)


def name_morphology(radius, type_="cylindrical"):
    return f"{type_}_morphology_{radius:.4f}"


def cylindrical_morphology_generator(radius=5.0, output_dir="morphologies"):
    """Creates a cylindrical morphology and save it as a .swc"""

    name = name_morphology(radius, type_="cylindrical")
    morphology_path = pathlib.Path(output_dir) / f"{name}.swc"
    morphology_path.parent.mkdir(parents=True, exist_ok=True)

    content = f"# Cylindrical morphology of radius {radius}\n"
    content += f"1 1 -{radius} 0.0 0.0 {radius} -1\n"
    content += f"2 1 0.0 0.0 0.0 {radius} 1\n"
    content += f"3 1 {radius} 0.0 0.0 {radius} 2\n"

    with open(str(morphology_path), "w+") as fp:
        fp.write(content)

    return morphology_path.resolve()
