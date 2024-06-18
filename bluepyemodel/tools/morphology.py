"""Morphology related functions"""

"""
Copyright 2023-2024 Blue Brain Project / EPFL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import pathlib

logger = logging.getLogger(__name__)


def name_morphology(radius, type_="cylindrical"):
    return f"{type_}_morphology_{radius:.4f}"


def cylindrical_morphology_generator(
    radius=5.0, radius_axon=0.0, length_axon=0.0, output_dir="morphologies"
):
    """Creates a cylindrical morphology and save it as a .swc"""

    name = name_morphology(radius, type_="cylindrical")
    morphology_path = pathlib.Path(output_dir) / f"{name}.swc"
    morphology_path.parent.mkdir(parents=True, exist_ok=True)

    content = f"# Cylindrical morphology of radius {radius}\n"
    content += f"1 1 -{radius} 0.0 0.0 {radius} -1\n"
    content += f"2 1 0.0 0.0 0.0 {radius} 1\n"
    content += f"3 1 {radius} 0.0 0.0 {radius} 2\n"

    if radius_axon and length_axon:
        content += f"4 2 -{radius} 0.0 0.0 {radius_axon} 1\n"
        content += f"5 2 -{radius + length_axon} 0.0 0.0 {radius_axon} 4\n"

    with open(str(morphology_path), "w+") as fp:
        fp.write(content)

    return morphology_path.resolve()
