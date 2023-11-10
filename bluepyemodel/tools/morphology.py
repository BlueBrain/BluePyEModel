"""Morphology related functions"""

"""
Copyright 2023, EPFL/Blue Brain Project

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

from morphio import PointLevel
from morphio import SectionType
from morphio.mut import Morphology

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

    morph = Morphology()
    morph.soma.points = [[-radius, 0, 0], [0, 0, 0], [radius, 0, 0]]
    morph.soma.diameters = 3 * [2 * radius]

    if radius_axon and length_axon:
        morph.append_root_section(
            PointLevel([[-radius, 0, 0], [-radius - length_axon, 0, 0]], 2 * [2 * radius_axon]),
            SectionType.axon,
        )

    morph.write(morphology_path)
