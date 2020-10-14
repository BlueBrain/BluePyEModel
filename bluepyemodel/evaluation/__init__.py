"""Evaluation module

In addition, if the env variable USE_NEURODAMUS exists, it will load mechanisms
from the neurodamus modules, as done in BGLibPy. The functions below are adapted
from bglibpy.importer.py.
"""

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

import os


def check_hoc_lib():
    """Check of hoc library path is present"""

    if "HOC_LIBRARY_PATH" not in os.environ:
        raise RuntimeError(
            "BGLibPy: HOC_LIBRARY_PATH not found, this is required to find "
            "Neurodamus. Did you install neurodamus correctly ?"
        )
    return os.environ["HOC_LIBRARY_PATH"]


def import_mod_lib(_neuron):
    """Import mod files"""

    mod_lib_list = None
    if "BGLIBPY_MOD_LIBRARY_PATH" in os.environ:
        mod_lib_path = os.environ["BGLIBPY_MOD_LIBRARY_PATH"]
        mod_lib_list = mod_lib_path.split(":")
        for mod_lib in mod_lib_list:
            _neuron.h.nrn_load_dll(mod_lib)

    return mod_lib_list


def _nrn_disable_banner():
    """Disable Neuron banner"""

    import ctypes
    import importlib

    nrnpy_path = os.path.join(importlib.util.find_spec("neuron").submodule_search_locations[0])
    import glob

    hoc_so_list = glob.glob(os.path.join(nrnpy_path, "hoc*.so"))

    if len(hoc_so_list) != 1:
        raise FileNotFoundError(f"hoc shared library not found in {nrnpy_path}")

    hoc_so = hoc_so_list[0]
    nrndll = ctypes.cdll[hoc_so]
    ctypes.c_int.in_dll(nrndll, "nrn_nobanner_").value = 1


def import_neurodamus(_neuron):
    """Import neurodamus"""
    _neuron.h("objref simConfig")

    _neuron.h.load_file("stdrun.hoc")
    _neuron.h.load_file("defvar.hoc")
    _neuron.h.default_var("simulator", "NEURON")
    _neuron.h.load_file("Cell.hoc")
    _neuron.h.load_file("TDistFunc.hoc")
    _neuron.h.load_file("SerializedSections.hoc")
    _neuron.h.load_file("TStim.hoc")
    _neuron.h.load_file("ShowProgress.hoc")
    _neuron.h.load_file("SimSettings.hoc")
    _neuron.h.load_file("RNGSettings.hoc")

    _neuron.h("obfunc new_IClamp() { return new IClamp($1) }")
    _neuron.h("objref p")
    _neuron.h("p = new PythonObject()")

    _neuron.h("simConfig = new SimSettings()")


if os.getenv("USE_NEURODAMUS"):
    # this will loads mechanisms if proper neurodamus module is loaded
    check_hoc_lib()
    _nrn_disable_banner()
    import neuron

    import_mod_lib(neuron)
    import_neurodamus(neuron)
