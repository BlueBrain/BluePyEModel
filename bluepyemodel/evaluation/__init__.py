"""Evaluation module

In addition, if the env variable USE_NEURODAMUS exists, it will load mechanisms
from the neurodamus modules, as done in BGLibPy. The functions below are adapted
from bglibpy.importer.py.
"""
import os


def check_hoc_lib():
    """Check of hoc library path is present"""

    if "HOC_LIBRARY_PATH" not in os.environ:
        raise Exception(
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
    import imp

    nrnpy_path = os.path.join(imp.find_module("neuron")[1])
    import glob

    hoc_so_list = glob.glob(os.path.join(nrnpy_path, "hoc*.so"))

    if len(hoc_so_list) != 1:
        raise Exception("hoc shared library not found in %s" % nrnpy_path)

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
