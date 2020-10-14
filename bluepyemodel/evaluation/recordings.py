"""Module with recording classes and functions."""

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

import numpy
from bluepyopt import ephys

logger = logging.getLogger(__name__)


def get_loc_ions(isection):
    """Get all ion concentrations available in a location."""
    local_overall_ions = set()

    # ion overall current & concentration
    ions = isection.psection()["ions"]
    for _, ion in ions.items():
        for var in ion.keys():
            # concentration should have 'i' at the end (e.g. ki, cai, nai, ...)
            if var[-1] == "i":
                local_overall_ions.add(var)

    return local_overall_ions


def get_loc_currents(isection):
    """Get all overall currents available in a location."""
    local_overall_currs = set()

    # ion overall current & concentration
    ions = isection.psection()["ions"]
    for _, ion in ions.items():
        for var in ion.keys():
            # current should have 'i' at the beginning (e.g. ik, ica, ina, ...)
            if var[0] == "i":
                local_overall_currs.add(var)

    return local_overall_currs


def get_loc_varlist(isection):
    """Get all possible variables in a location."""
    local_varlist = []

    # currents etc.
    raw_dict = isection.psection()["density_mechs"]
    for channel, vars_ in raw_dict.items():
        for var in vars_.keys():
            local_varlist.append("_".join((var, channel)))

    local_varlist.append("v")

    return local_varlist


def get_i_membrane(isection):
    """Look for i_membrane in a location."""
    raw_dict = isection.psection()["density_mechs"]
    if "extracellular" in raw_dict:
        if "i_membrane" in raw_dict["extracellular"]:
            return ["i_membrane"]
    return []


def check_recordings(recordings, icell, sim):
    """Returns a list of valid recordings (where the variable is in the location)."""

    new_recs = []  # to return
    varlists = {}  # keep varlists to avoid re-computing them every time

    for rec in recordings:
        # get section from location
        try:
            seg = rec.location.instantiate(sim=sim, icell=icell)
        except ephys.locations.EPhysLocInstantiateException:
            continue
        sec = seg.sec
        section_key = str(sec)

        # get list of variables available in the section
        if section_key in varlists:
            local_varlist = varlists[section_key]
        else:
            local_varlist = (
                get_loc_varlist(sec)
                + list(get_loc_ions(sec))
                + list(get_loc_currents(sec))
                + get_i_membrane(sec)
            )
            varlists[section_key] = local_varlist

        # keep recording if its variable is available in its location
        if rec.variable in local_varlist:
            rec.checked = True
            new_recs.append(rec)

    return new_recs


class LooseDtRecordingCustom(ephys.recordings.CompRecording):
    """Recording that can be checked, but that do not records at fixed dt."""

    def __init__(self, name=None, location=None, variable="v"):
        """Constructor.

        Args:
            name (str): name of this object
            location (Location): location in the model of the recording
            variable (str): which variable to record from (e.g. 'v')
        """
        super().__init__(name=name, location=location, variable=variable)

        # important to turn current densities into currents
        self.segment_area = None
        # important to detect ion concentration variable
        self.local_ion_list = None
        self.checked = False

    def instantiate(self, sim=None, icell=None):
        """Instantiate recording."""
        logger.debug("Adding compartment recording of %s at %s", self.variable, self.location)

        self.varvector = sim.neuron.h.Vector()
        seg = self.location.instantiate(sim=sim, icell=icell)
        self.varvector.record(getattr(seg, f"_ref_{self.variable}"))

        self.segment_area = seg.area()
        self.local_ion_list = get_loc_ions(seg.sec)

        self.tvector = sim.neuron.h.Vector()
        self.tvector.record(sim.neuron.h._ref_t)  # pylint: disable=W0212

        self.instantiated = True

    @property
    def response(self):
        """Return recording response. Turn current densities into currents."""

        if not self.instantiated:
            return None

        # do not modify voltage or ion concentration
        if self.variable == "v" or self.variable in self.local_ion_list:
            return ephys.responses.TimeVoltageResponse(
                self.name, self.tvector.to_python(), self.varvector.to_python()
            )

        # ionic current: turn mA/cm2 (*um2) into pA
        return ephys.responses.TimeVoltageResponse(
            self.name,
            self.tvector.to_python(),
            numpy.array(self.varvector.to_python()) * self.segment_area * 10.0,
        )


class FixedDtRecordingCustom(LooseDtRecordingCustom):
    """Recording that can be checked, with recording every 0.1 ms."""

    def instantiate(self, sim=None, icell=None):
        """Instantiate recording."""
        logger.debug("Adding compartment recording of %s at %s", self.variable, self.location)

        self.varvector = sim.neuron.h.Vector()
        seg = self.location.instantiate(sim=sim, icell=icell)
        self.varvector.record(getattr(seg, f"_ref_{self.variable}"), 0.1)

        self.segment_area = seg.area()
        self.local_ion_list = get_loc_ions(seg.sec)

        self.tvector = sim.neuron.h.Vector()
        self.tvector.record(sim.neuron.h._ref_t, 0.1)  # pylint: disable=W0212

        self.instantiated = True
