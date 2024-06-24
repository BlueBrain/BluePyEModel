"""Functions for morphology modifications in evaluator."""

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

import numpy as np

logger = logging.getLogger(__name__)
ZERO = 1e-6


def taper_function(distance, strength, taper_scale, terminal_diameter, scale=1.0):
    """Function to model tappered AIS."""
    return strength * np.exp(-distance / taper_scale) + terminal_diameter * scale


def synth_axon(sim=None, icell=None, params=None, scale=1.0):
    """Replace axon with tappered axon initial segment.

    Args:
        sim and icell: neuron related arguments
        params (list): fixed parameter for an emodel, should be length, strenght, taper_scale and
            terminal_diameter
        scale (floag): scale parameter for each cell
    """
    for section in icell.axonal:
        sim.neuron.h.delete_section(sec=section)

    sim.neuron.h.execute("create axon[2]", icell)

    nseg_total = 10
    L_target = params[0]
    diameters = taper_function(np.linspace(0, L_target, nseg_total), *params[1:], scale=scale)
    count = 0
    for section in icell.axon:
        section.nseg = nseg_total // 2
        section.L = L_target / 2
        for seg in section:
            seg.diam = diameters[count]
            count += 1

        icell.axonal.append(sec=section)
        icell.all.append(sec=section)

    icell.axon[0].connect(icell.soma[0], 1.0, 0.0)
    icell.axon[1].connect(icell.axon[0], 1.0, 0.0)

    sim.neuron.h.execute("create myelin[1]", icell)
    icell.myelinated.append(sec=icell.myelin[0])
    icell.all.append(sec=icell.myelin[0])
    icell.myelin[0].nseg = 5
    icell.myelin[0].L = 1000
    icell.myelin[0].diam = diameters[-1]  # this assigns the value of terminal_diameter
    icell.myelin[0].connect(icell.axon[1], 1.0, 0.0)


def get_synth_axon_hoc(params):
    # pylint: disable=consider-using-f-string
    return """
proc replace_axon(){ local count, i1, i2, L_target, strenght, taper_scale, terminal_diameter localobj diams

    access axon[0]
    axon[0] i1 = v(0.0001) // used when serializing sections prior to sim start
    axon[1] i2 = v(0.0001) // used when serializing sections prior to sim start
    axon[2] i3 = v(0.0001) // used when serializing sections prior to sim start

    // get rid of the old axon
    forsec axonal{delete_section()}
    execute1("create axon[2]", CellRef)

    // creating diameter profile
    nseg_total = 10
    L_target = %s
    strength = %s
    taper_scale = %s
    terminal_diameter = %s
    scale = $1
    diams = new Vector()
    count = 0
    for i=0,nseg_total{
        count = count + 1
        diams.resize(count)
        diams.x[count-1] = strength * exp(-L_target * i / nseg_total / taper_scale) + terminal_diameter * scale
    }

    // assigning diameter to axon
    count = 0
    for i=0,1{
        access axon[i]
        L =  L_target/2
        nseg = nseg_total/2
        for (x) {
            if (x > 0 && x < 1) {
                diam(x) = diams.x[count]
                count = count + 1
            }
        }
        all.append()
        axonal.append()

        if (i == 0) {
            v(0.0001) = i1
        } else {
            v(0.0001) = i2
        }


    }

    soma[0] connect axon[0](0), 1
    axon[0] connect axon[1](0), 1

    // add myelin part
    create myelin[1]
    access myelin{
            L = 1000
            diam = diams.x[count-1]
            nseg = 5
            v(0.0001) = i3
            all.append()
            myelinated.append()
    }
    connect myelin(0), axon[1](1)
}
    """ % tuple(
        params
    )


def replace_axon_with_taper(sim=None, icell=None):
    """Replace axon with tappered axon initial segment"""
    if len(list(icell.axonal)) < 3:
        raise ValueError(
            "Less than three axon sections are present! "
            "The replace_axon_with_taper morph modifier cannot be applied to such a morphology. "
            "Please use 'bluepyopt_replace_axon' or other valid modifiers from "
            "https://github.com/BlueBrain/BluePyEModel/blob/main/"
            "bluepyemodel/evaluation/modifiers.py or a custom modifier."
        )

    L_target = 60  # length of stub axon
    nseg0 = 5  # number of segments for each of the two axon sections

    nseg_total = nseg0 * 2
    chunkSize = L_target / nseg_total

    diams = []
    lens = []

    count = 0
    for section in icell.axonal:
        L = section.L
        nseg = 1 + int(L / chunkSize / 2.0) * 2  # nseg to get diameter
        section.nseg = nseg

        for seg in section:
            count = count + 1
            diams.append(seg.diam)
            lens.append(L / nseg)
            if count == nseg_total:
                break
        if count == nseg_total:
            break

    for section in icell.axonal:
        sim.neuron.h.delete_section(sec=section)

    #  new axon array
    sim.neuron.h.execute("create axon[2]", icell)

    L_real = 0
    count = 0
    for _, section in enumerate(icell.axon):
        section.nseg = nseg_total // 2
        section.L = L_target / 2

        for seg in section:
            if count >= len(diams):
                break
            seg.diam = diams[count]
            L_real = L_real + lens[count]
            count = count + 1

        icell.axonal.append(sec=section)
        icell.all.append(sec=section)

        if count >= len(diams):
            break
    # childsec.connect(parentsec, parentx, childx)
    icell.axon[0].connect(icell.soma[0], 1.0, 0.0)
    icell.axon[1].connect(icell.axon[0], 1.0, 0.0)

    sim.neuron.h.execute("create myelin[1]", icell)
    icell.myelinated.append(sec=icell.myelin[0])
    icell.all.append(sec=icell.myelin[0])
    icell.myelin[0].nseg = 5
    icell.myelin[0].L = 1000
    icell.myelin[0].diam = diams[count - 1]
    icell.myelin[0].connect(icell.axon[1], 1.0, 0.0)

    logger.debug(
        "Replace axon with tapered AIS of length %f, target length was %f, diameters are %s",
        L_real,
        L_target,
        diams,
    )


def remove_soma(sim=None, icell=None):
    """Remove the soma and connect dendrites together.

    For this to work, we leave the soma connected to the axon,
    and with diameter 1e-6. BluePyOp requires a soma for
    parameter scaling, and NEURON fails is the soma size is =0.
    """
    for section in icell.basal:
        if section.parentseg().sec in list(icell.soma):
            sim.neuron.h.disconnect(section)
            section.connect(icell.axon[0])

    for section in icell.apical:
        if section.parentseg().sec in list(icell.soma):
            sim.neuron.h.disconnect(section)
            section.connect(icell.axon[0])

    for section in icell.soma:
        section.diam = ZERO

    logger.debug("Remove soma")


def isolate_soma(sim=None, icell=None):
    """Remove everything except the soma."""
    for section in icell.axonal:
        sim.neuron.h.delete_section(sec=section)
    for section in icell.basal:
        sim.neuron.h.delete_section(sec=section)
    for section in icell.apical:
        sim.neuron.h.delete_section(sec=section)

    logger.debug("Keep only soma")


def remove_axon(sim=None, icell=None):  # pylint: disable=unused-argument
    """Remove the axon.

    Removing the axonal branches breaks the code, so we set it to ZERO
    """
    for section in icell.myelin:
        sim.neuron.h.delete_section(sec=section)
    for section in icell.axonal:
        section.diam = ZERO


def isolate_axon(sim=None, icell=None):
    """Remove everything except the axon."""
    for section in icell.basal:
        sim.neuron.h.delete_section(sec=section)
    for section in icell.apical:
        sim.neuron.h.delete_section(sec=section)
    for section in icell.soma:
        sim.neuron.h.delete_section(sec=section)


replace_axon_hoc = """
    proc replace_axon(){ local nSec, L_chunk, dist, i1, i2, count, L_target, chunkSize, L_real localobj diams, lens

        L_target = 60  // length of stub axon
        nseg0 = 5  // number of segments for each of the two axon sections

        nseg_total = nseg0 * 2
        chunkSize = L_target/nseg_total

        nSec = 0
        forsec axonal{nSec = nSec + 1}

        // Try to grab info from original axon
        if(nSec < 3){ //At least two axon sections have to be present!

            execerror("Less than three axon sections are present! This emodel can't be run with such a morphology!")

        } else {

            diams = new Vector()
            lens = new Vector()

            access axon[0]
            axon[0] i1 = v(0.0001) // used when serializing sections prior to sim start
            axon[1] i2 = v(0.0001) // used when serializing sections prior to sim start
            axon[2] i3 = v(0.0001) // used when serializing sections prior to sim start

            count = 0
            forsec axonal{ // loop through all axon sections

                nseg = 1 + int(L/chunkSize/2.)*2  //nseg to get diameter

                for (x) {
                    if (x > 0 && x < 1) {
                        count = count + 1
                        diams.resize(count)
                        diams.x[count-1] = diam(x)
                        lens.resize(count)
                        lens.x[count-1] = L/nseg
                        if( count == nseg_total ){
                            break
                        }
                    }
                }
                if( count == nseg_total ){
                    break
                }
            }

            // get rid of the old axon
            forsec axonal{delete_section()}
            execute1("create axon[2]", CellRef)

            L_real = 0
            count = 0

            // new axon dependant on old diameters
            for i=0,1{
                access axon[i]
                L =  L_target/2
                nseg = nseg_total/2

                for (x) {
                    if (x > 0 && x < 1) {
                        diam(x) = diams.x[count]
                        L_real = L_real+lens.x[count]
                        count = count + 1
                    }
                }

                all.append()
                axonal.append()

                if (i == 0) {
                    v(0.0001) = i1
                } else {
                    v(0.0001) = i2
                }
            }

            nSecAxonal = 2
            soma[0] connect axon[0](0), 1
            axon[0] connect axon[1](0), 1

            create myelin[1]
            access myelin{
                    L = 1000
                    diam = diams.x[count-1]
                    nseg = 5
                    v(0.0001) = i3
                    all.append()
                    myelinated.append()
            }
            connect myelin(0), axon[1](1)
        }
    }
"""


def replace_axon_legacy(sim=None, icell=None):
    """Replace axon used in legacy thalamus project"""
    if len(list(icell.axonal)) < 2:
        raise ValueError(
            "Less than two axon sections are present! "
            "The replace_axon_legacy morph modifier cannot be applied to such a morphology. "
            "Please use 'bluepyopt_replace_axon' or other valid modifiers from "
            "https://github.com/BlueBrain/BluePyEModel/blob/main/"
            "bluepyemodel/evaluation/modifiers.py or a custom modifier."
        )

    L_target = 60  # length of stub axon
    nseg0 = 5  # number of segments for each of the two axon sections

    nseg_total = nseg0 * 2
    chunkSize = L_target / nseg_total

    diams = []
    lens = []

    count = 0
    for section in icell.axonal:
        L = section.L
        nseg = 1 + int(L / chunkSize / 2.0) * 2  # nseg to get diameter
        section.nseg = nseg

        for seg in section:
            diams.append(seg.diam)
            lens.append(L / nseg)
            if count == nseg_total:
                break
        if count == nseg_total:
            break

    for section in icell.axonal:
        sim.neuron.h.delete_section(sec=section)

    #  new axon array
    sim.neuron.h.execute("create axon[2]", icell)

    L_real = 0
    count = 0

    for section in icell.axon:
        section.nseg = nseg_total // 2
        section.L = L_target / 2

        for seg in section:
            seg.diam = diams[count]
            L_real = L_real + lens[count]
            count = count + 1
            # print seg.x, seg.diam
        icell.axonal.append(sec=section)
        icell.all.append(sec=section)

    icell.axon[0].connect(icell.soma[0], 1.0, 0.0)
    icell.axon[1].connect(icell.axon[0], 1.0, 0.0)

    logger.debug("Replace axon with tapered AIS")


replace_axon_legacy_hoc = """
proc replace_axon(){local nSec, L_chunk, dist, i1, i2, count, L_target, chunkSize, L_real localobj   diams, lens

    L_target = 60  // length of stub axon
    nseg0 = 5  // number of segments for each of the two axon sections

    nseg_total = nseg0 * 2
    chunkSize = L_target/nseg_total

    nSec = 0
    forsec axonal{nSec = nSec + 1}

    // Try to grab info from original axon
    //At least two axon sections have to be present!
    if(nSec < 2){
        execerror("Less than two axon sections are present! Add an axon to the morphology and try again!")
    } else {

        diams = new Vector()
        lens = new Vector()

        access axon[0]
        i1 = v(0.0001) // used when serializing sections prior to sim start
        access axon[1]
        i2 = v(0.0001) // used when serializing sections prior to sim start

        count = 0
        forsec axonal{ // loop through all axon sections

            nseg = 1 + int(L/chunkSize/2.)*2  //nseg to get diameter

            for (x) {
                if (x > 0 && x < 1) {
                    count = count + 1
                    diams.resize(count)
                    diams.x[count-1] = diam(x)
                    lens.resize(count)
                    lens.x[count-1] = L/nseg
                    if( count == nseg_total ){
                        break
                    }
                }
            }
            if( count == nseg_total ){
                break
            }
        }

        // get rid of the old axon
        forsec axonal{delete_section()}
        execute1("create axon[2]", CellRef)

        L_real = 0
        count = 0

        // new axon dependant on old diameters
        for i=0,1{

            access axon[i]
            L =  L_target/2
            nseg = nseg_total/2

            for (x) {
                if (x > 0 && x < 1) {
                    diam(x) = diams.x[count]
                    L_real = L_real+lens.x[count]
                    count = count + 1
                }
            }

            all.append()
            axonal.append()

            if (i == 0) {
                v(0.0001) = i1
            } else {
                v(0.0001) = i2
            }
        }

        nSecAxonal = 2
        soma[0] connect axon[0](0), 1
        axon[0] connect axon[1](0), 1

        //print 'Target stub axon length:', L_target, 'um, equivalent length: ', L_real 'um'
    }
}
"""


def replace_axon_olfactory_bulb(sim=None, icell=None):
    """Replace axon used in olfactory bulb models."""
    for section in icell.axonal:
        sim.neuron.h.delete_section(sec=section)

    sim.neuron.h.execute("create hillock", icell)
    sim.neuron.h.execute("create initialseg", icell)
    sim.neuron.h.execute("create node[5]", icell)
    sim.neuron.h.execute("create myelin[5]", icell)

    section = icell.hillock
    section.nseg = 3
    section.L = 5
    for seg in section:
        seg.diam = 10.0925
    icell.all.append(sec=section)
    icell.axonal.append(sec=section)

    section = icell.initialseg
    section.nseg = 3
    section.L = 30
    for seg in section:
        seg.diam = 1.5
    icell.all.append(sec=section)
    icell.axonal.append(sec=section)

    for _, section in enumerate(icell.node):
        section.L = 1
        for seg in section:
            seg.diam = 1

        icell.all.append(sec=section)
        icell.axonal.append(sec=section)

    for _, section in enumerate(icell.myelin):
        section.L = 1000
        for seg in section:
            seg.diam = 1.5

        icell.all.append(sec=section)
        icell.myelinated.append(sec=section)

    icell.hillock.connect(icell.soma[0], 0.0, 0.0)
    icell.initialseg.connect(icell.hillock, 1.0, 0.0)

    icell.myelin[0].connect(icell.initialseg, 1.0, 0.0)
    icell.node[0].connect(icell.myelin[0], 1.0, 0.0)
    icell.myelin[1].connect(icell.node[0], 1.0, 0.0)
    icell.node[1].connect(icell.myelin[1], 1.0, 0.0)
    icell.myelin[2].connect(icell.node[1], 1.0, 0.0)
    icell.node[2].connect(icell.myelin[2], 1.0, 0.0)
    icell.myelin[3].connect(icell.node[2], 1.0, 0.0)
    icell.node[3].connect(icell.myelin[3], 1.0, 0.0)
    icell.myelin[4].connect(icell.node[3], 1.0, 0.0)
    icell.node[4].connect(icell.myelin[4], 1.0, 0.0)


replace_axon_olfactory_bulb_hoc = """
proc replace_axon(){ local nSec, L_chunk, dist, i1, i2, count, L_target, chunkSize, L_real localobj diams, lens

    L_target = 60  // length of stub axon
    nseg0 = 5  // number of segments for each of the two axon sections

    nseg_total = nseg0 * 2
    chunkSize = L_target/nseg_total

    nSec = 0
    forsec axonal{nSec = nSec + 1}

    // Try to grab info from original axon
    if(nSec < 1){ //At least two axon sections have to be present!

        execerror("Less than two axon sections are present! Add an axon to the morphology and try again!")

    } else {

        diams = new Vector()
        lens = new Vector()

        access axon[0]
        i1 = v(0.0001) // used when serializing sections prior to sim start

        access axon[1]
        i2 = v(0.0001) // used when serializing sections prior to sim start

        count = 0

        forsec axonal{ // loop through all axon sections

            nseg = 1 + int(L/chunkSize/2.)*2  //nseg to get diameter

        for (x) {
            if (x > 0 && x < 1) {
                count = count + 1
                diams.resize(count)
                diams.x[count-1] = diam(x)
                lens.resize(count)
                lens.x[count-1] = L/nseg
                if( count == nseg_total ){
                    break
                }
            }
        }
        if( count == nseg_total ){
            break
        }
    }

    forsec axonal{delete_section()}
    execute1("create hillock", CellRef)
    execute1("create initialseg", CellRef)
    execute1("create node[5]", CellRef)
    execute1("create myelin[5]", CellRef)


    access hillock
    nseg = 3
    L=5
    diam = 10.0925
    v(0.0001) = i1
    nseg=3
    all.append
    axonal.append
    access initialseg
    nseg = 3
    L=30
    diam=1.5
    v(0.0001) = i2
    nseg=3
    all.append
    axonal.append

    index = i2
    for i=0,4{
        access node[i]
        index += 1
        L=1
        diam=1
        v(0.0001) = index
        all.append()
        axonal.append()
    }
    for i=0,4{
        access myelin[i]
        index += 1
        v(0.0001) = index
        L=1000
        diam=1.5
        v(0.0001) = i2
        all.append()
        myelinated.append()
    }

    nSecAxonal = 12
    connect hillock(0), soma(0)
    connect initialseg(0), hillock(1)

    connect myelin[0](0), initialseg(1)
    connect node[0](0), myelin[0](1)
    connect myelin[1](0), node[0](1)
    connect node[1](0), myelin[1](1)
    connect myelin[2](0), node[1](1)
    connect node[2](0), myelin[2](1)
    connect myelin[3](0), node[2](1)
    connect node[3](0), myelin[3](1)
    connect myelin[4](0), node[3](1)
    connect node[4](0), myelin[4](1)
    }
}
"""
