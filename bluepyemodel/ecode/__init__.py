"""eCode init script"""

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

from .apwaveform import APWaveform
from .comb import Comb
from .customfromfile import CustomFromFile
from .dehyperpol import DeHyperpol
from .dendrite import BAC
from .dendrite import DendriticStep
from .dendrite import Synaptic
from .firepattern import FirePattern
from .hyperdepol import HyperDepol
from .idrest import IDrest
from .iv import IV
from .negcheops import NegCheops
from .noiseou3 import NoiseOU3
from .poscheops import PosCheops
from .probampanmda_ems import ProbAMPANMDA_EMS
from .ramp import Ramp
from .random_square_inputs import MultipleRandomStepInputs
from .sahp import sAHP
from .sinespec import SineSpec
from .spikerec import SpikeRecMultiSpikes
from .subwhitenoise import SubWhiteNoise
from .thresholdaddition import ThresholdAddition
from .whitenoise import WhiteNoise

# The ecode names have to be lower case only, to avoid having to
# define duplicates.
eCodes = {
    "spontaneous": IDrest,
    "idrest": IDrest,
    "idthres": IDrest,
    "idthresh": IDrest,
    "step": IDrest,
    "spontaps": IDrest,
    "sponnohold30": IDrest,
    "sponhold30": IDrest,
    "rinholdcurrent": IDrest,
    "bap": IDrest,
    "spikerecmultispikes": SpikeRecMultiSpikes,
    "iv": IV,
    "spikerec": IDrest,
    "apwaveform": APWaveform,
    "firepattern": FirePattern,
    "sahp": sAHP,
    "idhyperpol": sAHP,
    "irdepol": sAHP,
    "irhyperpol": sAHP,
    "iddepol": sAHP,
    "hyperdepol": HyperDepol,
    "dehyperpol": DeHyperpol,
    "poscheops": PosCheops,
    "negcheops": NegCheops,
    "ramp": Ramp,
    "ap_thresh": Ramp,
    "apthresh": Ramp,
    "sinespec": SineSpec,
    "subwhitenoise": SubWhiteNoise,
    "noiseou3": NoiseOU3,
    "whitenoise": WhiteNoise,
    "highfreq": Comb,
    "synaptic": Synaptic,
    "bac": BAC,
    "dendritic": DendriticStep,
    "randomsteps": MultipleRandomStepInputs,
    "custom": CustomFromFile,
    "startnohold": IDrest,
    "thresholdaddition": ThresholdAddition,
    "probampanmda_ems": ProbAMPANMDA_EMS,
}

fixed_timestep_eCodes = ["probampanmda_ems"]
