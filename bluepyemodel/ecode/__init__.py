"""eCode init script"""

from .apwaveform import APWaveform
from .comb import Comb
from .dehyperpol import DeHyperpol
from .firepattern import FirePattern
from .hyperdepol import HyperDepol
from .idrest import IDrest
from .iv import IV
from .negcheops import NegCheops
from .noiseou3 import NoiseOU3
from .poscheops import PosCheops
from .ramp import Ramp
from .sahp import sAHP
from .sinespec import SineSpec
from .spikerec import SpikeRecMultiSpikes
from .subwhitenoise import SubWhiteNoise
from .whitenoise import WhiteNoise

# The ecode names have to be lower case only, to avoid having to
# define duplicates.
eCodes = {
    "idrest": IDrest,
    "idthres": IDrest,
    "step": IDrest,
    "spontaps": IDrest,
    "sponnohold30": IDrest,
    "sponhold30": IDrest,
    "rinholdcurrent": IDrest,
    "bap": IDrest,
    "spikerec": IDrest,
    "spikerecmultispikes": SpikeRecMultiSpikes,
    "iv": IV,
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
}
