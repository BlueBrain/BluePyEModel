"""eCode init script"""

from .apwaveform import APWaveform
from .dehyperpol import DeHyperpol
from .firepattern import FirePattern
from .hyperdepol import HyperDepol
from .idrest import IDrest
from .iv import IV
from .poscheops import PosCheops
from .ramp import Ramp
from .sahp import sAHP
from .sinespec import SineSpec
from .subwhitenoise import SubWhiteNoise

# The ecode names have to be lower case only, to avoid having to
# define duplicates.
eCodes = {
    "idrest": IDrest,
    "step": IDrest,
    "spontaps": IDrest,
    "sponnohold30": IDrest,
    "sponhold30": IDrest,
    "rinholdcurrent": IDrest,
    "bap": IDrest,
    "spikerec": IDrest,
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
    "ramp": Ramp,
    "ap_thresh": Ramp,
    "apthresh": Ramp,
    "sinespec": SineSpec,
    "subwhitenoise": SubWhiteNoise,
}
