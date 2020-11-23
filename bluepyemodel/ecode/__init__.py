"""eCode init script"""

from .step import Step
from .poscheops import PosCheops
from .ramp import Ramp
from .sAHP import sAHP

# The ecode names have to be lower case only to avoid having to
# define duplicates.
eCodes = {
    "idrest": Step,
    "idthresh": Step,
    "idthres": Step,
    "apwaveform": Step,
    "iv": Step,
    "step": Step,
    "spontaps": Step,
    "firepattern": Step,
    "sponnohold30": Step,
    "sponhold30": Step,
    "rinholdcurrent": Step,
    "rmp": Step,
    "bap": Step,
    "spikerec": Step,
    "ramp": Ramp,
    "ap_thresh": Ramp,
    "apthresh": Ramp,
    "poscheops": PosCheops,
    "sahp": sAHP,
    "idhyperpol": sAHP,
    "irdepol": sAHP,
    "irhyperpol": sAHP,
    "iddepol": sAHP,
}
