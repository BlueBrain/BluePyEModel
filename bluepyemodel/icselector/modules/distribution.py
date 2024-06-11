"""Mechanism class corresponding to mechanisms fields in the icmapping file."""

from dataclasses import asdict
from dataclasses import dataclass


@dataclass
class Distribution:
    """Holds subcellular distribution for each compartment."""

    all: str = ""  # If 'all' is set, other fields will be ignored!
    somatic: str = "uniform"
    basal: str = "uniform"
    apical: str = "uniform"
    axonal: str = "uniform"
    myelinated: str = ""

    def set_all(self, dist):
        """Set distribution on all compartments, but not on the 'all' field.

        Args:
            dist (str): key of the subcellular distribution
        """

        fields = asdict(self)
        fields.pop("all")
        for field in fields:
            setattr(self, field, dist)

    def set_fields(self, **fields):
        """Set distributions on compartments.

        Args:
            fields (dict): fields to be set specified as
                {compartment: distribution}
        """

        # If 'fields' contains 'all', then 'all' is set and other compartments
        # are emptied
        if "all" in fields:
            if not fields["all"] == "":
                self.set_all("")
                self.all = fields["all"]
                return

        for k, v in fields.items():
            if isinstance(v, str):
                if v == "nan":
                    v = ""
                setattr(self, k, v)
            else:
                setattr(self, k, "")

    def get(self):
        """Get distributions as a dict.

        Returns:
            distr (dict): distributions for all comparments"""

        distr = {k: v for k, v in asdict(self).items() if not v == ""}
        return distr

    def __str__(self):
        out_str = []
        distr = self.get()
        for k, v in distr.items():
            out_str += [f"{k} = {v}"]
        return ", ".join(out_str)
