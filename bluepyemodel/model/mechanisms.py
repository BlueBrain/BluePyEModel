"""Module with mechanism class."""

from bluepyopt.ephys.mechanisms import NrnMODMechanism


class NrnMODMechanismCustom(NrnMODMechanism):
    """MOD mechanism class that always instantiate all stochastic variables"""

    def __init__(
        self,
        name,
        mod_path=None,
        suffix=None,
        locations=None,
        preloaded=True,
        deterministic=True,
        prefix=None,
        comment="",
    ):
        """Constructor"""
        super().__init__(
            name, mod_path, suffix, locations, preloaded, deterministic, prefix, comment
        )

        # If deterministic is False, it might get changed to True.
        # Use this variable to change it back to False when needed.
        self.always_deterministic = deterministic

    def instantiate_determinism(self, deterministic, icell, isec, sim):
        """Instantiate enable/disable determinism"""

        if "Stoch" in self.suffix:
            setattr(isec, f"deterministic_{self.suffix}", 1 if deterministic else 0)

            # Set the seeds
            # Even when deterministic, that way neuron's psection does not crash
            # when encountering a stoch mech variable not set (e.g. rng)
            short_secname = sim.neuron.h.secname(sec=isec).split(".")[-1]
            for iseg in isec:
                seg_name = f"{short_secname}.{iseg.x:.19g}"
                getattr(sim.neuron.h, f"setdata_{self.suffix}")(iseg.x, sec=isec)
                seed_id1 = icell.gid
                seed_id2 = self.hash_py(seg_name)
                getattr(sim.neuron.h, f"setRNG_{self.suffix}")(seed_id1, seed_id2)
        else:
            if not deterministic:
                # can't do this for non-Stoch channels
                raise TypeError(
                    "Deterministic can only be set to False for "
                    f"Stoch channel, not {self.suffix}"
                )
