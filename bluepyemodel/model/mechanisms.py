"""Module with mechanism class."""

from bluepyopt.ephys.mechanisms import NrnMODMechanism


class NrnMODMechanismCustom(NrnMODMechanism):
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
        super(NrnMODMechanismCustom, self).__init__(
            name, mod_path, suffix, locations, preloaded, deterministic, prefix, comment
        )

        # If deterministic is False, it might get changed to True.
        # Use this variable to change it back to False when needed.
        self.always_deterministic = deterministic

    def instantiate_determinism(self, deterministic, icell, isec, sim):
        """Instantiate enable/disable determinism"""

        if "Stoch" in self.suffix:
            setattr(isec, "deterministic_%s" % (self.suffix), 1 if deterministic else 0)

            # Set the seeds
            # Even when deterministic, that way neuron's psection does not crash
            # when encountering a stoch mech variable not set (e.g. rng)
            short_secname = sim.neuron.h.secname(sec=isec).split(".")[-1]
            for iseg in isec:
                seg_name = "%s.%.19g" % (short_secname, iseg.x)
                getattr(sim.neuron.h, "setdata_%s" % self.suffix)(iseg.x, sec=isec)
                seed_id1 = icell.gid
                seed_id2 = self.hash_py(seg_name)
                getattr(sim.neuron.h, "setRNG_%s" % self.suffix)(seed_id1, seed_id2)
        else:
            if not deterministic:
                # can't do this for non-Stoch channels
                raise TypeError(
                    "Deterministic can only be set to False for "
                    "Stoch channel, not %s" % self.suffix
                )
