"""Model selector selects NEURON mechanisms for cell model configuration"""

"""
Copyright 2023-2024, EPFL/Blue Brain Project

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


import copy

from .mechanism import Mechanism


class ModelSelector:
    """Selects NEURON mechanisms for cell model configuration"""

    def __init__(self, ic_map):
        """
        Args:
            ic_map (dict): content of the icmapping.json file
        """

        self._ic_map = ic_map
        self._mechanisms = self._load_mechanisms()
        self.status = "stable"
        self.mode = "mixed"

    def _get_ic_map_entry(self, key):
        """Get an entry from the icmap

        Args:
            key (str): name of the field

        Returns:
            entry: content of the field
        """

        entry = self._ic_map[key]
        entry = {k: v for k, v in entry.items() if not k[0] == "_"}
        return entry

    def _load_mechanisms(self):
        """Load mechanism field from the icmap

        Returns:
            mechanisms (dict [Mechanism]):
        """
        ic_map_mechs = self._get_ic_map_entry("mechanisms")
        mechanisms = {k: Mechanism(**v) for k, v in ic_map_mechs.items()}
        return mechanisms

    def _set_mechanism_parameters(self):
        """Set mechanism parameters from icmap settings."""

        mechs = self._mechanisms
        params = self._get_ic_map_entry("mechanism_parameters")
        for name, info in params.items():
            mech = mechs[name]
            mech.set_from_icmap(info)

    def _get_mech_from_channel(self, channel_name):
        """Get mechanism corresponding to channel

        Args:
            channel_name (str): name of the ion channel

        Returns:
            mech (Mechanism): corresponding mechanism
        """

        chan_to_model = self._get_ic_map_entry("channels")
        models = chan_to_model.pop(channel_name, [])
        mechs = [self._get_mech_from_model(model) for model in models]

        for mech in mechs:
            if self._check_mech(mech):
                return mech
        return None

    def _check_mech(self, mech):
        if not mech:
            return False

        if hasattr(mech, "type"):
            return self.mode in (mech.type, "mixed")
        return True

    def _get_mech_from_model(self, model_name):
        """Get mechanism by model name

        Args:
            model_name (str): model name

        Returns:
            (Mechanism): corresponding mechanism
        """

        if model_name in self._mechanisms:
            mech = self._mechanisms[model_name]
            if self._check_mech(mech):
                return mech
        return None

    def get(self, name):
        """Get mechanism from channel or model name
        Args:
            name (str): channel or model name
        Returns:
            (Mechanism): corresponding mechanism
        """

        mech = self._get_mech_from_channel(name)
        if not mech:
            mech = self._get_mech_from_model(name)
        return mech

    def select(self, name):
        """Select and return mechanism by channel or model name
        Args:
            name (str): channel or model name
        Returns:
            (Mechanism): corresponding mechanism
        """

        mech = self.get(name)
        if not mech:
            return None
        mech.select(check_status=self.status)
        if mech.is_selected():
            for req_mod in mech.requires:
                req_mech = self.select(req_mod)  # NB: recursion!
                if req_mech.is_selected():
                    req_mech.set_distribution(mech.distribution)
        return mech

    def get_mechanisms(self, selected_only=True):
        """Get a copy of all available mechanisms from the icmapping file.
        Args:
            selected_only (bool): flag to get only selected channels
        Returns:
            mechs (dict [Mechanism]): mechanisms with all associated info
                fields
        """

        self.select("pas")
        self._set_mechanism_parameters()
        if selected_only:
            mechs = {name: mech for name, mech in self._mechanisms.items() if mech.is_selected()}
        else:
            mechs = self._mechanisms
        return copy.deepcopy(mechs)

    def __str__(self):
        out_str = []
        out_str += ["\n>>> Mechanisms <<<"]
        n = 1
        mechs = self.get_mechanisms()
        for v in mechs.values():
            items_str = f"-{n}- {str(v)}"
            n += 1
            out_str.append(items_str)
        return "\n".join(out_str)
