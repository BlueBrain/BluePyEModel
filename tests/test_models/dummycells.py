"""Dummy cell model used for testing"""

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

import bluepyopt.ephys as ephys


class DummyCellModel1(ephys.models.Model):
    """Dummy cell model 1"""

    def __init__(self, name=None):
        """Constructor"""

        super(DummyCellModel1, self).__init__(name)
        self.persistent = []
        self.icell = None

    def freeze(self, param_values):
        """Freeze model"""
        pass

    def unfreeze(self, param_names):
        """Freeze model"""
        pass

    def instantiate(self, sim=None):
        """Instantiate cell in simulator"""

        class Cell(object):
            """Empty cell class"""

            def __init__(self):
                """Constructor"""
                self.soma = None
                self.somatic = None

        self.icell = Cell()

        self.icell.soma = [sim.neuron.h.Section(name="soma", cell=self.icell)]
        self.icell.apic = [sim.neuron.h.Section(name="apic1", cell=self.icell)]

        self.icell.somatic = sim.neuron.h.SectionList()  # pylint: disable = W0201
        self.icell.somatic.append(sec=self.icell.soma[0])

        self.icell.apical = sim.neuron.h.SectionList()
        self.icell.apical.append(sec=self.icell.apic[0])

        self.persistent.append(self.icell)
        self.persistent.append(self.icell.soma[0])

        return self.icell

    def destroy(self, sim=None):
        """Destroy cell from simulator"""

        self.persistent = []
