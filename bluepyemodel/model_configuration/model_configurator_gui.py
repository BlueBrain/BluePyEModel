"""Model Configurator GUI"""
import logging
from tkinter import *
from tkinter import ttk

logger = logging.getLogger(__name__)

class NewConfigPopup(object):

    def __init__(self, root):

        self.config_name = ""
        self.use_gene_data = False

        self.root = Toplevel(root)

        self.label = Label(self.root, text="Enter configuration name:")
        self.label.pack()

        self.name_entry = Entry(self.root)
        self.name_entry.pack()

        self.use_gene_data_b = BooleanVar()
        self.use_gene_data_b.set(False)
        self.use_gene_checkbutton = ttk.Checkbutton(
            self.root, text='Use gene data', var=self.use_gene_data_b
        )
        self.use_gene_checkbutton.pack()

        self.ok_button = Button(self.root, text='Ok', command=self.cleanup)
        self.ok_button.pack()

    def cleanup(self):
        self.config_name = self.name_entry.get()
        self.use_gene_data = self.use_gene_data_b.get()
        self.root.destroy()


class ParameterWidget():

    def __init__(self, root, parameter_config):

        self.root = root
        self.parameter_config = parameter_config

        self.frame = ttk.Frame(self.root)
        self.frame.pack(side=TOP)

        self.name_label = Label(self.root, text="Name:")
        self.name_label.pack(side=LEFT)
        self.name_entry = Entry(self.frame)
        self.name_entry(END, parameter_config.name)
        self.name_entry.pack(side=RIGHT)


class DistributionWidget():
    pass


class MechanismsWidget():
    pass


class ModelConfiguratorGUI:
    """GUI for the model configurator"""

    def __init__(self, configurator):

        self.configurator = configurator
        self.popup = None

        self.root = Tk()
        self.root.title("Neuron Model Configurator")

        self.setup_menu()
        self.setup_main_tabs()
        self.setup_parameters()
        self.setup_mechanisms()
        self.setup_distributions()

        self.root.mainloop()

    def setup_menu(self):

        self.menu_bar = ttk.Frame(self.root)
        self.menu_bar.pack(side=TOP)

        self.button_new_config = ttk.Button(
            self.menu_bar,
            text="New config",
            command=self.new_configuration
        )
        self.button_new_config.pack(side=LEFT)

    def setup_main_tabs(self):

        self.content_area = ttk.Notebook(self.root)
        self.content_area.pack(side=TOP)
        self.panel_parameters = ttk.Frame(self.content_area)
        self.panel_mechanisms = ttk.Frame(self.content_area)
        self.panel_distributions = ttk.Frame(self.content_area)
        self.content_area.add(self.panel_parameters, text='Parameters')
        self.content_area.add(self.panel_mechanisms, text='Mechanisms')
        self.content_area.add(self.panel_distributions, text='Distributions')

    def setup_parameters(self):

        if not self.configurator.configuration:
            pass

    def setup_mechanisms(self):

        if not self.configurator.configuration:
            pass

    def setup_distributions(self):

        if not self.configurator.configuration:
            pass

    def new_configuration(self):

        self.popup = NewConfigPopup(self.root)
        self.button_new_config["state"] = "disabled"
        self.root.wait_window(self.popup.root)
        self.button_new_config["state"] = "normal"

        if self.configurator:
            self.configurator.new_configuration(self.popup.config_name, self.popup.use_gene_data)
        else:
            pass


if __name__ == "__main__":

    from bluepyemodel.model_configuration.model_configurator import ModelConfigurator
    from bluepyemodel.access_point.nexus import NexusAccessPoint

    access_point = NexusAccessPoint(
        emodel="L5_TPC:B_cAC",
        species="mouse",
        brain_region="SSCX",
        project="ncmv3",
        organisation="bbp",
        endpoint="https://bbp.epfl.ch/nexus/v1",
        forge_path="./forge.yml",
        ttype="L4/5 IT_1",
        iteration_tag="test"
    )

    configurator = ModelConfigurator(access_point)
    gui = ModelConfiguratorGUI(configurator)
