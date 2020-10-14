"""API using sql"""
import logging
import pickle
import pandas
import psycopg2
from psycopg2.extras import Json

from bluepyemodel.api.databaseAPI import DatabaseAPI

logger = logging.getLogger("__main__")

# pylint: disable=W0231


class PostgreSQL_API(DatabaseAPI):
    """API using sql"""

    def __init__(self):
        """"""

        self.connection = psycopg2.connect(
            user="emodel_pipeline",
            password="Cells2020!",
            host="bbpdbsrv06.bbp.epfl.ch",
            port="5432",
            database="emodel_pipeline",
        )

    def execute(self, query):
        """Execute a query."""

        cursor = self.connection.cursor()
        logger.debug("PostgreSQL query: %s", query)

        try:
            cursor.execute(query)
            self.connection.commit()

        except Exception as e:
            raise Exception(f"PostgreSQL error: {e}") from e

        finally:
            cursor.close()

    def execute_fill(self, query):
        """Execute a fill query."""

        cursor = self.connection.cursor()
        logger.debug("PostgreSQL query: %s", query)

        try:
            cursor.execute(query)
            self.connection.commit()

        except Exception as e:
            cursor.execute("ROLLBACK")
            self.connection.commit()
            raise Exception(f"PostgreSQL error: {e}") from e

        finally:
            cursor.close()

    def execute_fetch(self, query):
        """Execute a fetch query."""

        cursor = self.connection.cursor()
        logger.debug("PostgreSQL query: %s", query)

        try:
            cursor.execute(query)
            fetched = cursor.fetchall()

        except Exception as e:
            raise Exception("PostgreSQL error: {}".format(e)) from e

        finally:
            cursor.close()

        return fetched

    def fill(self, table, entries, replace, replace_keys):
        """
        Fill a table with entries. If replace is True, will replace the entry
        if it is already in the table.

        Args:
            table (str): name of the table in which the entry should be added
            entries (list): list of dictionaries
            replace (bool): if the entry exists, should it be replaced ?
            replace_keys (list): list of keys to check if entry is already
                in the table
        """

        cursor = self.connection.cursor()

        for entry in entries:

            for k in entry:
                if isinstance(entry[k], dict):
                    entry[k] = Json(entry[k])

            replace_keys_values = tuple([entry[rk] for rk in replace_keys])

            # Check of entry is in table
            query_exist = "SELECT 1 FROM {} WHERE ".format(table)
            for i, rk in enumerate(replace_keys):
                query_exist += "{} = %s".format(rk)
                if i < (len(replace_keys) - 1):
                    query_exist += " AND "
            query_exist = cursor.mogrify(query_exist, replace_keys_values)
            entry_exist = self.execute_fetch(query_exist)

            # Prepare the query to delete the entry if needed
            query_remove = "DELETE FROM {} WHERE ".format(table)
            for i, rk in enumerate(replace_keys):
                query_remove += "{} = %s".format(rk)
                if i < (len(replace_keys) - 1):
                    query_remove += " AND "
            query_remove = cursor.mogrify(query_remove, replace_keys_values)

            # Prepare the query to insert the entry if needed
            query_fill = "INSERT INTO {} (".format(table)
            for i, k in enumerate(entry.keys()):
                query_fill += "{}".format(k)
                if i < (len(entry.keys()) - 1):
                    query_fill += ","
            query_fill += ") VALUES ("
            for i, k in enumerate(entry.keys()):
                query_fill += "%s"
                if i < (len(entry.keys()) - 1):
                    query_fill += ","
            query_fill += ");"
            fill_values = tuple(entry.values())
            query_fill = cursor.mogrify(query_fill, fill_values)

            if replace and entry_exist:
                self.execute(query_remove)
                self.execute_fill(query_fill)
            elif not (entry_exist):
                self.execute_fill(query_fill)

        cursor.close()

    def remove(self, table, conditions):
        """
        Remove entries from a table based on conditions.

        Args:
            table (str): name of the table
            conditions (dict): keys and values used for the WHERE.
        """
        cursor = self.connection.cursor()

        values = tuple(conditions.values())
        query_remove = "DELETE FROM {} WHERE ".format(table)
        for i, rk in enumerate(conditions):
            query_remove += "{} = %s".format(rk)
            if i < (len(conditions) - 1):
                query_remove += " AND "
        query_remove = cursor.mogrify(query_remove, values)

        self.execute(query_remove)

    def fetch(self, table, conditions):
        """
        Retrieve entries from a talbe based on conditions.

        Args:
            table (str): name of the table
            conditions (dict): keys and values used for the WHERE.

        Returns:
            DataFrame

        """
        cursor = self.connection.cursor()

        col_query = (
            "SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_NAME = '{}';".format(table)
        )
        column_names = self.execute_fetch(col_query)
        column_names = [c[0] for c in column_names]

        query = "SELECT * FROM {} WHERE ".format(table)
        for i, cond in enumerate(conditions):
            if isinstance(conditions[cond], tuple):
                query += " {} IN %s".format(cond)
            else:
                query += " {} = %s".format(cond)
            if i < len(conditions) - 1:
                query += " AND "
        values = tuple(conditions.values())
        query = cursor.mogrify(query, values)

        data = self.execute_fetch(query)

        return pandas.DataFrame(data, columns=column_names)

    def close(self):
        self.connection.close()

    def get_extraction_metadata(self, emodel, species):
        """Gather the metadata used to build the config dictionary used as an
        input by BluePyEfe.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)

        Returns:
            cells (dict): return the cells recordings metadata
            protocols (dict): return the protocols metadata

        """

        targets_metadata = self.fetch("extraction_targets", {"emodel": emodel})
        targets_metadata = targets_metadata.to_dict("records")

        protocols = {}
        protocols_threshold = []
        for t in targets_metadata:
            protocols[t["ecode"]] = {
                "tolerances": t["tolerance"],
                "targets": t["targets"],
                "efeatures": t["efeatures"],
                "location": t["location"],
            }
            if t["threshold"]:
                protocols_threshold.append(t["ecode"])

        path_metadata = self.fetch(
            table="extraction_files",
            conditions={
                "emodel": emodel,
                "species": species,
                "ecode": tuple(protocols.keys()),
            },
        )

        if not (path_metadata.empty):

            path_metadata = path_metadata.to_dict("records")
            cells = {}
            for p in path_metadata:

                if p["cell_id"] not in cells:
                    cells[p["cell_id"]] = {}

                if p["ecode"] not in cells[p["cell_id"]]:
                    cells[p["cell_id"]][p["ecode"]] = []

                trace_metadata = {
                    "filepath": p["path"],
                    "ljp": p["liquid_junction_potential"],
                }

                for opt_key in ["ton", "toff", "i_unit", "v_unit", "t_unit"]:
                    if opt_key in p and p[opt_key] and not (pandas.isnull(p[opt_key])):
                        trace_metadata[opt_key] = p[opt_key]

                cells[p["cell_id"]][p["ecode"]].append(trace_metadata)

            return cells, protocols, protocols_threshold

        logger.warning(
            "PostgreSQL warning: could not get the extraction metadata for emodel %s",
            emodel,
        )
        return None, None, None

    def store_efeatures(self, emodel, species, efeatures, current):
        """ Save the efeatures and currents obtained from BluePyEfe"""
        entries = []
        for protocol in efeatures:
            for feature in efeatures[protocol]["soma"]:
                entries.append(
                    {
                        "emodel": emodel,
                        "species": species,
                        "name": feature["feature"],
                        "protocol": protocol,
                        "mean": feature["val"][0],
                        "std": feature["val"][1],
                    }
                )

        # Add holding and threshold current
        for cur in current:
            entries.append(
                {
                    "emodel": emodel,
                    "species": species,
                    "name": cur,
                    "protocol": "global",
                    "mean": current[cur][0],
                    "std": current[cur][1],
                }
            )

        replace_keys = ["emodel", "species", "name", "protocol"]
        self.fill(
            table="extraction_efeatures",
            entries=entries,
            replace=True,
            replace_keys=replace_keys,
        )

    def store_protocols(self, emodel, species, stimuli):
        """ Save the protocols obtained from BluePyEfe"""
        entries = []
        for stim_name, stim in stimuli.items():
            entries.append(
                {
                    "emodel": emodel,
                    "species": species,
                    "name": stim_name,
                    "type": "StepProtocol",
                    "stimulus_amp": stim["step"]["amp"],
                    "stimulus_thresh_perc": stim["step"]["thresh_perc"],
                    "stimulus_delay": stim["step"]["delay"],
                    "stimulus_duration": stim["step"]["duration"],
                    "stimulus_totduration": stim["step"]["totduration"],
                    "holding_amp": stim["holding"]["amp"],
                    "holding_delay": stim["holding"]["delay"],
                    "holding_duration": stim["holding"]["duration"],
                    "holding_totduration": stim["holding"]["totduration"],
                }
            )

        replace_keys = ["emodel", "species", "name"]
        self.fill(
            table="extraction_protocols",
            entries=entries,
            replace=True,
            replace_keys=replace_keys,
        )

    def store_model(
        self,
        emodel,
        species,
        checkpoint_path,
        param_names,
        feature_names,
        optimizer_name,
        opt_params,
        validated=False,
    ):
        """ Save a model obtained from BluePyOpt"""

        # Try to read the checkpoint
        try:
            run = pickle.load(open(checkpoint_path, "rb"))
        except BaseException:  # pylint: disable=W0703

            # If we can't read the checkpoint, tries to read the temporary one
            try:
                run = pickle.load(open(checkpoint_path + ".tmp", "rb"))
            except BaseException:  # pylint: disable=W0703
                logger.info(
                    "Couldn't open file %s to store the model in the SQL database",
                    checkpoint_path,
                )

        model = run["halloffame"][0]

        entry = {
            "emodel": emodel,
            "species": species,
            "fitness": sum(model.fitness.wvalues),
            "parameters": dict(zip(param_names, model)),
            "scores": dict(zip(feature_names, model.fitness.wvalues)),
            "validated": validated,
            "optimizer": str(optimizer_name),
        }

        if "seed" in opt_params:
            entry["seed"] = int(opt_params["seed"])

        replace_keys = ["emodel", "species", "fitness", "optimizer", "seed"]
        self.fill(
            table="models", entries=[entry], replace=True, replace_keys=replace_keys
        )

    def get_models(
        self,
        emodel,
        species,
    ):
        """ Get the models obtained from BluePyopt"""

        models = self.fetch("models", {"emodel": emodel, "species": species})
        if models.empty:
            logger.warning(
                "PostgreSQL warning: could not get the models for emodel %s", emodel
            )
            return None

        return models.to_dict(orient="records")

    def get_parameters(self, emodel, species):
        """Get the definition of the parameters to optimize as well as the
         locations of the mechanisms. Also returns the name to the mechanisms.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)

        Returns:
            params_definition (dict):
            mech_definition (dict):
            mech_names (list):

        """

        params_definition = {"distributions": {}, "parameters": {}}

        # Get parameters
        params = self.fetch(
            "optimisation_parameters", {"emodel": emodel, "species": species}
        )
        if params.empty:
            logger.warning(
                "PostgreSQL warning: could not get the parameters for emodel %s", emodel
            )
            return None, None, None

        # Put the definition in the correct format (as described on top of the
        # evaluator.py file)
        dists = []
        mech_names = []
        mech_definition = {}
        for param in params.to_dict(orient="records"):

            param_def = {
                "name": param["name"],
            }

            if len(param["value"]) == 1:
                param_def["val"] = param["value"][0]
            else:
                param_def["val"] = param["value"]

            if param["distribution"]:
                param_def["dist"] = param["distribution"]

            for loc in param["locations"]:
                if loc in params_definition["parameters"]:
                    params_definition["parameters"][loc].append(param_def)
                else:
                    params_definition["parameters"][loc] = [param_def]

                if param["mechanism"]:
                    if loc in mech_definition:
                        if param["mechanism"] not in mech_definition[loc]["mech"]:
                            mech_definition[loc]["mech"].append(param["mechanism"])
                    else:
                        mech_definition[loc] = {"mech": [param["mechanism"]]}

            dists.append(param["distribution"])
            mech_names.append(param["mechanism"])

        mech_names = set(mech_names)

        # Stochasticity
        mechanism_paths = self.fetch(
            table="mechanisms_path", conditions={"name": tuple(mech_names)}
        )
        if mechanism_paths.empty:
            logger.warning("PostgreSQL warning: could not get the mechanism paths")
            return None
        mechanism_paths = mechanism_paths.to_dict(orient="records")

        for loc in mech_definition:
            stoch = []
            for m in mech_definition[loc]["mech"]:
                is_stock = next(
                    item["stochastic"] for item in mechanism_paths if item["name"] == m
                )
                stoch.append(is_stock)
                mech_definition[loc]["stoch"] = stoch

        # Get the functions matching the distributions
        dists = set(dists)
        dist_def = self.fetch(
            table="optimisation_distributions", conditions={"name": tuple(dists)}
        )
        if dist_def.empty:
            logger.warning(
                "PostgreSQL warning: could not get the distributions for emodel %s",
                emodel,
            )
            return params, None, None

        for dd in dist_def.to_dict("records"):
            params_definition["distributions"][dd["name"]] = {
                "fun": dd["function"],
                "parameters": dd["parameters"],
            }

        return params_definition, mech_definition, mech_names

    def get_protocols(
        self,
        emodel,
        species,
        delay=0.0,
    ):
        """Get the protocols from the database and put in a format that fits
         the MainProtocol needs.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)
            delay (float): additional delay in ms to add at the start of
                the protocols.

        Returns:
            protocols_out (dict): protocols definitions

        """
        # TODO: handle extra recordings

        # Get the optimization targets
        targets = self.fetch(
            table="optimisation_targets",
            conditions={"emodel": emodel, "species": species},
        )
        if targets.empty:
            logger.warning(
                "PostgreSQL warning: could not get the protocols for emodel %s", emodel
            )
            return None

        # Get the matching protocols
        protocols_out = {}
        for target in targets.to_dict(orient="records"):

            protocols = self.fetch(
                table="extraction_protocols",
                conditions={
                    "emodel": emodel,
                    "species": species,
                    "name": f"{target['ecode']}_{target['target']}",
                },
            )
            if protocols.empty:
                logger.warning(
                    "PostgreSQL warning: could not get the protocols for emodel %s",
                    emodel,
                )
                return None

            for prot in protocols.to_dict(orient="records"):
                print(prot)

                if target["type"] == "RinHoldCurrent":

                    protocols_out["RinHoldCurrent"] = {
                        "type": "RinHoldCurrent",
                        "stimuli": {
                            "step": {
                                "delay": delay + prot["stimulus_delay"],
                                "amp": prot["stimulus_amp"],
                                "thresh_perc": prot["stimulus_thresh_perc"],
                                "duration": prot["stimulus_duration"],
                                "totduration": delay + prot["stimulus_totduration"],
                            },
                            "holding": {
                                "delay": 0,
                                "amp": None,
                                "duration": prot["holding_duration"],
                                "totduration": delay + prot["holding_totduration"],
                            },
                        },
                    }

                elif target["type"] == "RMP":
                    # The name_rmp_protocol is used only for the efeatures, the
                    # protocol itself is fixed:
                    protocols_out["RMP"] = {
                        "type": "RMP",
                        "stimuli": {
                            "step": {
                                "delay": 250,
                                "amp": 0,
                                "duration": 400,
                                "totduration": 650,
                            },
                            "holding": {
                                "delay": 0,
                                "amp": 0,
                                "duration": 650,
                                "totduration": 650,
                            },
                        },
                    }

                elif target["type"] in ("StepThresholdProtocol", "StepProtocol"):

                    protocols_out[prot["name"]] = {
                        "type": "StepThresholdProtocol",
                        "stimuli": {
                            "step": {
                                "delay": delay + prot["stimulus_delay"],
                                "amp": None,
                                "thresh_perc": prot["stimulus_thresh_perc"],
                                "duration": prot["stimulus_duration"],
                                "totduration": delay + prot["stimulus_totduration"],
                            }
                        },
                    }
        return protocols_out

    def get_features(
        self,
        emodel,
        species,
    ):
        """Get the efeatures from the database and put in a format that fits
         the MainProtocol needs.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)

        Returns:
            efeatures_out (dict): efeatures definitions

        """
        # Get the optimization targets
        targets = self.fetch(
            table="optimisation_targets",
            conditions={"emodel": emodel, "species": species},
        )
        if targets.empty:
            logger.warning(
                "PostgreSQL warning: could not get the protocols for emodel %s", emodel
            )
            return None

        efeatures_out = {
            "RMP": {"soma.v": []},
            "RinHoldCurrent": {"soma.v": []},
            "Threshold": {"soma.v": []},
        }

        # Get the values for the efeatures matching the targets
        for target in targets.to_dict(orient="records"):

            for feat in target["efeatures"]:

                efeatures = self.fetch(
                    table="extraction_efeatures",
                    conditions={
                        "emodel": emodel,
                        "species": species,
                        "protocol": f"{target['ecode']}_{target['target']}",
                        "name": feat,
                    },
                )
                if efeatures.empty:
                    logger.warning(
                        "PostgreSQL warning: could not get the efeatures %s for emodel %s",
                        feat,
                        emodel,
                    )
                    continue

                for efeat in efeatures.to_dict(orient="records"):
                    if target["type"] == "RMP":
                        if efeat["name"] == "steady_state_voltage_stimend":
                            efeatures_out["RMP"]["soma.v"].append(
                                {
                                    "feature": "voltage_base",
                                    "val": [efeat["mean"], efeat["std"]],
                                    "strict_stim": True,
                                }
                            )

                    elif target["type"] == "RinHoldCurrent":
                        if efeat["name"] in (
                            "voltage_base",
                            "ohmic_input_resistance_vb_ssse",
                        ):
                            efeatures_out["RinHoldCurrent"]["soma.v"].append(
                                {
                                    "feature": efeat["name"],
                                    "val": [efeat["mean"], efeat["std"]],
                                    "strict_stim": True,
                                }
                            )

                    else:
                        if efeat["protocol"] not in efeatures_out:
                            efeatures_out[efeat["protocol"]] = {"soma.v": []}

                        efeatures_out[efeat["protocol"]]["soma.v"].append(
                            {
                                "feature": efeat["name"],
                                "val": [efeat["mean"], efeat["std"]],
                                "strict_stim": True,
                            }
                        )

        # Get the hypamp and thresh currents
        efeatures = self.fetch(
            table="extraction_efeatures",
            conditions={"emodel": emodel, "species": species, "protocol": "global"},
        )
        if efeatures.empty:
            logger.warning(
                "PostgreSQL warning: could not get the efeatures for emodel %s", emodel
            )
            return None

        for efeat in efeatures.to_dict(orient="records"):
            if efeat["name"] == "hypamp":
                efeatures_out["RinHoldCurrent"]["soma.v"].append(
                    {
                        "feature": "bpo_holding_current",
                        "val": [efeat["mean"], efeat["std"]],
                    }
                )

            elif efeat["name"] == "thresh":
                efeatures_out["Threshold"]["soma.v"].append(
                    {
                        "feature": "bpo_threshold_current",
                        "val": [efeat["mean"], efeat["std"]],
                    }
                )

        return efeatures_out

    def get_morphologies(self, emodel, species):
        """Get the name and path to the morphologies.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)

        Returns:
            morphology_definition (list): [{'name': morph_name,
                                            'path': 'morph_path'}

        """
        morphologies = self.fetch(
            table="optimisation_morphology",
            conditions={"emodel": emodel, "species": species},
        )
        if morphologies.empty:
            logger.warning(
                "PostgreSQL warning: could not get the morphologies for emodel %s",
                emodel,
            )
            return None

        morphology_definition = self.fetch(
            table="morphologies",
            conditions={
                "name": tuple(
                    [m["name"] for m in morphologies.to_dict(orient="records")]
                )
            },
        )
        morphology_definition = morphology_definition.to_dict(orient="records")

        return morphology_definition

    def get_mechanism_paths(self, mechanism_names):
        """Get the path of the mod files

        Args:
            mechanism_names (list): names of the mechanisms

        Returns:
            mechanism_paths (dict): {'mech_name': 'mech_path'}

        """

        mechanism_paths = self.fetch(
            table="mechanisms_path", conditions={"name": tuple(mechanism_names)}
        )
        if mechanism_paths.empty:
            logger.warning("PostgreSQL warning: could not get the mechanism paths")
            return None

        return mechanism_paths.to_dict(orient="records")
