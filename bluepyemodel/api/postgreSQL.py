"""API using sql"""
import logging

import pandas
import psycopg2
from psycopg2.extras import Json

import bluepyemodel.api.postgreSQL_tables as sql_tables
from bluepyemodel.api.databaseAPI import DatabaseAPI
from bluepyemodel.feature_extraction.extract import get_config

logger = logging.getLogger("__main__")

# pylint: disable=W0231,W0703,R1702


class PostgreSQL_API(DatabaseAPI):
    """API using sql"""

    tables = [
        "extraction_targets",
        "extraction_files",
        "extraction_efeatures",
        "extraction_protocols",
        "optimisation_targets",
        "morphologies",
        "optimisation_morphology",
        "optimisation_parameters",
        "optimisation_distributions",
        "mechanisms_path",
        "models",
        "validation_targets",
    ]

    def __init__(self, project_name):
        """"""
        self.project_name = project_name
        self.connection = psycopg2.connect(
            user="emodel_pipeline",
            password="Cells2020!",
            host="bbpdbsrv06.bbp.epfl.ch",
            port="5432",
            database="emodel_pipeline",
        )

    def reset_project(self, answer=""):
        """Delete and re-create all the tables for a project."""
        while answer not in ["y", "n"]:
            answer = input(
                "You are about to reset the tables of proj {}"
                ". All data will be lost. Are you sure (y/n)"
                "".format(self.project_name)
            )

        if answer == "y":
            self.delete_project(answer="y")
            self.create_project()
        else:
            logger.info("Aborting reset.")

    def delete_project(self, answer=""):
        """Delete all the tables for a project"""
        while answer not in ["y", "n"]:
            answer = input(
                "You are about to delete all the tables of proj {}"
                ". All data will be lost. Are you sure (y/n)"
                "".format(self.project_name)
            )

        if answer == "y":
            for table in self.tables:
                query = "DROP TABLE {}_{};".format(self.project_name, table)
                self.execute_no_error(query)
        else:
            logger.info("Aborting deletion.")

    def create_project(self):
        """Create all tables neede for a project."""
        for def_table in [
            sql_tables.def_extraction_targets,
            sql_tables.def_extraction_files,
            sql_tables.def_extraction_efeatures,
            sql_tables.def_extraction_protocols,
            sql_tables.def_optimisation_targets,
            sql_tables.def_morphologies,
            sql_tables.def_optimisation_morphology,
            sql_tables.def_optimisation_parameters,
            sql_tables.def_optimisation_distributions,
            sql_tables.def_mechanisms_path,
            sql_tables.def_models,
            sql_tables.def_validation_targets,
        ]:
            self.execute_no_error(def_table.format(self.project_name))

    def execute_no_error(self, query):
        """Execute a query."""

        cursor = self.connection.cursor()
        logger.debug("PostgreSQL query: %s", query)

        try:
            cursor.execute(query)
            self.connection.commit()
            cursor.close()

        except Exception as e:
            logger.warning("PostgreSQL error: %s", e)
            self.connection.commit()
            cursor.close()

    def execute(self, query):
        """Execute a query."""

        cursor = self.connection.cursor()
        logger.debug("PostgreSQL query: %s", query)

        try:
            cursor.execute(query)
            self.connection.commit()
            cursor.close()

        except Exception as e:
            cursor.close()
            raise Exception(f"PostgreSQL error: {e}") from e

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
            query_exist = "SELECT 1 FROM {}_{} WHERE ".format(self.project_name, table)
            for i, rk in enumerate(replace_keys):
                query_exist += "{} = %s".format(rk)
                if i < (len(replace_keys) - 1):
                    query_exist += " AND "
            query_exist = cursor.mogrify(query_exist, replace_keys_values)
            entry_exist = self.execute_fetch(query_exist)

            if len(entry_exist):  # pylint: disable=len-as-condition
                entry_exist = bool(entry_exist[0])
            else:
                entry_exist = False

            # Prepare the query to delete the entry if needed
            query_remove = "DELETE FROM {}_{} WHERE ".format(self.project_name, table)
            for i, rk in enumerate(replace_keys):
                query_remove += "{} = %s".format(rk)
                if i < (len(replace_keys) - 1):
                    query_remove += " AND "
            query_remove = cursor.mogrify(query_remove, replace_keys_values)

            # Prepare the query to insert the entry if needed
            query_fill = "INSERT INTO {}_{} (".format(self.project_name, table)
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
        query_remove = "DELETE FROM {}_{} WHERE ".format(self.project_name, table)
        for i, rk in enumerate(conditions):
            query_remove += "{} = %s".format(rk)
            if i < (len(conditions) - 1):
                query_remove += " AND "
        query_remove = cursor.mogrify(query_remove, values)

        self.execute(query_remove)

    def fetch(self, table, conditions):
        """
        Retrieve entries from a table based on conditions.

        Args:
            table (str): name of the table
            conditions (dict): keys and values used for the WHERE.

        Returns:
            DataFrame

        """
        cursor = self.connection.cursor()

        col_query = (
            "SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_NAME = '{}_{}';".format(self.project_name, table)
        )
        column_names = self.execute_fetch(col_query)
        column_names = [c[0] for c in column_names]

        query = "SELECT * FROM {}_{} WHERE ".format(self.project_name, table)
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

                for opt_key in [
                    "ton",
                    "toff",
                    "i_unit",
                    "v_unit",
                    "t_unit",
                    "tmid",
                    "tmid2",
                    "tend",
                    "t_unit",
                ]:
                    if opt_key in p and p[opt_key] and not (pandas.isnull(p[opt_key])):
                        trace_metadata[opt_key] = p[opt_key]

                cells[p["cell_id"]][p["ecode"]].append(trace_metadata)

            return cells, protocols, protocols_threshold

        logger.warning(
            "PostgreSQL warning: could not get the extraction metadata for emodel %s",
            emodel,
        )
        return None, None, None

    def get_feature_extraction_config(self, emodel, species, threshold_nvalue_save=1):
        """Get config for feature extraction.

        Args:
            emodel (str): name of the emodels
            species (str): name of the species (rat, human, mouse)
            threshold_nvalue_save (int): lower bounds of the number of values required
                to save an efeature.

        Returns:
            extract_config (dict): config dict used in extract.extract_efeatures()
        """
        cells, protocols, protocols_threshold = self.get_extraction_metadata(
            emodel=emodel, species=species
        )

        return get_config(
            cells,
            protocols,
            protocols_threshold=protocols_threshold,
            threshold_nvalue_save=threshold_nvalue_save,
        )

    def store_efeatures(
        self,
        emodel,
        species,
        efeatures,
        current,
        name_Rin_protocol,
        name_rmp_protocol,
        validation_protocols,
    ):
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
        for cur in ["hypamp", "thresh"]:
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

    def store_protocols(self, emodel, species, stimuli, validation_protocols):
        """ Save the protocols obtained from BluePyEfe"""
        entries = []
        for stim_name, stim in stimuli.items():
            entries.append(
                {
                    "emodel": emodel,
                    "species": species,
                    "name": stim_name,
                    "definition": {"step": stim["step"], "holding": stim["holding"]},
                }
            )

        replace_keys = ["emodel", "species", "name"]
        self.fill(
            table="extraction_protocols",
            entries=entries,
            replace=True,
            replace_keys=replace_keys,
        )

    def store_emodel(
        self,
        emodel,
        scores,
        params,
        optimizer_name,
        seed,
        githash="",
        validated=False,
        scores_validation=None,
        species=None,
    ):
        """ Save an emodel obtained from BluePyOpt"""

        if scores_validation is None:
            scores_validation = {}

        entry = {
            "emodel": emodel,
            "species": species,
            "fitness": sum(list(scores.values())),
            "parameters": params,
            "scores": scores,
            "scores_validation": scores_validation,
            "validated": validated,
            "optimizer": str(optimizer_name),
            "seed": seed,
            "githash": githash,
        }

        if seed is not None:
            entry["seed"] = int(seed)

        replace_keys = ["emodel", "species", "optimizer", "seed"]
        self.fill(table="models", entries=[entry], replace=True, replace_keys=replace_keys)

    def get_emodels(self, emodels, species):
        """Get the list of emodels dictionaries.

        Args:
            emodels (list): list of names of the emodels
            species (str): name of the species (rat, human, mouse)
        """

        emodels_data = self.fetch("models", {"emodel": tuple(emodels), "species": species})
        if emodels_data.empty:
            logger.warning(
                "PostgreSQL warning: could not get the models for emodel %s",
                emodels,
            )
            return None

        return emodels_data.to_dict(orient="records")

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
        params = self.fetch("optimisation_parameters", {"emodel": emodel, "species": species})
        if params.empty:
            logger.warning("PostgreSQL warning: could not get the parameters for emodel %s", emodel)
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
            return None, None, None
        mechanism_paths = mechanism_paths.to_dict(orient="records")

        for loc in mech_definition:
            stoch = []
            for m in mech_definition[loc]["mech"]:
                is_stock = next(
                    (item["stochastic"] for item in mechanism_paths if item["name"] == m),
                    False,
                )
                stoch.append(is_stock)
                mech_definition[loc]["stoch"] = stoch

        # Get the functions matching the distributions
        dists = set(dists)
        dist_def = self.fetch(table="optimisation_distributions", conditions={"name": tuple(dists)})
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

            if not pandas.isnull(dd["soma_ref_location"]):
                params_definition["distributions"][dd["name"]]["soma_ref_location"] = dd[
                    "soma_ref_location"
                ]

        return params_definition, mech_definition, mech_names

    def get_protocols(self, emodel, species, delay=0.0, include_validation=False):
        """Get the protocols from the database and put in a format that fits
         the MainProtocol needs.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)
            delay (float): additional delay in ms to add at the start of
                the protocols.
            include_validation (bool): if True, returns the protocols for validation as well

        Returns:
            protocols_out (dict): protocols definitions

        """
        # TODO: handle extra recordings
        protocols_out = {}

        # Get the optimization targets
        targets = self.fetch(
            table="optimisation_targets",
            conditions={"emodel": emodel, "species": species},
        )
        if targets.empty:
            logger.warning("PostgreSQL warning: could not get the targets for emodel %s", emodel)
            return None
        targets = targets.to_dict(orient="records")

        # Get the validation targets
        if include_validation:
            validation_targets = self.fetch(
                table="validation_targets",
                conditions={"emodel": emodel, "species": species},
            )
            if validation_targets.empty:
                logger.warning(
                    "PostgreSQL warning: could not get the validation targets" " for emodel %s",
                    emodel,
                )
                return None
            validation_targets = validation_targets.to_dict(orient="records")
        else:
            validation_targets = []

        # Get the matching protocols
        for target in targets + validation_targets:

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

                if target["type"] in ("StepThresholdProtocol", "StepProtocol"):
                    stim_def = prot["definition"]["step"]
                    stim_def["holding_current"] = prot["definition"]["holding"]["amp"]
                    protocols_out[prot["name"]] = {
                        "type": target["type"],
                        "stimuli": stim_def,
                    }

                if not pandas.isnull(target["extra_recordings"]):
                    protocols_out[prot["name"]]["extra_recordings"] = target["extra_recordings"]

                    for i, extra in enumerate(protocols_out[prot["name"]]["extra_recordings"]):
                        if extra["type"] == "somadistanceapic":
                            morphologies = self.get_morphologies(emodel, species)

                            if not pandas.isnull(morphologies[0]["sec_index"]):
                                sec_index = morphologies[0]["sec_index"]
                                protocols_out[prot["name"]]["extra_recordings"][i][
                                    "sec_index"
                                ] = sec_index
                            else:
                                raise Exception(
                                    "Missing sec_index for morphology %s" % morphologies[0]["name"]
                                )

        return protocols_out

    def get_name_validation_protocols(self, emodel, species):
        """Get the names of the protocols used for validation """

        validation_targets = self.fetch(
            table="validation_targets",
            conditions={"emodel": emodel, "species": species},
        )
        if validation_targets.empty:
            logger.warning(
                "PostgreSQL warning: could not get the validation targets" " for emodel %s",
                emodel,
            )
            return []

        validation_targets = validation_targets.to_dict(orient="records")

        names = []
        for prot in validation_targets:
            names.append("{}_{}".format(prot["ecode"], prot["target"]))

        return names

    def get_features(
        self,
        emodel,
        species,
        include_validation=False,
    ):
        """Get the efeatures from the database and put in a format that fits
         the MainProtocol needs.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)
            include_validation (bool): should the features for validation be returned as well

        Returns:
            efeatures_out (dict): efeatures definitions

        """
        # Get the optimization targets
        targets = self.fetch(
            table="optimisation_targets",
            conditions={"emodel": emodel, "species": species},
        )
        if targets.empty:
            logger.warning("PostgreSQL warning: could not get the featurees for emodel %s", emodel)
            return None
        targets = targets.to_dict(orient="records")

        # Get the validation targets
        if include_validation:
            validation_targets = self.fetch(
                table="validation_targets",
                conditions={"emodel": emodel, "species": species},
            )
            if validation_targets.empty:
                logger.warning(
                    "PostgreSQL warning: could not get the validation features" " for emodel %s",
                    emodel,
                )
                return None
            validation_targets = validation_targets.to_dict(orient="records")
        else:
            validation_targets = []

        efeatures_out = {
            "RMPProtocol": {"soma.v": []},
            "RinProtocol": {"soma.v": []},
            "SearchHoldingCurrent": {"soma.v": []},
            "SearchThresholdCurrent": {"soma.v": []},
        }

        # Get the values for the efeatures matching the targets
        for target in targets + validation_targets:

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

                    if (
                        target["type"] == "RinProtocol"
                        and efeat["name"] == "ohmic_input_resistance_vb_ssse"
                    ):
                        efeatures_out["RinProtocol"]["soma.v"].append(
                            {
                                "feature": efeat["name"],
                                "val": [efeat["mean"], efeat["std"]],
                                "strict_stim": True,
                            }
                        )

                    # TODO: If the original amp of the RMP protocol is 0. It can be of:
                    # steady_state_voltage_stimend, voltage_base, voltage_deflection
                    elif (
                        target["type"] == "RMPProtocol"
                        and efeat["name"] == "steady_state_voltage_stimend"
                    ):
                        efeatures_out["RMPProtocol"]["soma.v"].append(
                            {
                                "feature": efeat["name"],
                                "val": [efeat["mean"], efeat["std"]],
                                "strict_stim": True,
                            }
                        )

                    # TODO: What we need here is the voltage when the cell is holded. Therefore
                    # it should be a global feature like hypamp and thresh,
                    # and not associated to a given protocol
                    elif target["type"] == "RinProtocol" and efeat["name"] == "voltage_base":
                        efeat["efeature"] = "steady_state_voltage_stimend"
                        efeatures_out["SearchHoldingCurrent"]["soma.v"].append(
                            {
                                "feature": "steady_state_voltage_stimend",
                                "val": [efeat["mean"], efeat["std"]],
                                "strict_stim": True,
                            }
                        )

                    else:
                        protocol_name = efeat["protocol"]
                        if protocol_name not in efeatures_out:
                            efeatures_out[protocol_name] = {"soma.v": []}

                        efeatures_out[protocol_name]["soma.v"].append(
                            {
                                "feature": efeat["name"],
                                "val": [efeat["mean"], efeat["std"]],
                                "strict_stim": True,
                            }
                        )

                        # Check if there is a stim_start and stim_end for this feature
                        if len(target["efeatures"][feat]) > 0:
                            efeatures_out[protocol_name]["soma.v"][-1]["stim_start"] = target[
                                "efeatures"
                            ][feat][0]
                            efeatures_out[protocol_name]["soma.v"][-1]["stim_end"] = target[
                                "efeatures"
                            ][feat][1]

        # Get the hypamp and thresh currents
        efeatures = self.fetch(
            table="extraction_efeatures",
            conditions={"emodel": emodel, "species": species, "protocol": "global"},
        )

        if efeatures.empty:
            logger.warning(
                "PostgreSQL warning: could not get the holding and threshold currents"
                " for emodel %s. Will ne be able to perform threshold-based optimization",
                emodel,
            )

        else:

            for efeat in efeatures.to_dict(orient="records"):

                if efeat["name"] == "hypamp":
                    efeatures_out["SearchHoldingCurrent"]["soma.v"].append(
                        {
                            "feature": "bpo_holding_current",
                            "val": [efeat["mean"], efeat["std"]],
                        }
                    )

                elif efeat["name"] == "thresh":
                    efeatures_out["SearchThresholdCurrent"]["soma.v"].append(
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
            conditions={"name": tuple([m["name"] for m in morphologies.to_dict(orient="records")])},
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

    def has_protocols_and_features(self, emodel, species=None):
        """Check if the efeatures and protocol exist."""

        targets = self.fetch(
            table="optimisation_targets",
            conditions={"emodel": emodel, "species": species},
        )

        tot_features = 0
        present_features = 0
        for target in targets.to_dict(orient="records"):

            # Check if protocol
            protocols = self.fetch(
                table="extraction_protocols",
                conditions={
                    "emodel": emodel,
                    "species": species,
                    "name": f"{target['ecode']}_{target['target']}",
                },
            )
            if protocols.empty:
                return False

            # Check if efeatures
            for feat in target["efeatures"]:
                tot_features += 1
                efeatures = self.fetch(
                    table="extraction_efeatures",
                    conditions={
                        "emodel": emodel,
                        "species": species,
                        "protocol": f"{target['ecode']}_{target['target']}",
                        "name": feat,
                    },
                )
                if not (efeatures.empty):
                    present_features += 1

        # Check if currents
        efeatures = self.fetch(
            table="extraction_efeatures",
            conditions={
                "emodel": emodel,
                "species": species,
                "protocol": "global",
            },
        )
        flag_hypamp = False
        flag_thresh = False
        for efeat in efeatures.to_dict(orient="records"):
            if efeat["name"] == "hypamp":
                flag_hypamp = True
            elif efeat["name"] == "thresh":
                flag_thresh = True

        return flag_hypamp and flag_thresh and present_features / tot_features > 0.8

    def optimisation_state(
        self, emodel, checkpoint_dir, species=None, seed=1
    ):  # pylint: disable=unused-argument
        """Return the state of the opttimisation.

        TODO: - should return three states: completed, in progress, empty
              - better management of checkpoints, we don't need checkpoint_file here?
        """
        return not self.fetch("models", {"emodel": emodel, "species": species}).empty
