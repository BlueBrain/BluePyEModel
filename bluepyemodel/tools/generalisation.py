import json
from functools import partial
from itertools import cycle
from pathlib import Path

import luigi
import matplotlib
import matplotlib.pyplot as plt
import neurom as nm
import numpy as np
import pandas as pd
from bluepyparallel import init_parallel_factory
from luigi_tools.target import OutputLocalTarget
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from morph_tool.apical_point import apical_point_section_segment
from morph_tool.morphdb import MorphDB
from morph_tool.resampling import resample_linear_density
from morphio import IterType
from morphio.mut import Morphology
from scipy.stats import pearsonr
from tqdm import tqdm

from bluepyemodel.access_point import get_db
from bluepyemodel.apps.emodel_release import modify_ais
from bluepyemodel.generalisation.ais_model import get_scales
from bluepyemodel.generalisation.ais_model import taper_function
from bluepyemodel.tasks.generalisation.base_task import BaseTask
from bluepyemodel.tools.misc_evaluators import feature_evaluation

matplotlib.use("Agg")
OutputLocalTarget.set_default_prefix("out")


class GetMorphologies(BaseTask):
    """Get relevant morphs"""

    neurondb_path = luigi.Parameter(default="neurondb.xml")
    mtypes = luigi.ListParameter(default=["L5_TPC:A"])
    morphology_path = luigi.Parameter(default="original_morphologies")
    target_path = luigi.Parameter(default="dataset.csv")
    linear_density = luigi.FloatParameter(default=0.5)

    def run(self):
        """ """
        db = MorphDB.from_neurondb(Path(self.neurondb_path))
        db.df = db.df[db.df.mtype.isin(self.mtypes)]
        morph_path = (self.output().pathlib_path).parent / self.morphology_path
        morph_path.mkdir(exist_ok=True, parents=True)
        for gid in tqdm(db.df.index):
            path = morph_path / (db.df.loc[gid, "name"] + ".asc")
            resample_linear_density(db.df.loc[gid, "path"], self.linear_density).write(path)
            db.df.loc[gid, "path"] = path.resolve()
        db.df[["name", "path", "mtype"]].to_csv(self.output().path, index=False)

    def output(self):
        """ """
        return OutputLocalTarget(self.target_path)


class FixSomaAIS(BaseTask):
    """Fix soma and AIS to given size/shape."""

    ais_diameter = luigi.FloatParameter(default=2.0)
    soma_radius = luigi.FloatParameter(default=11.0)  # soma surface area of 1348
    n_soma_points = luigi.IntParameter(default=10)
    morphology_path = luigi.Parameter(default="fixed_morphologies")
    target_path = luigi.Parameter(default="fixed_dataset.csv")
    skip = luigi.BoolParameter(default=True)

    def requires(self):
        """ """
        return GetMorphologies()

    def run(self):
        """ """
        df = pd.read_csv(self.input().path)
        taper_func = partial(
            taper_function,
            strength=0,
            taper_scale=1,
            terminal_diameter=self.ais_diameter,
        )
        morph_path = (self.output().pathlib_path).parent / self.morphology_path
        morph_path.mkdir(exist_ok=True, parents=True)
        for gid in tqdm(df.index):
            _morph_path = df.loc[gid, "path"]
            morph = Morphology(_morph_path)
            if not self.skip:
                modify_ais(morph, taper_func, L_target=100)
                morph.soma.points = [
                    [self.soma_radius * np.sin(x), self.soma_radius * np.cos(x), 0]
                    for x in np.linspace(0, 2 * np.pi, self.n_soma_points + 1)[:-1]
                ]
                morph.soma.diameters = self.n_soma_points * [0.0]
            new_path = morph_path / Path(_morph_path).name
            morph.write(new_path)
            df.loc[gid, "path"] = new_path.resolve()
        df.to_csv(self.output().path, index=False)

    def output(self):
        """ """
        return OutputLocalTarget(self.target_path)


def modify_surface_area(neuron, scale=1.2, path_bin=[0, 100], neurite_type="apical"):
    """Scale diameters with scale in given path lenth bin."""
    _types = {"apical": nm.NeuriteType.apical_dendrite, "basal": nm.NeuriteType.basal_dendrite}

    pathlength = {}  # outside to act as a cache

    def _get_pathlength(section):
        """Path lenght from soma to the middle of the section."""
        if section.id not in pathlength:
            if section.parent:
                pathlength[section.id] = section.parent.length + _get_pathlength(section.parent)
            else:
                pathlength[section.id] = 0
        return pathlength[section.id]

    path_dist = []
    for neurite in neuron.neurites:
        if neurite.type == _types[neurite_type]:
            for section in nm.iter_sections(neurite):
                path_dist = _get_pathlength(section) + np.insert(
                    np.cumsum(nm.features.sectionfunc.segment_lengths(section)), 0, 0
                )
                _p = section.points
                _mask = (path_bin[0] <= path_dist) & (path_dist < path_bin[1])
                _p[_mask, nm.COLS.R] *= scale
                section.points = _p


class ScaleConfig(luigi.Config):
    """Scales configuration."""

    scale_min = luigi.FloatParameter(default=0.2)
    scale_max = luigi.FloatParameter(default=2.0)
    scale_n = luigi.IntParameter(default=10)
    scale_lin = luigi.BoolParameter(default=True)

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.scales_params = {
            "min": self.scale_min,
            "max": self.scale_max,
            "n": self.scale_n,
            "lin": self.scale_lin,
        }


class BinConfig(luigi.Config):
    """Scales configuration."""

    bin_min = luigi.FloatParameter(default=0)
    bin_max = luigi.FloatParameter(default=600)
    bin_n = luigi.IntParameter(default=20)

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.bin_params = {
            "min": self.bin_min,
            "max": self.bin_max,
            "n": self.bin_n,
        }


def get_bins(bin_params):
    """Compute path lenghs bins from parameters."""
    _b = np.linspace(bin_params["min"], bin_params["max"], bin_params["n"])
    return [[_b[i], _b[i + 1]] for i in range(bin_params["n"] - 1)]


class ModifySurfaceArea(BaseTask):
    """Scale local diameters with various scales in several path lenght bins."""

    target_path = luigi.Parameter(default="modified_dataset.csv")
    morphology_path = luigi.Parameter(default="modified_morphologies")
    neurite_type = luigi.Parameter(default="apical")
    morphology_name = luigi.Parameter(default="C060114A5")

    def requires(self):
        """ """
        return FixSomaAIS()

    def run(self):
        """ """
        morph_path = (self.output().pathlib_path).parent / self.morphology_path
        morph_path.mkdir(exist_ok=True, parents=True)
        scales = get_scales(ScaleConfig().scales_params)
        path_bins = get_bins(BinConfig().bin_params)

        df = pd.read_csv(self.input().path)
        if self.morphology_name is not None:
            df = df[df.name == self.morphology_name]

        rows = []
        for gid in tqdm(df.index):
            for scale in scales:
                for bin_id, path_bin in enumerate(path_bins):
                    _row = df.loc[gid].copy()
                    _row["scale"] = scale
                    _row["path_bin"] = path_bin
                    _row["bin_id"] = bin_id
                    neuron = nm.load_neuron(df.loc[gid, "path"])
                    modify_surface_area(
                        neuron, scale=scale, path_bin=path_bin, neurite_type=self.neurite_type
                    )
                    name = f"{df.loc[gid, 'name']}_{scale}_{bin_id}"
                    _row["path"] = morph_path / (name + ".asc")
                    _row["name"] = name
                    neuron.write(_row["path"])
                    rows.append(pd.DataFrame(_row).T)
        pd.concat(rows).to_csv(self.output().path, index=False)

    def output(self):
        """ """
        return OutputLocalTarget(self.target_path)


def bin_data(distances, data, bin_size=10, path_max=None):
    """Bin data using distances."""
    if path_max is None:
        path_max = np.max(distances)
    all_bins = np.arange(0, path_max, bin_size)
    indices = np.digitize(np.array(distances), all_bins, right=False)
    for i in list(set(indices)):
        surf = np.sum(np.array(data)[indices == i]) / bin_size
        yield all_bins[i - 1], surf


def get_apical_path_length(neuron_path):
    """Get path lengtht to apical point."""
    morph = Morphology(neuron_path)
    apical_sec = apical_point_section_segment(morph)[0]
    return sum(
        np.linalg.norm(np.diff(section.points, axis=0), axis=1).sum()
        for section in morph.sections[apical_sec].iter(IterType.upstream)
    )


def get_surface_density(
    neuron, neurite_type="basal", bin_size=50, path_max=500, apical_path_length=1.0
):
    """Compute the binned surface densities of a neuron."""
    _types = {"apical": nm.NeuriteType.apical_dendrite, "basal": nm.NeuriteType.basal_dendrite}
    area, path_dist = [], []
    for neurite in neuron.neurites:
        if neurite.type == _types[neurite_type]:
            area += list(nm.get("segment_areas", neurite))
            path_dist += list(nm.get("segment_path_lengths", neurite))
    path_dist = np.array(path_dist) / apical_path_length
    return bin_data(path_dist, area, bin_size=bin_size, path_max=path_max)


class ExtractMFeatures(BaseTask):
    """Compute morphological feature as binned surface densities as a function of path length"""

    neurite_type = luigi.Parameter(default="apical")
    target_path = luigi.Parameter(default="m_features.csv")

    def requires(self):
        """ """
        return ModifySurfaceArea()

    def run(self):
        """ """
        df = pd.read_csv(self.input().path)
        surface_df = pd.DataFrame()
        for gid in tqdm(df.index):
            path = df.loc[gid, "path"]
            neuron = nm.load_neuron(path)
            for b, s in get_surface_density(
                neuron,
                neurite_type=self.neurite_type,
                bin_size=BinConfig().bin_max / BinConfig().bin_n,
                path_max=BinConfig().bin_max,
                # apical_path_length=get_apical_path_length(path),
            ):
                surface_df.loc[df.loc[gid, "name"], b] = s
        surface_df[surface_df.isna()] = 0

        surface_df.index.name = "name"
        surface_df.to_csv(self.output().path)
        plt.figure()
        for morph in surface_df.index:
            plt.plot(surface_df.columns, surface_df.loc[morph].to_list(), "k-", lw=0.2)
        plt.xlabel("path distance")
        plt.ylabel("surface area density")

        plt.savefig((self.output().pathlib_path).parent / "apical_area.pdf")

    def output(self):
        """ """
        return OutputLocalTarget(self.target_path)


class ExtractEFeatures(BaseTask):
    """Compute electrical features."""

    emodel = luigi.Parameter(default="cADpyr_L5TPC")
    target_path = luigi.Parameter(default="e_features.csv")
    emodel_dir = luigi.Parameter(
        default="/gpfs/bbp.cscs.ch/project/proj38/sscx_emodel_paper/data/emodels/configs"
    )
    parallel_lib = luigi.Parameter(default="multiprocessing")
    nseg_frequency = luigi.FloatParameter(default=40)

    def requires(self):
        return ModifySurfaceArea()

    def run(self):

        emodel_db = get_db(
            "local",
            "cADpyr_L5TPC",
            emodel_dir=self.emodel_dir,
            legacy_dir_structure=True,
        )
        combos_df = pd.read_csv(self.input().path)
        combos_df["etype"] = self.emodel.split("_")[0]
        combos_df["emodel"] = self.emodel

        features_df = feature_evaluation(
            combos_df,
            emodel_db,
            morphology_path="path",
            parallel_factory=init_parallel_factory(self.parallel_lib),
            nseg_frequency=self.nseg_frequency,
            trace_data_path='traces',
            timeout=10000000,
        )
        features_df.to_csv(self.output().path)

    def output(self):
        """ """
        return OutputLocalTarget(self.target_path)


class PlotCorrelations(BaseTask):
    """Plot correlation between m and e features."""

    target_path = luigi.Parameter(default="feat_corr.pdf")

    def requires(slf):
        """ """
        return {
            "m_feature": ExtractMFeatures(),
            "e_feature": ExtractEFeatures(),
            "dataset": ModifySurfaceArea(),
        }

    def run(self):
        """ """
        surface_df = pd.read_csv(self.input()["m_feature"].path).set_index("name")
        scores_df = pd.read_csv(self.input()["e_feature"].path).set_index("name")

        df = pd.read_csv(self.input()["dataset"].path).set_index("name")
        df["min_bin"] = df["path_bin"].apply(lambda b: json.loads(b)[0])
        df["max_bin"] = df["path_bin"].apply(lambda b: json.loads(b)[1])

        feat_df = pd.DataFrame()
        for name in scores_df.index:
            for feat, val in json.loads(scores_df.loc[name, "features"]).items():
                feat_df.loc[name, feat] = val
        feat_path = Path(self.output().pathlib_path.parent) / "features_plot"
        feat_path.mkdir(exist_ok=True, parents=True)
        for feat in feat_df.columns:
            plt.figure()
            colors = plt.cm.jet(np.linspace(0, 1, len(surface_df.columns)))
            cmappable = ScalarMappable(
                norm=Normalize(surface_df.columns[0], surface_df.columns[-1]), cmap="jet"
            )
            for col in range(len(surface_df.columns)):
                _s = surface_df.copy()
                _f = feat_df.copy()
                mask = (float(surface_df.columns[col]) >= df["min_bin"]) & (
                    float(surface_df.columns[col]) <= df["max_bin"]
                )
                _s = _s[mask]
                _f = _f[mask]
                scales = _s.reset_index()["name"].apply(lambda name: float(name.split("_")[1]))
                original_scale = _s[_s.columns[col]].iloc[np.argmin(abs(scales - 1))]
                original_feature = _f[feat].iloc[np.argmin(abs(scales - 1))]
                plt.plot(
                    _s[_s.columns[col]].to_numpy(dtype=float) / original_scale,
                    _f[feat] / original_feature,
                    "-",
                    c=colors[col],
                    lw=1.0,
                )
            plt.xlabel("% of original surface area")
            plt.ylabel("% of original feature value")
            plt.colorbar(cmappable, label="path distance")
            plt.savefig(f"{feat_path}/test_{feat}.pdf")
            plt.close()

        COLORS = {}
        for feat in feat_df.columns:
            if feat.split(".")[0] == "bAP":
                continue
            var = feat_df[feat].std() / abs(feat_df[feat].mean())
            corrs = []
            for dist in surface_df.columns:
                _s = surface_df.copy()
                _f = feat_df.copy()
                mask = (float(dist) >= df["min_bin"]) & (float(dist) <= df["max_bin"])
                _s = _s[mask]
                _f = _f[mask]
                x, y = _s[dist].to_numpy(), _f[feat].to_numpy()
                x = x[~np.isnan(y)]
                y = y[~np.isnan(y)]
                if len(y) > 2:
                    corrs.append(pearsonr(x, y)[0])
                else:
                    corrs.append(0)
            fig_name = ".".join(feat.split(".")[1:])
            plt.figure(fig_name, figsize=(10, 6))
            if fig_name not in COLORS:
                COLORS[fig_name] = cycle(["C{}".format(i) for i in range(10)])
            c = next(COLORS[fig_name])
            plt.plot(
                surface_df.columns.to_numpy(dtype="float"),
                corrs,
                "-",
                c=c,
                label=f"apical, {feat.split('.')[0]}, variability = {np.round(var ,4)}",
            )
            plt.axhline(0, ls="--", c="k")
            plt.axhline(-1, ls="--", c="k")
            plt.axhline(1, ls="--", c="k")
            plt.xlabel("path distance")
            plt.ylabel("pearson correlation")
            plt.gca().set_ylim(-1.01, 1.01)

        with PdfPages(self.output().path) as pdf:
            for fig_id in plt.get_fignums():
                fig = plt.figure(fig_id)
                plt.legend(loc="best")
                plt.suptitle(fig.get_label())
                pdf.savefig()
                plt.close()

    def output(self):
        """ """
        return OutputLocalTarget(self.target_path)
