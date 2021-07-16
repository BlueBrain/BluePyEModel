"""Module to simulate MCMC in emodel parameter space."""
import itertools
import logging
from collections import defaultdict
from copy import copy
from functools import partial
from pathlib import Path

import attr
import jpype as jp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bluepyparallel import evaluate
from bluepyparallel import init_parallel_factory
from joblib import Parallel
from joblib import delayed
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from bluepyemodel.access_point import get_db
from bluepyemodel.evaluation.evaluation import get_evaluator_from_db

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class ChainResult:
    """Container for data at each step."""

    parameters: list
    cost: float
    probability: float
    scores: list
    values: list


class MarkovChain:
    """Class to setup and run a markov chain on emodel parameter space."""

    def __init__(
        self,
        n_steps=100,
        result_df_path="result.csv",
        temperature=1.0,
        proposal_params=None,
        emodel_db=None,
        emodel=None,
        stochasticity=False,
        mcmc_type="metropolis_hastings",
        weights=None,
        seed=42,
    ):
        """Initialise the markov chain object."""
        self.result_df_path = result_df_path
        self.n_steps = n_steps
        self.temperature = temperature
        self.mcmc_type = mcmc_type
        self.weights = weights  # WIP: to use to weight feature differently for the cost

        self.seed = seed
        np.random.seed(self.seed)

        if proposal_params is None:
            proposal_params = {"type": "normal", "std": 0.02}
        self.proposal_params = proposal_params

        self.evaluator = get_evaluator_from_db(
            emodel,
            emodel_db,
            stochasticity=stochasticity,
        )

        self.lbounds = np.array([param.lower_bound for param in self.evaluator.params])
        self.ubounds = np.array([param.upper_bound for param in self.evaluator.params])
        self.bounds = {
            "center": 0.5 * (self.ubounds + self.lbounds),
            "width": 0.5 * (self.ubounds - self.lbounds),
        }

        self.param_names = list(self.evaluator.param_names)
        self.feature_names = [obj.name for obj in self.evaluator.fitness_calculator.objectives]

        _param_names = [("parameters", param) for param in self.param_names]
        _feature_names = [("features", feat) for feat in self.feature_names]
        self.result_df = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples(_param_names + _feature_names)
        )
        self.accepted = False

    def _un_normalize_parameters(self, parameters):
        return [
            parameter * width + center
            for parameter, center, width in zip(
                parameters, self.bounds["center"], self.bounds["width"]
            )
        ]

    def _normalize_parameters(self, parameters):
        return [
            (parameter - center) / width
            for parameter, center, width in zip(
                parameters, self.bounds["center"], self.bounds["width"]
            )
        ]

    def _probability_distribution(self, cost):
        """Convert cost function to Boltzman distribution with given temperature."""
        return np.exp(-cost / self.temperature)

    def _propose_parameters(self, parameters):
        """Propose new parameter by sampling."""
        if self.proposal_params["type"] == "normal":
            return np.clip(
                self._un_normalize_parameters(
                    np.random.normal(
                        self._normalize_parameters(parameters), self.proposal_params["std"]
                    )
                ),
                self.lbounds,
                self.ubounds,
            )

    def _evaluate(self, parameters):
        """Run evaluation and return cost."""
        parameters_dict = self.evaluator.param_dict(parameters)
        responses = self.evaluator.run_protocols(
            self.evaluator.fitness_protocols.values(), parameters_dict
        )
        scores = self.evaluator.objective_list(
            self.evaluator.fitness_calculator.calculate_scores(
                responses, self.evaluator.cell_model, parameters_dict
            )
        )
        values = self.evaluator.objective_list(
            self.evaluator.fitness_calculator.calculate_values(
                responses, self.evaluator.cell_model, parameters_dict
            )
        )
        for i, val in enumerate(values):
            try:
                values[i] = np.nanmean(val) if len(val) > 0 else None
            except (AttributeError, TypeError):
                values[i] = None

        cost = (
            np.array(self.weights).dot(np.array(scores))
            if self.weights is not None
            else sum(scores)
        )
        return ChainResult(
            parameters=parameters,
            cost=cost,
            probability=self._probability_distribution(cost),
            scores=scores,
            values=values,
        )

    def _metropolis_hastings_step(self, current):
        """Run a single metropolis-hastings step."""
        proposed_parameters = self._propose_parameters(current.parameters)
        proposed = self._evaluate(proposed_parameters)

        if proposed.probability / current.probability >= np.random.uniform(0, 1):
            self.accepted = True
            return proposed

        self.accepted = False
        return current

    def _run_one_step(self, current):
        """Run one MCMC step of choosen algorithm."""
        if self.mcmc_type == "metropolis_hastings":
            return self._metropolis_hastings_step(current)

    def _append_result(self, step, result):
        """Append result to resut dataframea and save."""
        self.result_df.loc[step, "cost"] = copy(result.cost)
        self.result_df.loc[step, "probability"] = copy(result.probability)

        for name, param in zip(self.param_names, result.parameters):
            self.result_df.loc[step, ("parameters", name)] = copy(param)
        for name, param in zip(self.param_names, self._normalize_parameters(result.parameters)):
            self.result_df.loc[step, ("normalized_parameters", name)] = copy(param)

        for name, feat in zip(self.feature_names, result.values):
            self.result_df.loc[step, ("features", name)] = copy(feat)
        for name, score in zip(self.feature_names, result.scores):
            self.result_df.loc[step, ("scores", name)] = copy(score)

        self.result_df.to_csv(self.result_df_path, index=False)

    def run(self, parameters):
        """Run the MCMC."""
        if isinstance(parameters, dict):
            parameters = [parameters[name] for name in self.param_names]

        current = self._evaluate(parameters)

        # scale the temperature with minimum cost
        self.temperature *= current.cost
        current.probability = self._probability_distribution(current.cost)

        for step in range(self.n_steps):
            logger.info(
                "Chain with seed %s at step %s / %s with cost %s",
                self.seed,
                step,
                self.n_steps,
                current.cost,
            )

            self._append_result(step, current)
            current = self._run_one_step(current)
        if self.accepted:
            self._append_result(step, current)
        return self.results_df


def _eval_one_chain(row, params=None, initial_parameters=None):
    """Internal function to evaluate one chain for parallel processing."""
    MarkovChain(**params, result_df_path=row["result_df_path"], seed=row["seed"]).run(
        initial_parameters
    )
    return {}


def run_several_chains(
    n_chains=50,
    n_steps=100,
    results_df_path="chains",
    run_df_path="run_df.csv",
    temperature=1.0,
    proposal_params=None,
    emodel_dir=None,
    final_path=None,
    legacy_dir_structure=True,
    emodel=None,
    stochasticity=False,
    mcmc_type="metropolis_hastings",
    parallel_lib="multiprocessing",
):
    """Main function to call to run several chains in parallel."""
    parallel_factory = init_parallel_factory(parallel_lib)

    Path(results_df_path).mkdir(exist_ok=True, parents=True)
    emodel_db = get_db(
        "local",
        emodel=emodel,
        emodel_dir=emodel_dir,
        final_path=final_path,
        legacy_dir_structure=legacy_dir_structure,
    )
    _eval = partial(
        _eval_one_chain,
        params=dict(
            n_steps=n_steps,
            temperature=temperature,
            proposal_params=proposal_params,
            emodel_db=emodel_db,
            emodel=emodel,
            stochasticity=stochasticity,
            mcmc_type=mcmc_type,
        ),
        initial_parameters=emodel_db.get_emodel()["parameters"],
    )
    df = pd.DataFrame()
    df["chain_id"] = list(range(n_chains))
    df["seed"] = list(range(n_chains))
    df["result_df_path"] = df["chain_id"].apply(
        lambda _id: f"{results_df_path}/results_df_{str(_id)}.csv"
    )
    df.to_csv(run_df_path, index=False)
    df = evaluate(df, _eval, parallel_factory=parallel_factory)

    # save again to get the exceptions if any
    df.to_csv(run_df_path, index=False)


def load_chains(run_df):
    """Load chains from main run_df file where the first row contains initial condition."""

    # load all but first point
    _df = []
    for res in run_df.result_df_path:
        try:
            _df.append(pd.read_csv(res, header=[0, 1]).loc[1:])
        except:
            print(f"Cannot read {res}")
    df = pd.concat(_df)
    df = df.rename(columns=lambda name: "" if name.startswith("Unnamed:") else name)
    df = df.reset_index(drop=True)

    # add the original point in front
    df_orig = pd.read_csv(run_df.result_df_path[0], header=[0, 1])
    df.loc[-1] = df_orig.loc[0]
    return df.sort_index().reset_index(drop=True)


def _plot_distributions(df, split, filename="parameters.pdf", column="normalized_parameters"):
    """Plot distributions below and above a split."""

    _df = df.copy()
    _df.loc[_df["cost"] < split, "cost"] = 0
    _df.loc[_df["cost"] >= split, "cost"] = 1

    plot_df = _df[column].reset_index(drop=True)

    df_1 = plot_df[_df["cost"] == 0].melt()
    df_1["cost"] = f"< {split}"
    df_2 = plot_df[_df["cost"] == 1].melt()
    df_2["cost"] = f">= {split}"
    plot_df = pd.concat([df_1, df_2])

    n_rows = len(plot_df.variable.unique())
    plt.figure(figsize=(3, int(0.3 * n_rows)))
    ax = plt.gca()

    sns.violinplot(
        data=plot_df,
        bw=0.005,
        orient="h",
        scale="area",
        ax=ax,
        y="variable",
        x="value",
        linewidth=0.1,
        hue="cost",
        split=True,
        order=sorted(plot_df.variable.unique()),
    )
    for i in range(n_rows):
        plt.axhline(i, c="k", ls="--", lw=0.8)

    if column == "normalized_parameters":
        plt.axvline(-1, c="k")
        plt.axvline(1, c="k")
        ax.set_xlim(-1.05, 1.05)
    if column == "scores":

        plt.axvline(0, c="k")
        plt.axvline(5, c="k")
        ax.set_xlim(-0.05, 10.00)

    plt.savefig(filename, bbox_inches="tight")


def plot_parameter_distributions(df, split, filename="figures/parameter_distributions.pdf"):
    """Plot parameter distributions below and above a split."""
    _plot_distributions(df, split, filename=filename, column="normalized_parameters")


def plot_score_distributions(df, split, filename="figures/score_distributions.pdf"):
    """Plot score below and above a split."""
    _plot_distributions(df, split, filename=filename, column="scores")


def plot_cost(df, split, filename="figures/costs.pdf"):
    """Plot histogram of costs."""
    plt.figure()
    df["cost"].plot.hist(bins=200)

    plt.axvline(df.loc[1, "cost"].to_numpy()[0], c="k", label="initial condition")
    plt.axvline(split, c="r", label="split")

    plt.legend(loc="best")
    plt.xlabel("cost")
    plt.title(f"total evaluation: {len(df.index)}, below threshold: {len(df[df['cost']<split])}")

    plt.savefig(filename, bbox_inches="tight")


def setup_jidt(jarlocation="/gpfs/bbp.cscs.ch/home/arnaudon/code/jidt/infodynamics.jar"):
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarlocation)
        jp.attachThreadToJVM()


def get_jidt_mi(tpe="MI"):

    if tpe == "MI":
        # _cls = "infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov1"
        _cls = "infodynamics.measures.continuous.gaussian.MultiInfoCalculatorGaussian"
    if tpe == "Oinfo":
        # _cls = "infodynamics.measures.continuous.kraskov.OInfoCalculatorKraskov"
        _cls = "infodynamics.measures.continuous.gaussian.OInfoCalculatorGaussian"

    _id = _cls.rfind(".")
    return getattr(jp.JPackage(_cls[:_id]), _cls[_id + 1 :])()


def _get_pair_correlation_function(correlation_type, pvalue_threshold=0.01):
    """Create  pair correlation function of given type."""
    if correlation_type == "pearson":
        from scipy.stats import pearsonr

        def corr_f(x, y):
            return pearsonr(x, y)[0]

    elif correlation_type == "MI":
        calc_mi = get_jidt_mi(tpe="MI")

        def corr_f(x, y):
            calc_mi.initialise(2)
            calc_mi.setObservations(np.array([x, y]).T)
            res = np.clip(calc_mi.computeAverageLocalOfObservations(), 0, 3)
            # si = calc_mi.computeSignificance(1000)
            # if si.pValue < pvalue_threshold:
            #    print(si.getMeanOfDistribution(), si.getStdOfDistribution(), res, si.pValue)
            #    return res
            return res

    else:
        raise Exception("Correlation type not understood")
    return corr_f


def _cluster_matrix(df, distance=False):
    """Return sorted labels to cluster a matrix with linkage.

    If distance matrix already set distance=True.
    """
    from scipy.cluster.hierarchy import dendrogram
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    with np.errstate(divide="ignore", invalid="ignore"):
        _data = df.to_numpy() if distance else 1.0 / df.to_numpy()

    _data[_data > 1e10] = 1000
    np.fill_diagonal(_data, 0)
    dists = squareform(_data)
    Z = linkage(dists, "ward")
    labels = df.columns.to_numpy()
    dn = dendrogram(Z, labels=labels, ax=None)
    return labels[dn["leaves"]]


def reduce_features(df, threshold=0.9):
    """Reduce number of feature to non-correlated features."""
    selected_features = []
    feature_map = defaultdict(list)
    for feature1 in sorted(df.index):
        to_add = True
        for feature2 in selected_features:
            if df.loc[feature1, feature2] > threshold:
                feature_map[feature2].append(feature1)
                to_add = False
        if to_add:
            selected_features.append(feature1)
    print(f"Found {len(selected_features)} out of {len(df.index)}")
    return selected_features, feature_map


def plot_pair_correlations(
    df,
    split,
    min_corr=0.3,
    column_1="normalized_parameters",
    column_2=None,
    filename="parameter_pairs.pdf",
    clip=0.4,
    correlation_type="pearson",
    with_plots=False,
):
    """Scatter plots of pairs with pearson larger than min_corr, and pearson correlation matrix.

    If column_2 is provided, the correlation will be non-square and no clustering will be applied.
    Args:
        min_corr (float): minumum correlation for plotting scatter plot
        clip (float): value to clip correlation matrix
    """
    if correlation_type != "pearson":
        setup_jidt()
    corr_f = _get_pair_correlation_function(correlation_type)
    signed = True if correlation_type == "pearson" else False

    _df = df[df["cost"] < split]
    cost = _df["cost"]
    _df_1 = _df[column_1]
    _df_2 = _df[column_2 or column_1]

    pairs = list(itertools.product(range(len(_df_1.columns)), range(len(_df_2.columns))))
    if column_2 is None:
        pairs = [p for p in pairs if p[0] != p[1]]

    if with_plots:
        pdf = PdfPages(filename)

    corr_df = pd.DataFrame()
    for id_1, id_2 in tqdm(pairs):
        col_1 = _df_1.columns[id_1]
        col_2 = _df_2.columns[id_2]
        x = _df_1[col_1].to_numpy()
        y = _df_2[col_2].to_numpy()
        corr = corr_f(x, y)
        corr_df.loc[col_1, col_2] = corr
        if column_2 is None:
            corr_df.loc[col_2, col_1] = corr
            corr_df.loc[col_1, col_1] = 0
            corr_df.loc[col_2, col_2] = 0
        if with_plots and abs(corr) > min_corr:
            plt.figure(figsize=(5, 4))
            plt.scatter(x, y, marker=".", c=cost, s=0.5)
            plt.colorbar()
            plt.scatter(x[0], y[0], marker="o", c="r", s=0.5)
            plt.xlabel(_df_1.columns[id_1])
            plt.ylabel(_df_2.columns[id_2])
            plt.suptitle(f"{correlation_type} = {np.around(corr, 2)}")
            pdf.savefig(bbox_inches="tight")
            plt.close()

    if with_plots:
        pdf.close()

    corr_df = corr_df.sort_index(axis=0).sort_index(axis=1)
    if column_2 is None:
        corr_df[corr_df.isna()] = 0.001
        sorted_labels = _cluster_matrix(corr_df.abs())
        corr_df = corr_df.loc[sorted_labels, sorted_labels]
    plt.figure(figsize=(15, 15))
    ax = plt.gca()

    corr_df[abs(corr_df) < clip] = np.nan
    if signed:
        sns.heatmap(
            data=corr_df,
            ax=ax,
            # vmin=-1, vmax=1,
            cmap="bwr",
            linewidths=1,
            linecolor="k",
        )
    else:
        sns.heatmap(
            data=corr_df,
            ax=ax,
            vmin=clip,
            # vmax=1,
            # cmap="Blues",
            linewidths=1,
            linecolor="k",
        )
    plt.savefig(
        Path(str(Path(filename).with_suffix("")) + "_matrix").with_suffix(Path(filename).suffix),
        bbox_inches="tight",
    )
    return corr_df


def _get_k_correlation_function(correlation_type, order=2):
    """Create  pair correlation function of given type."""
    if correlation_type == "Oinfo":
        calc_Oinfo = get_jidt_mi(tpe="Oinfo")

        def corr_f(x):
            calc_Oinfo.initialise(order)
            calc_Oinfo.setObservations(x)
            return calc_Oinfo.computeAverageLocalOfObservations()

    elif correlation_type == "MI":
        calc_Oinfo = get_jidt_mi(tpe="MI")

        def corr_f(x):
            calc_Oinfo.initialise(order)  # 1, order - 1)
            calc_Oinfo.setObservations(x)
            # if order - 1 > 1:
            #    calc_Oinfo.setObservations2D(x[:, 0][:, np.newaxis], x[:, 1:])
            # else:
            #    calc_Oinfo.setObservations1D(x[:, 0], x[:, 1])
            return calc_Oinfo.computeAverageLocalOfObservations()

    else:
        raise Exception("Correlation type not understood")
    return corr_f


def _compute_higher_order_single_feature(
    feature, _df_1, _df_2, param_tuples, order, correlation_type="MI"
):
    """Internal computation of higher order correlations for single feature."""
    setup_jidt()
    corr_f = _get_k_correlation_function(correlation_type, order=order)
    corr_df = pd.DataFrame()
    _feat_col = _df_1.columns[feature]
    gid = 0
    miniters = int(len(param_tuples) ** (1 / 1.5))
    for param_tuple in tqdm(param_tuples, miniters=miniters, maxinterval=1000.0):
        col_1 = _df_1.columns[feature]
        col_2 = _df_2.columns[tuple([param_tuple])]
        x = np.hstack([_df_1[col_1].to_numpy()[:, np.newaxis], _df_2[col_2].to_numpy()])
        corr = corr_f(x)

        corr_df.loc[gid, "feature"] = _feat_col
        for i, _col in enumerate(col_2):
            corr_df.loc[gid, f"param_{i}"] = _col
        corr_df.loc[gid, f"{correlation_type}"] = corr
        gid += 1

    return corr_df


def compute_higher_order(
    df,
    split,
    order=3,
    column_1="features",
    column_2="normalized_parameters",
    n_workers=50,
    correlation_type="MI",
):
    """Compute the Oinfo at any orders, in parallel over column_1."""
    _df = df[df["cost"] < split]
    _df_1 = _df[column_1]
    _df_1 = _df_1.T[_df_1.std().T > 0].T
    _df_2 = _df[column_2 or column_1]
    param_tuples = list(itertools.combinations(range(len(_df_2.columns)), order - 1))
    _compute = partial(
        _compute_higher_order_single_feature,
        _df_1=_df_1,
        _df_2=_df_2,
        order=order,
        param_tuples=param_tuples,
        correlation_type=correlation_type,
    )
    return pd.concat(
        list(
            Parallel(n_workers, verbose=10)(delayed(_compute)(i) for i in range(len(_df_1.columns)))
        )
    )


def filter_features(df):

    features = df["features"].columns.tolist()
    features_to_remove = []
    for feature in features:
        if feature.startswith("Step") and feature.split(".")[0] != "Step_200":
            features_to_remove.append(("features", feature))
        if feature.startswith("SpikeRec") and feature.endswith("Spikecount"):
            features_to_remove.append(("features", feature))
        if feature.startswith("bAP.dend2"):
            features_to_remove.append(("features", feature))
        if feature.endswith("inv_fourth_ISI") or feature.endswith("inv_fifth_ISI"):
            features_to_remove.append(("features", feature))
        if feature.endswith("APlast_amp"):
            features_to_remove.append(("features", feature))

    df = df.drop(columns=features_to_remove)
    df["features"] = df["features"].apply(lambda data: (data - data.mean()) / data.std(), axis=0)
    return df
