"""ROC estimation with unbalanced samples."""

import json
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma
from scipy.stats import norm
from tqdm import tqdm

from config import config
from mleroc import estimators
from mleroc.roc import get_roc

EstimatorName = Literal["MLE", "Split", "Fused", "E", "CE"]


@dataclass(kw_only=True)
class Composition:
    allow_no_neg: bool = False
    allow_no_pos: bool = False


@dataclass(kw_only=True)
class Estimator:
    name: EstimatorName
    color: int
    index: int
    style: str = "-"
    comp: Composition = field(default_factory=Composition)


@dataclass(kw_only=True)
class LRSamples:
    null: np.ndarray
    alt: np.ndarray


class Distribution(Protocol):
    def generate_samples(
        self, n_samp: list[int], rng: np.random.Generator | int | None = None
    ) -> LRSamples:
        """Generate samples from the distribution."""
        ...

    def roc(self, pfa: np.ndarray) -> np.ndarray:
        """ROC curve."""
        ...

    def name(self) -> str:
        """Name of the distribution."""
        ...


@dataclass(kw_only=True)
class Gaussian:
    mean_diff: float = 1

    def generate_samples(
        self, n_samp: list[int], rng: np.random.Generator | int | None = None
    ) -> LRSamples:
        rng = np.random.default_rng(rng)
        null = np.exp(
            rng.normal(size=n_samp[0]) * self.mean_diff - 1 / 2 * self.mean_diff**2
        )
        alt = np.exp(
            rng.normal(loc=self.mean_diff, size=n_samp[1]) * self.mean_diff
            - 1 / 2 * self.mean_diff**2
        )
        return LRSamples(null=null, alt=alt)

    def roc(self, pfa: np.ndarray) -> np.ndarray:
        return 1 - norm.cdf(norm.ppf(1 - pfa) - self.mean_diff)

    def name(self) -> str:
        return "gaussian"


@dataclass(kw_only=True)
class Gamma:

    def generate_samples(
        self, n_samp: list[int], rng: np.random.Generator | int | None = None
    ) -> LRSamples:
        rng = np.random.default_rng(rng)
        null = rng.gamma(shape=1.0, scale=1.0, size=n_samp[0])
        alt = rng.gamma(shape=2.0, scale=1.0, size=n_samp[1])
        return LRSamples(null=null, alt=alt)

    def roc(self, pfa: np.ndarray) -> np.ndarray:
        return 1 - gamma.cdf(-np.log(pfa + 0.0000000000001), 2)

    def name(self) -> str:
        return "gamma"


class BHTSim:
    """Binary hypothesis testing simulations."""

    def __init__(
        self,
        digits: int = 4,
        distribution: Distribution | None = None,
    ):
        """Initialization.

        Args:
            digits: Number of digits for tolerance.
            distribution: Distribution of the hypotheses.
        """
        self._digits = digits
        self._distribution = distribution or Gaussian()

    @property
    def tol(self):
        """Tolerance."""
        return 10**-self._digits

    def levy_single(
        self,
        n_samp: list[int],
        rng: np.random.Generator | int | None = None,
        plot: bool = False,
        plot_estimators: set[EstimatorName] | None = None,
    ) -> list[float]:
        """Gets Levy distances from a single estimation.

        Args:
            n_samp: Numbers of likelihood ratio samples under H0 and H1.
            rng: Random number generator.
            plot: Whether to plot the ROC curve.
            plot_estimators: Estimators to plot.

        Returns:
            Levy metrics from MLE, empirical and CE to the true ROC.
        """
        if plot_estimators is None:
            plot_estimators = set()
        rng = np.random.default_rng(rng)
        lr_samples = self._distribution.generate_samples(n_samp, rng)
        val = np.append(lr_samples.null, lr_samples.alt)
        val.sort()
        count = np.concatenate(([0], np.ones(sum(n_samp)), [0]))
        pmf, _ = estimators.mle(val, count)
        pfa = np.linspace(0, 1, 101)
        roc_est = get_roc(np.insert(val, 0, 0), pmf, self.tol)
        dist_alt = [self.levy(roc_est)]
        amle_val, amle_pmf0 = estimators.amle(val, count, n_samp[1] / sum(n_samp))
        if amle_pmf0[0]:
            amle_val = np.insert(amle_val, 0, 0)
        else:
            amle_pmf0 = amle_pmf0[1:]
        try:
            amle_est = get_roc(amle_val, amle_pmf0, self.tol)
            dist_alt.append(self.levy(amle_est))
        except ValueError:
            dist_alt.append(-1)
        split_est = estimators.amle_lin(
            val, count, n_samp[1] / sum(n_samp), np.round(n_samp[1] / sum(n_samp))
        )
        amle_lin_est = estimators.amle_lin(
            val, count, n_samp[1] / sum(n_samp), n_samp[1] / sum(n_samp)
        )
        dist_alt.append(self.levy(split_est))
        dist_alt.append(self.levy(amle_lin_est))
        line_width = 3
        if plot:
            plt.figure()
            plt.plot(
                pfa,
                self.roc(pfa),
                ":",
                label="true",
                linewidth=line_width,
                color="C0",
            )
            if "MLE" in plot_estimators:
                plt.plot(
                    roc_est[0, :],
                    roc_est[1, :],
                    "-",
                    label="MLE",
                    linewidth=line_width,
                    color="C1",
                )
            if "Split" in plot_estimators:
                plt.plot(
                    split_est[0, :],
                    split_est[1, :],
                    "--",
                    label="Split",
                    linewidth=line_width,
                    color="C2",
                )
            if "Fused" in plot_estimators:
                plt.plot(
                    amle_lin_est[0, :],
                    amle_lin_est[1, :],
                    "--",
                    label="AMLE-lin",
                    linewidth=line_width,
                    color="C3",
                )
        if n_samp[0] and n_samp[1]:
            n_det = estimators.tp_at_fp(lr_samples.null, lr_samples.alt)
            s_roc = estimators.eroc(n_det, False)
            cs_roc = estimators.eroc(n_det, True)
            if plot:
                if "E" in plot_estimators:
                    plt.plot(
                        s_roc[0, :],
                        s_roc[1, :],
                        "--",
                        drawstyle="steps-post",
                        label="E",
                        linewidth=line_width,
                        color="C4",
                    )
                if "CE" in plot_estimators:
                    plt.plot(
                        cs_roc[0, :],
                        cs_roc[1, :],
                        "-.",
                        label="CE",
                        fillstyle="none",
                        linewidth=line_width,
                        color="C5",
                    )
            dist_alt.append(self.levy(estimators.zigzag(s_roc)))
            dist_alt.append(self.levy(cs_roc))
        else:
            dist_alt.append(-1)
            dist_alt.append(-1)
        if plot:
            plt.legend()
            plt.xlabel("prob. of false alarm")
            plt.ylabel("prob. of detection")
            plt.tight_layout()
            plt.savefig(
                f"{config.config['paths']['plots_dir']}roc-{n_samp[0]}-{n_samp[1]}.pdf"
            )
        return dist_alt

    def levy(self, piecewise: np.ndarray) -> float:
        """Lévy metric w.r.t. the Gaussian ROC curve.

        Calculates ROC at discrete values first, and then projects to the
        ROC estimate.

        Args:
            piecewise: The points that define the piecewise nondecreasing
                function.  Each column is a point.

        Returns:
            The Lévy metric from piecewise to the true Gaussian ROC.
        """
        piecewise = _force_nondecreasing_x_axis(piecewise)
        piecewise = _delete_duplicates(piecewise)

        if sum(piecewise[:, 0]) > self.tol:
            piecewise = np.insert(piecewise, 0, [0, 0], axis=1)
        if sum(piecewise[:, -1]) < 2 - self.tol:
            piecewise = np.append(piecewise, np.ones((2, 1)), axis=1)
        pfa = np.linspace(0, 1, 10**self._digits + 1)
        begin = 0
        max_dist = 0
        for false_alarm, detection in zip(pfa, self.roc(pfa)):
            while sum(piecewise[:, begin + 1]) < false_alarm + detection:
                begin += 1
            alpha = (sum(piecewise[:, begin + 1]) - false_alarm - detection) / (
                sum(piecewise[:, begin + 1]) - sum(piecewise[:, begin])
            )

            dist = abs(
                alpha * piecewise[0, begin]
                + (1 - alpha) * piecewise[0, begin + 1]
                - false_alarm
            )
            if dist > max_dist:
                max_dist = dist
        return max_dist

    def levy_multiple(
        self,
        n_samp: list[int],
        n_sims: int,
        *,
        rng: np.random.Generator | int | None = None,
        from_file: bool = False,
    ) -> np.ndarray:
        """Average Levy metrics from multiple estimations.

        Args:
            n_samp: Numbers of samples.
            n_sims: Number of simulations.
            rng: Random number generator.
            from_file: Loads data from file.

        Returns:
            Average Levy metrics.
        """
        filename = _get_filename(n_samp, n_sims)
        if from_file:
            with open(filename, encoding="utf-8") as f:
                dist_full = json.load(f)
                dist = np.mean(dist_full, axis=0)
        else:
            rng = np.random.default_rng(rng)
            dist = np.zeros(6)
            dist_full = []
            for _ in tqdm(range(n_sims)):
                new_dist = self.levy_single(n_samp, rng)
                dist += np.array(new_dist)
                dist_full.append(new_dist)
            with open(
                filename,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(dist_full, f)
            dist = dist / n_sims
        return dist

    def roc(self, pfa: np.ndarray) -> np.ndarray:
        """ROC curve.

        Args:
            pfa: Probability of false alarm.

        Returns:
            Maximum probability of detection.
        """
        return self._distribution.roc(pfa)


@dataclass(kw_only=True)
class SamplePair:
    pos: int
    neg: int
    example_rng: int | None = None
    estimators_in_example: set[EstimatorName] | None = None
    estimators_in_avg_levy: list[Estimator] | None = None

    def __post_init__(self):
        if self.estimators_in_example is None:
            self.estimators_in_example = set()
        if self.estimators_in_avg_levy is None:
            self.estimators_in_avg_levy = []

    def get_output_path(self) -> str:
        return f"{config.config['paths']['plots_dir']}levy-{self.neg}-{self.pos}.pdf"

    def mle_only(self) -> bool:
        """Only run MLE based estimators for single-class samples."""
        return not self.neg or not self.pos


def calc_avg_levy(samples: list[SamplePair], n_sims: int):
    """Calculates average Levy metrics."""
    gauss = BHTSim()
    for sample_pair in samples:
        print(gauss.levy_multiple([sample_pair.neg, sample_pair.pos], n_sims))


def gen_roc_examples(samples: list[SamplePair]):
    """Generates ROC examples."""
    gauss = BHTSim()
    for sample_pair in samples:
        _ = gauss.levy_single(
            [sample_pair.neg, sample_pair.pos],
            rng=sample_pair.example_rng,
            plot=True,
            plot_estimators=sample_pair.estimators_in_example,
        )


def plot_avg_levy(samples: list[SamplePair], n_sims: int, ylim_upper: float = 0.16):
    """Plots average Levy metrics."""
    for sample_pair in samples:
        with open(
            _get_filename([sample_pair.neg, sample_pair.pos], n_sims),
            encoding="utf-8",
        ) as f:
            levy_data = json.load(f)
        plt.figure()
        for idx, est in enumerate(sample_pair.estimators_in_avg_levy):
            data = np.array(levy_data)[:, est.index]
            avg = data.mean()
            error = np.std(data, ddof=1) / np.sqrt(len(data))
            plt.bar(
                idx,
                avg,
                yerr=error,
                color=f"C{est.color}",
                capsize=10,
            )
        plt.ylim(0, ylim_upper)
        plt.xticks(
            list(range(len(sample_pair.estimators_in_avg_levy))),
            [est.name for est in sample_pair.estimators_in_avg_levy],
        )
        plt.xlabel("estimator")
        plt.ylabel("average Lévy distance")
        plt.tight_layout()
        plt.savefig(sample_pair.get_output_path())


def main():
    plt.style.use("ggplot")
    plt.rcParams.update({"font.size": 20, "pdf.fonttype": 42})
    plt.rcParams.update({"font.size": 20})
    samples = _structure_sample_pairs(config.config["average-levy"]["sample_pairs"])
    gen_roc_examples(samples)
    calc_avg_levy(samples, config.config["average-levy"]["n_sims"])
    plot_avg_levy(
        samples,
        config.config["average-levy"]["n_sims"],
    )
    error_v_diff(
        [
            structure_estimator(est)
            for est in config.config["average-levy"]["vary_mu"]["estimators"]
        ],
        config.config["average-levy"]["n_sims"],
    )
    error_v_composition(
        [
            structure_estimator(est)
            for est in config.config["average-levy"]["vary_alpha"]["estimators"]
        ],
        config.config["average-levy"]["n_sims"],
    )


def _structure_sample_pairs(sample_pairs: list[dict[str, Any]]) -> list[SamplePair]:
    structured_sample_pairs = []
    for sample_pair in sample_pairs:
        estimators_in_avg_levy = [
            structure_estimator(est) for est in sample_pair["estimators_in_avg_levy"]
        ]
        structured_sample_pairs.append(
            SamplePair(
                neg=sample_pair["neg"],
                pos=sample_pair["pos"],
                example_rng=sample_pair["example_rng"],
                estimators_in_example=set(sample_pair["estimators_in_example"]),
                estimators_in_avg_levy=estimators_in_avg_levy,
            )
        )
    return structured_sample_pairs


def structure_estimator(estimator: dict[str, Any]) -> Estimator:
    return Estimator(
        name=estimator["name"],
        color=estimator["color"],
        index=estimator["index"],
        style=estimator.get("style", "-"),
        comp=Composition(
            allow_no_neg=estimator.get("comp", {}).get("allow_no_neg", False),
            allow_no_pos=estimator.get("comp", {}).get("allow_no_pos", False),
        ),
    )


def _delete_duplicates(piecewise: np.ndarray) -> np.ndarray:
    """Delete duplicate points from a piecewise linear function."""
    return np.unique(piecewise, axis=1)


def _force_nondecreasing_x_axis(piecewise: np.ndarray) -> np.ndarray:
    """Force a piecewise linear function to be nondecreasing on the x-axis.

    E.g., if the input is:
    [[0, 0, 0.5, 0.4, 1], [0, 0.5, 0.5, 1, 1]],
    the output will be:
    [[0, 0, 0.5, 0.5, 1], [0, 0.5, 0.5, 1, 1]].
    """
    for i in range(len(piecewise[0]) - 1):
        if piecewise[0, i] > piecewise[0, i + 1]:
            piecewise[0, i + 1] = piecewise[0, i]
    return piecewise


def error_v_diff(estimators: list[Estimator], n_sims: int):
    """Evaluates errors of MLE with respect to difference in hypotheses."""
    diff_list = [0.1] + np.linspace(0.5, 4, 8).tolist()
    dist = []
    for diff in diff_list:
        gauss = BHTSim(distribution=Gaussian(mean_diff=diff))
        dist.append(gauss.levy_multiple([100, 100], n_sims, rng=42))
    dist = np.array(dist)
    plt.figure()
    line_width = 3
    for est in estimators:
        plt.plot(
            diff_list,
            dist[:, est.index],
            est.style,
            label=est.name,
            linewidth=line_width,
            color=f"C{est.color}",
            markersize=12,
            fillstyle="none",
            markeredgewidth="2",
        )
    plt.legend()
    plt.xlabel(r"mean difference $\mu$")
    plt.ylabel("average Lévy distance")
    plt.tight_layout()
    plt.savefig(f"{config.config['paths']['plots_dir']}levy-v-diff-delta-mu.pdf")


def error_v_composition(
    estimators: list[Estimator], n_sims: int, ylim_upper: float = 0.11
):
    """Evaluates errors with different sample composition."""
    total_samp = 200
    gauss = BHTSim()
    error = []
    rng = np.random.default_rng(42)
    alpha_list = np.linspace(0, 1, 11)
    for alpha in alpha_list:
        error.append(
            gauss.levy_multiple(
                [int(total_samp * alpha), int(total_samp * (1 - alpha))],
                n_sims,
                rng=rng,
            )
        )
    error = np.array(error)
    plt.figure()
    line_width = 3
    for est in estimators:
        plt.plot(
            _get_alpha_list(alpha_list, est.comp),
            error[_get_alpha_list(np.arange(len(alpha_list)), est.comp), est.index],
            est.style,
            label=est.name,
            linewidth=line_width,
            color=f"C{est.color}",
            markersize=12,
            fillstyle="none",
            markeredgewidth="2",
        )
    plt.legend()
    plt.xlabel(r"fraction of null samples $\alpha$")
    plt.ylabel("average Lévy distance")
    plt.ylim(0, ylim_upper)
    plt.tight_layout()
    plt.savefig(f"{config.config['paths']['plots_dir']}levy-v-alpha.pdf")


def _get_alpha_list(alpha_list: np.ndarray, comp: Composition) -> np.ndarray:
    alpha_list = alpha_list.tolist()
    if not comp.allow_no_neg:
        alpha_list = alpha_list[1:]
    if not comp.allow_no_pos:
        alpha_list = alpha_list[:-1]
    return np.array(alpha_list)


def _get_filename(n_samp: list[int], n_sims: int) -> str:
    return f"{config.config['paths']['data_dir']}levy-all-{n_samp[0]}-{n_samp[1]}-n{n_sims}.json"


if __name__ == "__main__":
    main()
