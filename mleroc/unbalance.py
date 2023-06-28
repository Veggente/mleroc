"""ROC estimation with unbalanced samples."""
import json
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from . import estimators, config
from .roc import get_roc


class GaussSim:
    """Gaussian simulations."""

    def __init__(self, digits: int = 4, mean_diff: float = 1, mle_only: bool = False):
        """Initialization.

        Args:
            digits: Number of digits for tolerance.
            mean_diff: Mean difference of hypotheses.
            mle_only: Does MLE only.
        """
        self._digits = digits
        self._mean_diff = mean_diff
        self._mle_only = mle_only

    @property
    def tol(self):
        """Tolerance."""
        return 10**-self._digits

    def levy_single(  # pylint: disable=too-many-locals
        self,
        n_samp: list[int],
        rng: np.random.Generator | int | None = None,
        suffix: str = "",
    ) -> list[float]:
        """Gets Levy distances from a single estimation.

        Args:
            n_samp: Numbers of likelihood ratio samples under H0 and H1.
            rng: Random number generator.
            suffix: Suffix for saving ROC curve plot.

        Returns:
            Levy metrics from MLE, empirical and CE to the true ROC.
        """
        rng = np.random.default_rng(rng)
        null = np.exp(
            rng.normal(size=n_samp[0]) * self._mean_diff - 1 / 2 * self._mean_diff**2
        )
        alt = np.exp(
            rng.normal(loc=self._mean_diff, size=n_samp[1]) * self._mean_diff
            - 1 / 2 * self._mean_diff**2
        )
        val = np.append(null, alt)
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
        amle_est = get_roc(amle_val, amle_pmf0, self.tol)
        dist_alt.append(self.levy(amle_est))
        line_width = 3
        if suffix:
            plt.figure()
            plt.plot(
                pfa,
                self.roc(pfa),
                ":",
                label="true",
                linewidth=line_width,
            )
            plt.plot(
                roc_est[0, :], roc_est[1, :], "-", label="MLE", linewidth=line_width
            )
            plt.plot(
                amle_est[0, :], amle_est[1, :], "-.", label="AMLE", linewidth=line_width
            )
        if n_samp[0] and n_samp[1] and not self._mle_only:
            n_det = estimators.split(null, alt)
            s_roc = estimators.eroc(n_det, False)
            cs_roc = estimators.eroc(n_det, True)
            if suffix:
                plt.plot(
                    s_roc[0, :],
                    s_roc[1, :],
                    "--",
                    drawstyle="steps-post",
                    label="E",
                    linewidth=line_width,
                )
                plt.plot(
                    cs_roc[0, :],
                    cs_roc[1, :],
                    "-.",
                    label="CE",
                    fillstyle="none",
                    linewidth=line_width,
                )
            dist_alt.append(self.levy(estimators.zigzag(s_roc)))
            dist_alt.append(self.levy(cs_roc))
        else:
            dist_alt.append(-1)
            dist_alt.append(-1)
        if suffix:
            plt.legend()
            plt.xlabel("prob. of false alarm")
            plt.ylabel("prob. of detection")
            plt.tight_layout()
            plt.savefig(f"{config.PATH}roc-{n_samp[0]}-{n_samp[1]}-{suffix}.pdf")
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
        if sum(piecewise[:, 0]) > self.tol:
            piecewise = np.insert(piecewise, 0, [0, 0], axis=1)
        if sum(piecewise[:, -1]) < 2 - self.tol:
            piecewise = np.append(piecewise, np.zeros((2, 1)), axis=1)
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
        filename = f"{config.PATH}levy-all-{n_samp[0]}-{n_samp[1]}-n{n_sims}-mo{self._mle_only}-mu{self._mean_diff}.json"  # pylint: disable=line-too-long
        if from_file:
            with open(filename, encoding="utf-8") as f:
                dist_full = json.load(f)
                dist = np.mean(dist_full, axis=0)
        else:
            rng = np.random.default_rng(rng)
            dist = np.zeros(4)
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
        return 1 - norm.cdf(norm.ppf(1 - pfa) - self._mean_diff)


def calc_avg_levy():
    """Calculates average Levy metrics."""
    gauss = GaussSim()
    print(gauss.levy_multiple([10, 10], 500, 0))
    print(gauss.levy_multiple([100, 100], 500, 1))
    print(gauss.levy_multiple([1000, 1000], 500, 2))
    print(gauss.levy_multiple([10, 100], 500, 3))
    print(gauss.levy_multiple([10, 1000], 500, 4))
    print(gauss.levy_multiple([100, 1000], 500, 5))
    gauss = GaussSim(mle_only=True)
    print(gauss.levy_multiple([100, 0], 500, 6))


def gen_roc_examples():
    """Generates ROC examples."""
    gauss = GaussSim()
    print(gauss.levy_single([10, 10], 0, "0"))
    print(gauss.levy_single([100, 100], 1, "1"))
    print(gauss.levy_single([1000, 1000], 2, "2"))
    print(gauss.levy_single([10, 100], 3, "3"))
    print(gauss.levy_single([10, 1000], 4, "4"))
    print(gauss.levy_single([100, 1000], 5, "5"))
    gauss = GaussSim(mle_only=True)
    print(gauss.levy_single([100, 0], 6, "6"))


def plot_avg_levy():
    """Plots average Levy metrics."""
    samples = [[10, 10], [100, 100], [1000, 1000], [10, 100], [10, 1000], [100, 1000]]
    levy_all = []
    for n_samps in samples:
        with open(
            f"{config.PATH}levy-all-{n_samps[0]}-{n_samps[1]}-n500-mo{False}-mu1.json",
            encoding="utf-8",
        ) as f:
            levy_all.append(json.load(f))
    avg = np.array([np.mean(data, axis=0) for data in levy_all])
    error = np.array(
        [np.std(data, axis=0, ddof=1) / np.sqrt(len(data)) for data in levy_all]
    )
    for i, this_avg in enumerate(avg):
        plt.figure()
        plt.bar(
            list(range(4)),
            this_avg,
            yerr=error[i],
            color=["C1", "C2", "C3", "C4"],
            capsize=10,
        )
        plt.ylim(0, 0.16)
        plt.xticks(list(range(4)), ["MLE", "AMLE", "E", "CE"])
        plt.xlabel("estimator")
        plt.ylabel("average Lévy distance")
        plt.tight_layout()
        plt.savefig(f"{config.PATH}levy-{samples[i][0]}-{samples[i][1]}.pdf")


def plot_avg_levy_mle_only():
    """Plots average Levy metric for MLE."""
    with open(f"{config.PATH}levy-all-0-100-n500-mo{True}.json", encoding="utf-8") as f:
        data = json.load(f)
    avg = np.mean(data, axis=0)
    error = np.std(data, axis=0, ddof=1) / np.sqrt(500)
    plt.figure()
    plt.bar(
        [-1, 0, 1],
        [0, avg[0], 0],
        yerr=[0, error[0], 0],
        color=["C0", "C1", "C2"],
        capsize=10,
    )
    plt.ylim(0, 0.16)
    plt.xticks([-1, 0, 1], ["", "MLE", ""])
    plt.xlabel("estimator")
    plt.ylabel("average Lévy distance")
    plt.tight_layout()
    plt.savefig(f"{config.PATH}levy-0-100.pdf")


if __name__ == "__main__":
    plt.style.use("ggplot")
    plt.rcParams.update({"font.size": 20, "pdf.fonttype": 42})
    plt.rcParams.update({"font.size": 20})
    gen_roc_examples()
    calc_avg_levy()
    plot_avg_levy()
