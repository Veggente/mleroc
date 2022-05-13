"""MLE for likelihoods."""
import math
import json
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
import bhatta_bound

plt.style.use("ggplot")
plt.rcParams.update({"font.size": 20, "pdf.fonttype": 42})
PATH = ""


@dataclass
class ROC:
    """Receiver operating characteristic curve.

    Linear interpolation is done for two adjacent points.

    Attributes:
        pfa: np.ndarray
            Probability of false alarm.  Must be nondecreasing (but
            not necessarily strictly increasing to allow vertical
            jumps in ROC).
        pdet: np.ndarray
            Probability of detection.  Must be nondecreasing (but not
            necessarily strictly increasing to allow horizontal lines
            in ROC).
        name: str
            ROC name.
    """

    pfa: np.ndarray
    pdet: np.ndarray
    name: str

    def plot(self):
        """Plots ROC curve."""
        plt.figure()
        plt.plot(
            self.pfa,
            self.pdet,
        )
        plt.xlabel("prob. of false alarm")
        plt.ylabel("prob. of detection")
        plt.tight_layout()
        plt.show()


class ROCSet:  # pylint: disable=too-few-public-methods
    """A set of ROCs."""

    def __init__(self, list_roc: list[ROC]):
        """Initialization."""
        self.rocs = list_roc

    def plot(self, saveas: str = ""):
        """Plots ROCs."""
        plt.figure()
        for roc in self.rocs:
            plt.plot(
                roc.pfa,
                roc.pdet,
                label=roc.name,
            )
        plt.xlabel("prob. of false alarm")
        plt.ylabel("prob. of detection")
        plt.tight_layout()
        plt.legend()
        if saveas:
            plt.savefig(saveas)
        else:
            plt.show()


class MLE:  # pylint: disable=too-few-public-methods
    """ML estimator."""

    def __init__(self, likelihoods: list[float]):
        """Initialization."""
        self.likelihoods = likelihoods

    def roc(self, label="MLE") -> ROC:
        """ML estimator of the ROC."""
        pmf = mle(
            np.array(self.likelihoods),
            np.array([0] + [1] * len(self.likelihoods) + [0]),
        )
        roc = get_roc(np.insert(self.likelihoods, 0, 0), pmf[0], 1e-6)
        return ROC(roc[0, :], roc[1, :], label)


class GaussROC:
    """ROC of binormal BHT."""

    def __init__(self, means: list[float], var: list[float]):
        """Initialization.

        Args:
            means: Means of binormal.
            var: Variances of binormal.
        """
        self.means = means
        self.var = var

    def bhatta(self) -> float:
        """Bhattacharyya coefficient."""
        return (
            np.exp(-((self.means[0] - self.means[1]) ** 2) / 4 / sum(self.var))
            * (self.var[0] * self.var[1]) ** (1 / 4)
            / np.sqrt(sum(self.var) / 2)
        )

    def plot(self, saveas: str = ""):
        """Plots ROC bounds."""
        pfa = np.linspace(0, 1, 101)
        roc_set = ROCSet(
            [
                ROC(pfa, bhatta_bound.new_roc_bound([self.bhatta() ** 2], pfa), "BC"),
                ROC(
                    pfa,
                    np.array([gauss_roc(false_alarm)[0] for false_alarm in pfa]),
                    "true",
                ),
            ]
        )
        roc_set.plot(saveas)


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

    def levy_single(
        self,
        n_samp: list[int],
        rng: Optional[Union[np.random.Generator, int]] = None,
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
        count = np.concatenate(([0], np.ones(sum(n_samp)), [0]))
        pmf, _ = mle(val, count)
        pfa = np.linspace(0, 1, 101)
        roc_est = get_roc(np.insert(val, 0, 0), pmf, self.tol)
        dist_alt = [self.levy(roc_est)]
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
        if n_samp[0] and n_samp[1] and not self._mle_only:
            n_det = split(null, alt)
            s_roc = eroc(n_det, False)
            cs_roc = eroc(n_det, True)
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
            dist_alt.append(self.levy(zigzag(s_roc)))
            dist_alt.append(self.levy(cs_roc))
        else:
            dist_alt.append(-1)
            dist_alt.append(-1)
        if suffix:
            plt.legend()
            plt.xlabel("prob. of false alarm")
            plt.ylabel("prob. of detection")
            plt.tight_layout()
            plt.savefig(f"{PATH}roc-{n_samp[0]}-{n_samp[1]}-{suffix}.pdf")
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
        rng: Optional[Union[np.random.Generator, int]] = None,
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
        filename = f"{PATH}levy-all-{n_samp[0]}-{n_samp[1]}-n{n_sims}-mo{self._mle_only}-mu{self._mean_diff}.json"  # pylint: disable=line-too-long
        if from_file:
            with open(filename, encoding="utf-8") as f:
                dist_full = json.load(f)
                dist = np.mean(dist_full, axis=0)
        else:
            rng = np.random.default_rng(rng)
            dist = np.zeros(3)
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


def mle(val: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, float]:
    """Finds MLE given likelihoods.

    The MLE is represented by the pmf estimate on F_0 over 0 and val.

    Args:
        val: Likelihood values (excluding zero and infinity).
        count: Counts for likelihood values (including zero and
            infinity).  Only the first and the last counts can be
            zero.

    Returns:
        pmf of F_0 and the Lagrangian dual variable lambda.
    """

    def dual_for_fsolve(
        lag_mul: float, *data: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        return dual_single(lag_mul, data[0], data[1])

    if not count[-1] and sum(val * count[1:-1]) <= sum(count):
        lag_mul = 1
        return count[:-1] / lag_mul / sum(count), lag_mul
    if not count[0] and sum(count[1:-1] / val) <= sum(count):
        lag_mul = 0
        return (
            np.array(
                [1 - sum(count[1:-1] / val) / sum(count)]
                + list(count[1:-1] / val / sum(count))
            ),
            lag_mul,
        )
    lag_mul = fsolve(dual_for_fsolve, 0.1, args=(val, count))
    return (
        np.insert(
            count[1:-1] / (lag_mul + (1 - lag_mul) * val) / sum(count),
            0,
            count[0] / lag_mul / sum(count),
        ),
        lag_mul,
    )


def dual_single(lag_mul: float, val: np.ndarray, count: np.ndarray) -> float:
    """Weighted average minus 1 for estimate of F_0.

    Computes phi_n(lambda_n) - 1.

    Args:
        lag_mul: The normalized Lagrangian multiplier.  Must be
            strictly between 0 and 1.
        val: Likelihood values (excluding zero and infinity).
        count: Counts for likelihood values (including zero and
            infinity).  Only the first and the last counts can be
            zero.  Sum must be positive.

    Returns:
        Weighted average minus 1.
    """
    return (
        sum(count[1:-1] / (lag_mul + (1 - lag_mul) * val)) + count[0] / lag_mul
    ) / sum(count) - 1


def gauss_roc(pfa: float, tol: float = 1e-9) -> np.ndarray:
    """Gaussian hypothesis true ROC curve.

    OBSOLETE.

    Args:
        pfa: Probability of false alarm.  Assumed to be increasing.
        tol: Tolerance.

    Returns:
        Maximum probability of detection and the derivative.
    """
    if math.isclose(pfa, 0, abs_tol=tol):
        return np.array([0, np.inf])
    if math.isclose(pfa, 1, abs_tol=tol):
        return np.array([1, 0])
    return np.array(
        [
            1 - norm.cdf(norm.ppf(1 - pfa) - 1),
            norm.pdf(norm.ppf(1 - pfa) - 1) / norm.pdf(norm.ppf(1 - pfa)),
        ]
    )


def get_roc(val: np.ndarray, pmf: np.ndarray, tol: float) -> np.ndarray:
    """Gets ROC from F0 pmf.

    F0 is assumed to be valid; i.e., sum_r (r * pmf(r)) >= 1.

    Args:
        val: Values of pmf.  Must be all nonnegative.
        pmf: Masses of pmf.  Must be all positive.
        tol: Tolerance.

    Returns:
        A 2xn array of points on the ROC curve including (0, 0) and
        (1, 1).
    """
    sorted_pmf = np.array([val, pmf])
    sorted_pmf = sorted_pmf[:, np.argsort(sorted_pmf[0, :])]
    points = [[1, 1]]
    for slope, h_dist in sorted_pmf.T:
        pfa, pdet = points[-1]
        points.append([pfa - h_dist, pdet - h_dist * slope])
    if not math.isclose(points[-1][0], 0, abs_tol=tol):
        raise ValueError("F0 pmf does not sum to 1.")
    if points[-1][1] < -tol:
        raise ValueError("F0 pmf is not valid.")
    if math.isclose(points[-1][1], 0, abs_tol=tol):
        points.pop()
    points.append([0, 0])
    return np.array(points[::-1]).T


def eroc(n_det: list[int], concavify: bool) -> np.ndarray:
    """Empirical ROC estimator.

    Args:
        n_det: Numbers of detection w.r.t. numbers of false alarm.
        concavify: Concavifies the curve.

    Returns:
        A 2xn array of points on the concavified ROC curve.
    """
    points = [[1, 1]]
    for i, det in enumerate(n_det[-2::-1]):
        pfa = 1 - (i + 1) / (len(n_det) - 1)
        pdet = det / n_det[-1]
        if concavify:
            while len(points) > 1:
                if (points[-1][1] - pdet) / (points[-1][0] - pfa) <= (
                    points[-2][1] - pdet
                ) / (points[-2][0] - pfa):
                    points.pop()
                else:
                    break
        points.append([pfa, pdet])
    return np.array(points[::-1]).T


def split(null: np.ndarray, alt: np.ndarray) -> list[int]:
    """Split estimator for ROC.

    Assumes all values are distinct.

    Args:
        null: Likelihood ratios for null hypothesis.
        alt: Likelihood ratios for alternative hypothesis.

    Returns:
        Numbers of detection at numbers of false alarm.
    """
    return [sum(alt > threshold) for threshold in np.sort(null)[::-1]] + [len(alt)]


def levy(func: np.ndarray, cont: bool, tol: float = 1e-9) -> float:
    """Lévy metric w.r.t. the Gaussian ROC curve.

    OBSOLETE.

    Args:
        func: The points that define the nondecreasing function.
        cont: Does linear interpolation if True, and horizontal hold
            otherwise.
        tol: Tolerance.

    Returns:
        The Lévy metric from func to the true Gaussian ROC.
    """

    def update(begin: np.ndarray, end: np.ndarray) -> float:
        """Updates stats for a segment.

        Args:
            begin: Begin point.
            end: End point.

        Returns:
            New distance.
        """
        nonlocal prev_proj, prev_slope
        new_dist, prev_proj, prev_slope = levy_linear(
            begin, end, prev_proj, prev_slope, tol
        )
        return new_dist

    dist = 0
    prev_proj, prev_slope = proj(func[:, 0], tol)  # pylint: disable=unused-variable
    for i in range(func.shape[1] - 1):
        if cont:
            new_dist = update(func[:, i], func[:, i + 1])
            dist = max(dist, new_dist)
        else:
            new_dist = update(func[:, i], np.array([func[0, i + 1], func[1, i]]))
            dist = max(dist, new_dist)
            new_dist = update(np.array([func[0, i + 1], func[1, i]]), func[:, i + 1])
            dist = max(dist, new_dist)
    return dist


def levy_linear(
    begin: np.ndarray,
    end: np.ndarray,
    prev_proj: np.ndarray,
    prev_slope: float,
    tol: float = 1e-9,
) -> tuple[float, np.ndarray, float]:
    """Lévy metric for a linear piece.

    Requires begin <= end.

    Args:
        begin: Coordinates of the lower left point.
        end: Coordinates of the upper right point.
        prev_proj: Previous projection.
        prev_slope: Previous slope.
        tol: Tolerance.

    Returns:
        Lévy distance to the true ROC, projection of the end point,
        and the slope at the projection.
    """
    begin_disp = begin[0] - prev_proj[0]
    horiz_inc = sum(end) - sum(begin)
    end_proj, end_slope = proj(
        end,
        tol,
        (prev_proj[0] + horiz_inc / (1 + prev_slope), min(prev_proj[0] + horiz_inc, 1)),
    )
    end_disp = end[0] - end_proj[0]
    max_end = max(abs(begin_disp), abs(end_disp))
    if math.isclose(begin[0], end[0], abs_tol=tol) or math.isclose(
        begin[1], end[1], abs_tol=tol
    ):
        return max_end, end_proj, end_slope
    slope = (end[1] - begin[1]) / (end[0] - begin[0])
    left = prev_proj[0]
    right = end_proj[0]
    if gauss_roc(left, tol)[1] <= slope or gauss_roc(right, tol)[1] >= slope:
        return max_end, end_proj, end_slope
    while not math.isclose(left, right, abs_tol=tol):
        middle = (left + right) / 2
        if gauss_roc(middle, tol)[1] > slope:
            left = middle
        else:
            right = middle
    alpha = (sum(end) - left - gauss_roc(left, tol)[0]) / (sum(end) - sum(begin))
    max_disp = alpha * begin[0] + (1 - alpha) * end[0] - left
    return max(max_end, max_disp), end_proj, end_slope


def proj(
    point: np.ndarray, tol: float = 1e-9, bounds: tuple[float, float] = (0, 1)
) -> tuple[np.ndarray, float]:
    """Find projection on true ROC.

    Args:
        point: A point in [0, 1]^2.
        tol: Tolerance.
        bounds: Bounds of projection to help with the calculation.

    Returns:
        Projection on the true ROC and the slope there.
    """
    left, right = bounds
    roc_left = gauss_roc(left, tol)
    if left + roc_left[0] >= sum(point):
        return np.array([left, roc_left[0]]), roc_left[1]
    roc_right = gauss_roc(right, tol)
    if right + roc_right[0] <= sum(point):
        return np.array([right, roc_right[0]]), roc_right[1]
    while not math.isclose(left, right, abs_tol=tol):
        middle = (left + right) / 2
        if middle + gauss_roc(middle, tol)[0] > sum(point):
            right = middle
        else:
            left = middle
    if left == bounds[0]:
        return np.array([left, roc_left[0]]), roc_left[1]
    if right == bounds[1]:
        return np.array([right, roc_right[0]]), roc_right[1]
    roc_middle = gauss_roc(left, tol)
    return np.array([left, roc_middle[0]]), roc_middle[1]


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
    print(gauss.levy_multiple([0, 100], 500, 6))


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
    print(gauss.levy_single([0, 100], 6, "6"))


def plot_avg_levy():
    """Plots average Levy metrics."""
    samples = [[10, 10], [100, 100], [1000, 1000], [10, 100], [10, 1000], [100, 1000]]
    levy_all = []
    for n_samps in samples:
        with open(
            f"{PATH}levy-all-{n_samps[0]}-{n_samps[1]}-n500-mo{False}.json",
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
            list(range(3)),
            this_avg,
            yerr=error[i],
            color=["C1", "C2", "C3"],
            capsize=10,
        )
        plt.ylim(0, 0.16)
        plt.xticks(list(range(3)), ["MLE", "E", "CE"])
        plt.xlabel("estimator")
        plt.ylabel("average Lévy distance")
        plt.tight_layout()
        plt.savefig(f"{PATH}levy-{samples[i][0]}-{samples[i][1]}.pdf")


def plot_avg_levy_mle_only():
    """Plots average Levy metric for MLE."""
    with open(f"{PATH}levy-all-0-100-n500-mo{True}.json", encoding="utf-8") as f:
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
    plt.savefig(f"{PATH}levy-0-100.pdf")


def zigzag(func: np.ndarray) -> np.ndarray:
    """Get zigzag piecewise linear function."""
    new_func = [list(func[:, 0])]
    for idx, point in enumerate(func[:, :-1].T):
        new_func.append([func[0, idx + 1], point[1]])
        new_func.append(list(func[:, idx + 1]))
    return np.array(new_func).T


def speed_test():
    """Compares running times of the two Levy distance methods."""
    n_samp = [0, 2]
    rng = np.random.default_rng(0)
    null = np.exp(rng.normal(size=n_samp[0]) - 1 / 2)
    alt = np.exp(rng.normal(loc=1, size=n_samp[1]) - 1 / 2)
    val = np.append(null, alt)
    count = np.concatenate(([0], np.ones(sum(n_samp)), [0]))
    pmf, _ = mle(val, count)
    roc_est = get_roc(np.insert(val, 0, 0), pmf, 1e-6)
    print("Samples: ", n_samp)
    tol = 1e-2
    dist = levy(roc_est, True, tol)
    print(f"Calculated from estimated ROC (tol = {tol}): ", dist)
    tol = 1e-4
    dist = levy(roc_est, True, tol)
    print(f"Calculated from estimated ROC (tol = {tol}): ", dist)
    tol = 1e-6
    dist = levy(roc_est, True, tol)
    print(f"Calculated from estimated ROC (tol = {tol}): ", dist)
    digits = 2
    gauss = GaussSim(digits)
    dist_alt = gauss.levy(roc_est)
    print(f"Calculated from true ROC (digits = {digits}): ", dist_alt)
    digits = 4
    gauss = GaussSim(digits)
    dist_alt = gauss.levy(roc_est)
    print(f"Calculated from true ROC (digits = {digits}): ", dist_alt)
    digits = 6
    gauss = GaussSim(digits)
    dist_alt = gauss.levy(roc_est)
    print(f"Calculated from true ROC (digits = {digits}): ", dist_alt)


def plot_bc_v_true_for_binormal():
    """Plots BC bound against true ROC for binormal BHT.

    This generates Fig. 16 in the causal network paper (commit 408fe8d).
    """
    GaussROC([0, 1], [1, 1]).plot(f"{PATH}binormal-bc-v-true-roc.pdf")


def error_v_diff(from_file: bool):
    """Evaluates errors of MLE with respect to difference in hypotheses."""
    diff_list = [0.1] + np.linspace(0.5, 4, 8).tolist()
    dist = []
    for diff in diff_list:
        gauss = GaussSim(mean_diff=diff)
        dist.append(gauss.levy_multiple([100, 100], 500, from_file=from_file))
    dist = np.array(dist)
    plt.figure()
    line_width = 3
    plt.plot(diff_list, dist[:, 0], "-o", label="MLE", linewidth=line_width, color="C1")
    plt.plot(diff_list, dist[:, 1], "--o", label="E", linewidth=line_width, color="C2")
    plt.plot(diff_list, dist[:, 2], "-.o", label="CE", linewidth=line_width, color="C3")
    plt.legend()
    plt.xlabel(r"mean difference $\mu$")
    plt.ylabel("average Lévy distance")
    plt.tight_layout()
    plt.savefig(f"{PATH}levy-v-diff-delta-mu.pdf")


def error_v_composition(from_file: bool):
    """Evaluates errors with different sample composition."""
    total_samp = 100
    n_trials = 500
    gauss_mle = GaussSim(mle_only=True)
    gauss = GaussSim()
    error = [gauss_mle.levy_multiple([0, total_samp], n_trials, from_file=from_file)]
    for alpha in np.linspace(0.1, 0.9, 9):
        error.append(
            gauss.levy_multiple(
                [int(total_samp * alpha), int(total_samp * (1 - alpha))],
                n_trials,
                from_file=from_file,
            )
        )
    error.append(
        gauss_mle.levy_multiple([total_samp, 0], n_trials, from_file=from_file)
    )
    error = np.array(error)
    plt.figure()
    line_width = 3
    plt.plot(
        np.linspace(0, 1, 11),
        error[:, 0],
        "-o",
        label="MLE",
        linewidth=line_width,
        color="C1",
    )
    plt.plot(
        np.linspace(0.1, 0.9, 9),
        error[1:-1, 1],
        "--o",
        label="E",
        linewidth=line_width,
        color="C2",
    )
    plt.plot(
        np.linspace(0.1, 0.9, 9),
        error[1:-1, 2],
        "-.o",
        label="CE",
        linewidth=line_width,
        color="C3",
    )
    plt.legend()
    plt.xlabel(r"fraction of null samples $\alpha$")
    plt.ylabel("average Lévy distance")
    plt.tight_layout()
    plt.savefig(f"{PATH}levy-v-alpha.pdf")


if __name__ == "__main__":
    # calc_avg_levy()
    # gen_roc_examples()
    # plot_avg_levy()
    # plot_avg_levy_mle_only()
    # error_v_diff(True)
    error_v_composition(True)
