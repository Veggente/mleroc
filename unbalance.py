"""MLE for likelihoods."""  # pylint: disable=too-many-lines
import math
import json
from dataclasses import dataclass
from typing import Optional, Union
import bisect
import itertools
import numpy as np
from scipy.optimize import fsolve
from scipy.stats._distn_infrastructure import rv_sample
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from bhatta_bound import bhatta_bound
from config import PATH


@dataclass
class ROC:
    """Receiver operating characteristic curve.

    Linear interpolation is done for two adjacent points.  The ROC is
    not necessarily optimal, i.e., it may not be concave.  Note that
    some methods assume concavity.

    Attributes:
        pfa: np.ndarray
            Probability of false alarm.  Must be nondecreasing (but
            could have repeating 0's to allow vertical jumps in ROC at
            0 fa).
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

    def plot(self, name: str = ""):
        """Plots ROC curve."""
        plt.figure()
        plt.plot(
            self.pfa,
            self.pdet,
        )
        plt.xlabel("prob. of false alarm")
        plt.ylabel("prob. of detection")
        plt.tight_layout()
        if name:
            plt.savefig(name)
        else:
            plt.show()

    def get_pdet(self, pfa: float) -> float:
        """Gets pdet by interpolation."""
        return np.interp(pfa, self.pfa, self.pdet)

    def like_dist(self, null: bool) -> np.ndarray:
        """Likelihood distribution.

        Assumes the representation is minimal in the sense that no
        points can be removed.  Also assumes the slopes are bounded.
        """
        slopes = []
        pmf = []
        for i, false_alarm in enumerate(self.pfa[:-1]):
            pfa_inc = self.pfa[i + 1] - false_alarm
            pdet_inc = self.pdet[i + 1] - self.pdet[i]
            slopes.append(pdet_inc / pfa_inc)
            pmf.append(pfa_inc if null else pdet_inc)
        return np.array([slopes[::-1], pmf[::-1]])

    def like_dist_mix(self, alpha: float) -> np.ndarray:
        """Likelihood distribution mixture."""
        null = self.like_dist(True)
        alt = self.like_dist(False)
        return alpha * null + (1 - alpha) * alt

    def offset(self, pfa: float) -> float:
        """Offset of a linear piece."""
        left = bisect.bisect_left(self.pfa, pfa) - 1
        return (
            self.pdet[left]
            - (self.pdet[left + 1] - self.pdet[left])
            / (self.pfa[left + 1] - self.pfa[left])
            * self.pfa[left]
        )

    def gen_mix_like(
        self, alpha: float, size: int, random_state: np.random.Generator
    ) -> np.ndarray:
        """Generates likelihood ratios from the mixed distribution.

        Args:
            alpha: Fraction of null samples.
            size: Number of samples.
            random_state: Random number generator.

        Returns:
            Likelihood ratio samples.
        """
        dist = DiscDist(self.like_dist_mix(alpha))
        return dist.rvs(size=size, random_state=random_state)


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

    def levy_single(  # pylint: disable=too-many-locals
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
        Pmf of F_0 and the Lagrangian dual variable lambda.
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


def levy_linear(  # pylint: disable=too-many-locals
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
    plt.plot(
        diff_list,
        dist[:, 0],
        "-x",
        label="MLE",
        linewidth=line_width,
        color="C1",
        markersize=12,
        fillstyle="none",
        markeredgewidth="2",
    )
    plt.plot(
        diff_list,
        dist[:, 1],
        "--o",
        label="E",
        linewidth=line_width,
        color="C2",
        markersize=12,
        fillstyle="none",
        markeredgewidth="2",
    )
    plt.plot(
        diff_list,
        dist[:, 2],
        "-.^",
        label="CE",
        linewidth=line_width,
        color="C3",
        markersize=12,
        fillstyle="none",
        markeredgewidth="2",
    )
    plt.legend()
    plt.xlabel(r"mean difference $\mu$")
    plt.ylabel("average Lévy distance")
    plt.tight_layout()
    plt.savefig(f"{PATH}levy-v-diff-delta-mu.pdf")


def error_v_composition(from_file: bool):
    """Evaluates errors with different sample composition."""
    total_samp = 200
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
        "-x",
        label="MLE",
        linewidth=line_width,
        color="C1",
        markersize=12,
        fillstyle="none",
        markeredgewidth="2",
    )
    plt.plot(
        np.linspace(0.1, 0.9, 9),
        error[1:-1, 1],
        "--o",
        label="E",
        linewidth=line_width,
        color="C2",
        markersize=12,
        fillstyle="none",
        markeredgewidth="2",
    )
    plt.plot(
        np.linspace(0.1, 0.9, 9),
        error[1:-1, 2],
        "-.^",
        label="CE",
        linewidth=line_width,
        color="C3",
        markersize=12,
        fillstyle="none",
        markeredgewidth="2",
    )
    plt.legend()
    plt.xlabel(r"fraction of null samples $\alpha$")
    plt.ylabel("average Lévy distance")
    plt.ylim(0, 0.11)
    plt.tight_layout()
    plt.savefig(f"{PATH}levy-v-alpha.pdf")


def isit_example():
    """ISIT presentation example."""
    likelihoods = [0.9, 2.2, 1.5, 0.6, 0.2, 2.5, 1.8]
    print(mle(np.array(likelihoods), np.array([0] + [1] * 7 + [0])))


# def mle_convergence_3_slope_values():
#     """MLE convergence study for 3 slope values."""
#     infty = 1000
#     p_null_0 = 0.6
#     slope = 2
#     small_pfa = (1 - slope * (1 - p_null_0)) / (infty - slope)
#     true_roc = ROC(
#         np.array([0, small_pfa, 1 - p_null_0, 1]),
#         np.array([0, small_pfa * infty, 1, 1]),
#         "true",
#     )
#     alpha = 0.3
#     n_samp = 100
#     mix = true_roc.like_dist_mix(alpha)
#     rng = np.random.default_rng(0)
#     samples = rv_discrete(values=(mix[0], mix[1])).rvs(size=n_samp, random_state=rng)
#     est = MLE(list(samples))
#     est_roc = est.roc()
#     print(est_roc.pdet)
#     print(f"{est_roc.offset(0.2) = }", f"{true_roc.offset(0.2) = }")
#     true_roc.plot()
#     est_roc.plot()


def mle_convergence_3_slope_values_gen():
    """MLE convergence sutdy with 3 general slope values."""
    slope = [0.3, 1]
    p_null = [0.5, 0.3]
    pfa = np.array([1, 1 - p_null[0], 1 - sum(p_null), 0])[::-1]
    pdet = np.array([1, 1 - slope[0] * p_null[0], 1 - np.inner(slope, p_null), 0])[::-1]
    true_roc = ROC(pfa, pdet, "true")
    alpha = 0.3
    n_samp = 10000
    rng = np.random.default_rng()
    dist = DiscDist(true_roc.like_dist_mix(alpha))
    n_sims = 10000
    errors = []
    mid_point = 1 - p_null[0] - p_null[1] / 2
    true_offset = true_roc.offset(mid_point)
    for _ in tqdm(range(n_sims)):
        est_roc = MLE(list(dist.rvs(size=n_samp, random_state=rng))).roc()
        errors.append(est_roc.offset(mid_point) - true_offset)
    plt.figure()
    plt.hist(errors, bins=50)
    plt.show()


def mle_convergence():
    """MLE convergence study with general slopes."""
    slope = [0.3, 1]
    p_null = [0.5, 0.3]
    true_roc = OROC(slope, p_null, "true")
    alpha = 0.3
    n_samp = 1000
    rng = np.random.default_rng()
    errors = []
    mid_point = 1 - p_null[0] - p_null[1] / 2
    true_offset = true_roc.offset(mid_point)
    for _ in tqdm(range(1000)):
        est_roc = MLE(list(true_roc.gen_mix_like(alpha, n_samp, rng))).roc()
        errors.append(est_roc.offset(mid_point) - true_offset)
    plt.figure()
    plt.hist(errors, bins=50)
    plt.show()


class DiscDist(rv_sample):
    """Discrete distribution on float numbers."""

    def __init__(self, flt_values: np.ndarray):
        super().__init__(values=(list(range(flt_values.shape[1])), list(flt_values[1])))
        self.val = flt_values[0]

    def rvs(
        self,
        *args,
        size: Optional[int] = None,
        random_state: Optional[np.random.Generator] = None,
        **kwds,  # pylint: disable=unused-argument
    ) -> np.ndarray:
        """Generates i.i.d. random variables."""
        indices = super().rvs(size=size, random_state=random_state)
        return self.val[indices]


def mle_convergence_4_val():
    """MLE convergence study with 4 slopes."""
    slope = [0.1, 0.4, 1]
    p_null = [0.4, 0.3, 0.2]
    true_roc = OROC(slope, p_null, "true")
    alpha = 0.3
    n_samp = 10000
    rng = np.random.default_rng()
    errors = []
    mid_points = [1 - p_null[0] - p_null[1] / 2, 1 - sum(p_null[:2]) - p_null[2] / 2]
    true_offsets = np.array([true_roc.offset(x) for x in mid_points])
    n_sim = 10000
    for _ in tqdm(range(n_sim)):
        est_roc = MLE(list(true_roc.gen_mix_like(alpha, n_samp, rng))).roc()
        est_offsets = np.array([est_roc.offset(x) for x in mid_points])
        errors.append(est_offsets - true_offsets)
    errors = np.array(errors)
    print(f"{ols(errors) = }")
    plt.figure()
    plt.hist2d(errors[:, 0], errors[:, 1], bins=50, range=get_range(n_samp))
    plt.savefig(
        f"/Users/veggente/Data/research/flowering/soybean-rna-seq-data/mleroc/mle-error-sl4-n{n_samp}-m{n_sim}.pdf"  # pylint: disable=line-too-long
    )


def get_range(n_samp: int) -> np.ndarray:
    """Gets range for asymptotic convergence."""
    left = -0.005 * 50 / 32
    right = 0.005 * 50 / 32
    top = 0.01 * 48.5 / 37
    bottom = -0.01 * 50.5 / 37
    return np.array([[left, right], [bottom, top]]) / np.sqrt(n_samp / 10000)


def ols(data: np.ndarray) -> tuple[np.ndarray, float]:
    """Ordinary least squares.

    Args:
        data: n-by-2 matrix with x in the first column and y in the second column.

    Returns:
        OLS slope and intercept, and OLS slope without intercept.
    """
    x_bar = np.mean(data[:, 0])
    y_bar = np.mean(data[:, 1])
    xy_bar = np.mean(data[:, 0] * data[:, 1])
    x2_bar = np.mean(data[:, 0] ** 2)
    return (
        np.array([xy_bar - x_bar * y_bar, x2_bar * y_bar - xy_bar * x_bar])
        / (x2_bar - x_bar**2),
        xy_bar / x2_bar,
    )


def lipschitz():
    """Lipschitz constant study for MLE."""
    rng = np.random.default_rng()
    n_sim = 100000
    for i in range(n_sim):
        slope = [rng.random()]
        slope.append(slope[0] + (2 - slope[0]) * rng.random())
        p_null = [get_p_null(slope, rng) for _ in range(2)]
        alpha = rng.random(2)
        oroc = [OROC(slope, p_null[i], f"true{i}") for i in range(2)]
        levy_dist = oroc[0].levy(oroc[1])
        sup_norm = oroc[0].sup_norm(oroc[1], alpha)
        if levy_dist > 2.5 * sup_norm:
            print(f"An extreme counterexample is found at {i} / {n_sim}:")
            print(f"{slope = }, {p_null = }, {alpha = }, {levy_dist = }, {sup_norm = }")
            oroc[0].plot("oroc5-1.pdf")
            oroc[1].plot("oroc5-2.pdf")
            return
    print("No counterexamples are found.")
    return


def get_p_null(slope: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gets p_null from slope.

    Assumes there are only two slope values and the first slope is
    less than 1, and the second slope is greater than the first slope.
    """
    uni = rng.random(len(slope) - 1)
    remain_fa = remain_det = 1
    pmf = []
    for i, x in enumerate(uni):
        lower = max(
            0, (slope[i + 1] * remain_fa - remain_det) / (slope[i + 1] - slope[i])
        )
        pmf.append(lower + x * (remain_fa - lower))
        remain_fa -= pmf[-1]
        remain_det -= pmf[-1] * slope[i]
    pmf.append(remain_fa)
    return np.array(pmf)


class OROC(ROC):
    """Optimal ROC curve.

    Attributes:
        slopes: Increasing slopes of the ROC curve.
        null_prob: Probability masses under the null hypothesis.
    """

    def __init__(self, slopes: list[float], null_prob: list[float], name: str):
        """Initializes the OROC curve.

        Assumes the inner product of slopes and null_prob is less than
        or equal to 1.

        Args:
            slopes: Increasing slopes of the ROC curve.  Exludes
                infinity.
            null_prob: Probability masses under the null hypothesis.
                Sums to 1.
        """
        self.slopes = slopes
        self.null_prob = null_prob
        pfa, pdet = [1], [1]
        for i, slo in enumerate(slopes):
            pfa.append(pfa[-1] - null_prob[i])
            pdet.append(pdet[-1] - null_prob[i] * slo)
            if pdet[-1] < pfa[-1] - 1e-6:
                print(f"{pdet = }, {pfa = }")
                raise ValueError("Slopes and probabilities do not form a valid ROC.")
        if pdet[-1]:
            pfa.append(0)
            pdet.append(0)
        super().__init__(pfa[::-1], pdet[::-1], name)

    def levy(self, other: "OROC") -> float:
        """Levy distance between two ROC curves.

        Assumes both OROC curves have the same set of slopes.  Note
        the last intercept does not need to be checked.
        """
        max_dist = 0
        for i, slope_i in enumerate(self.slopes[1:], start=1):
            intercept = 0
            for j, slope_j in enumerate(self.slopes[:i]):
                intercept += (slope_i - slope_j) * (
                    self.null_prob[j] - other.null_prob[j]
                )
            max_dist = max(max_dist, abs(intercept) / (slope_i + 1))
        return max_dist

    def sup_norm(self, other: "OROC", alpha: np.ndarray) -> float:
        """Supremum norm between two ROC curves.

        Assumes both OROC curves have the same set of slopes.
        """
        max_dist = 0
        for i in range(1, len(self.slopes) + 1):
            diff = 0
            for j, slope_j in enumerate(self.slopes[:i]):
                diff += (1 - alpha[0] + alpha[0] * slope_j) * self.null_prob[j] - (
                    1 - alpha[1] + alpha[1] * slope_j
                ) * other.null_prob[j]
            max_dist = max(max_dist, abs(diff))
        return max_dist


def counterexample():
    """Counterexample for Lipschitz continuity."""
    slopes = [0.5, 1.7]
    p_null = [[0.6, 0.4], [0.65, 0.35]]
    oroc1 = OROC(slopes, p_null[0], "oroc1")
    oroc2 = OROC(slopes, p_null[1], "oroc2")
    plt.figure()
    plt.plot(oroc1.pfa, oroc1.pdet, label="oroc1")
    plt.plot(oroc2.pfa, oroc2.pdet, label="oroc2")
    plt.legend()
    plt.savefig("oroc.pdf")
    print(f"{oroc1.levy(oroc2) = }")
    sup_norm = []
    for alpha in itertools.product(np.linspace(0, 1, 1001), repeat=2):
        sup_norm.append(oroc1.sup_norm(oroc2, alpha))
    loc = np.argmin(sup_norm)
    print(f"{min(sup_norm) = } at ({(loc // 1001) / 1000}, {(loc % 1001) / 1000})")
    plt.figure()
    image = plt.imshow(
        np.array(sup_norm).reshape(1001, 1001).T,
        origin="lower",
        cmap=plt.cm.RdBu,  # pylint: disable=no-member
        extent=[0, 1, 0, 1],
    )
    plt.colorbar(image)
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\lambda'$")
    plt.tight_layout()
    plt.savefig("sup_norm.pdf")


def amle(
    val: np.ndarray, count: np.ndarray, mix: float
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate MLE anchored at upper-right corner.

    Args:
        val: Likelihood values (excluding zero and infinity).  Must be
            strictly increasing.
        count: Counts for likelihood values (including zero and
            infinity).  Only the first and the last counts can be
            zero.
        mix: Mixing parameter as the fraction of H1 samples.  Must be
            in [0, 1[.

    Returns:
        Values and pmf of F_0 after truncation or concavifying.
        Values are positive, finite, and strictly increasing, and pmf
        includes the mass for 0 at the beginning.
    """
    pmf = count / sum(count)
    ex_val = np.insert(val, 0, 0)
    pmf0 = []
    f0_val = []
    f0_cumu = f1_cumu = 0
    for i, slope in enumerate(ex_val):
        shingle = [pmf[i] / (1 - mix + mix * slope)]
        shingle.append(shingle[0] * slope)
        if slope * (1 - f0_cumu) > 1 - f1_cumu:
            pmf0.append(1 - f0_cumu)
            f0_val.append((1 - f1_cumu) / (1 - f0_cumu))
            break
        if f0_cumu + shingle[0] >= 1:
            pmf0.append(1 - f0_cumu)
            f0_val.append(slope)
            break
        pmf0.append(shingle[0])
        f0_val.append(slope)
        f0_cumu += shingle[0]
        f1_cumu += shingle[1]
    return np.array(f0_val[1:]), np.array(pmf0)


if __name__ == "__main__":
    plt.style.use("ggplot")
    plt.rcParams.update({"font.size": 20, "pdf.fonttype": 42})
    plt.rcParams.update({"font.size": 20})
    gen_roc_examples()
    calc_avg_levy()
    plot_avg_levy()
