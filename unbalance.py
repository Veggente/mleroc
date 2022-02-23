"""MLE for likelihoods."""
import math
import json
from dataclasses import dataclass
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.style.use("ggplot")
plt.rcParams.update({"font.size": 20, "pdf.fonttype": 42})
PATH = ""


@dataclass
class ROC:
    """Receiver operating characteristic curve."""

    pfa: np.ndarray
    pdet: np.ndarray

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


class MLE:
    """ML estimator."""

    def __init__(self, likelihoods: list[float]):
        """Initialization."""
        self.likelihoods = likelihoods

    def roc(self) -> ROC:
        """ML estimator of the ROC."""
        pmf = mle(np.array(self.likelihoods), np.array([0] + [1] * len(self.likelihoods) + [0]))
        roc = get_roc(np.insert(self.likelihoods, 0, 0), pmf[0], 1e-6)
        return ROC(roc[0, :], roc[1, :])


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


def gauss_sim(
    n_samp: list[int],
    rng: np.random.Generator,
    suffix: str = "",
    digits: int = 4,
    mle_only: bool = False,
) -> list[float]:
    """Gaussian hypothesis simulations.

    Args:
        n_samp: Numbers of likelihood ratio samples under H0 and H1.
        rng: Random number generator.
        suffix: Suffix for saving ROC curve plot.
        digits: Number of digits for tolerance.
        mle_only: Do MLE only.

    Returns:
        Levy metrics from MLE, empirical and CE to the true ROC.
    """
    tol = 10 ** -digits
    null = np.exp(rng.normal(size=n_samp[0]) - 1 / 2)
    alt = np.exp(rng.normal(loc=1, size=n_samp[1]) - 1 / 2)
    val = np.append(null, alt)
    count = np.concatenate(([0], np.ones(sum(n_samp)), [0]))
    pmf, _ = mle(val, count)
    pfa = np.linspace(0, 1, 101)
    roc_est = get_roc(np.insert(val, 0, 0), pmf, tol)
    dist_alt = [levy_alt(roc_est, digits)]
    line_width = 3
    if suffix:
        plt.figure()
        plt.plot(
            pfa,
            [gauss_roc(p, tol)[0] for p in pfa],
            ":",
            label="true",
            linewidth=line_width,
        )
        plt.plot(roc_est[0, :], roc_est[1, :], "-", label="MLE", linewidth=line_width)
    if n_samp[0] and n_samp[1] and not mle_only:
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
        dist_alt.append(levy_alt(zigzag(s_roc), digits))
        dist_alt.append(levy_alt(cs_roc, digits))
    else:
        dist_alt.append(-1)
        dist_alt.append(-1)
    if suffix:
        plt.legend()
        plt.xlabel("prob. of false alarm")
        plt.ylabel("prob. of detection")
        plt.tight_layout()
        plt.savefig("{}roc-{}-{}-{}.pdf".format(PATH, *n_samp, suffix))
    return dist_alt


def gauss_roc(pfa: float, tol: float = 1e-9) -> np.ndarray:
    """Gaussian hypothesis true ROC curve.

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


def avg_levy(seed: int, n_samp: list[int], n_sims: int, mle_only: bool) -> np.ndarray:
    """Average Levy metrics.

    Args:
        seed: RNG seed.
        n_samp: Numbers of samples.
        n_sims: Number of simulations.
        mle_only: Do MLE only.

    Returns:
        Average Levy metrics.
    """
    rng = np.random.default_rng(seed)
    dist = np.zeros(3)
    dist_full = []
    for _ in tqdm(range(n_sims)):
        new_dist = gauss_sim(n_samp, rng, mle_only=mle_only)
        dist += np.array(new_dist)
        dist_full.append(new_dist)
    with open(
        "{}levy-all-{}-{}-n{}-mo{}.json".format(PATH, *n_samp, n_sims, mle_only),
        "w",
    ) as f:
        json.dump(dist_full, f)
    return dist / n_sims


def calc_avg_levy():
    """Calculates average Levy metrics."""
    print(avg_levy(0, [10, 10], 500, False))
    print(avg_levy(1, [100, 100], 500, False))
    print(avg_levy(2, [1000, 1000], 500, False))
    print(avg_levy(3, [10, 100], 500, False))
    print(avg_levy(4, [10, 1000], 500, False))
    print(avg_levy(5, [100, 1000], 500, False))
    print(avg_levy(6, [0, 100], 500, True))


def gen_roc_examples():
    """Generates ROC examples."""
    rng = np.random.default_rng(0)
    print(gauss_sim([10, 10], rng, "0"))
    rng = np.random.default_rng(1)
    print(gauss_sim([100, 100], rng, "1"))
    rng = np.random.default_rng(2)
    print(gauss_sim([1000, 1000], rng, "2"))
    rng = np.random.default_rng(3)
    print(gauss_sim([10, 100], rng, "3"))
    rng = np.random.default_rng(4)
    print(gauss_sim([10, 1000], rng, "4"))
    rng = np.random.default_rng(5)
    print(gauss_sim([100, 1000], rng, "5"))
    rng = np.random.default_rng(6)
    print(gauss_sim([0, 100], rng, "6", mle_only=True))


def plot_avg_levy():
    """Plots average Levy metrics."""
    samples = [[10, 10], [100, 100], [1000, 1000], [10, 100], [10, 1000], [100, 1000]]
    levy_all = []
    for n_samps in samples:
        with open("{}levy-all-{}-{}-n500-mo{}.json".format(PATH, *n_samps, False)) as f:
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
        plt.savefig("{}levy-{}-{}.pdf".format(PATH, *samples[i]))


def plot_avg_levy_mle_only():
    """Plots average Levy metric for MLE."""
    with open("{}levy-all-0-100-n500-mo{}.json".format(PATH, True)) as f:
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
    plt.savefig("{}levy-0-100.pdf".format(PATH))


def levy_alt(piecewise: np.ndarray, digits: int) -> float:
    """Lévy metric w.r.t. the Gaussian ROC curve.

    Calculates ROC at discrete values first, and then projects to the
    ROC estimate.

    Args:
        piecewise: The points that define the piecewise nondecreasing
            function.  Each column is a point.
        digits: Number of digits for ROC calculation.

    Returns:
        The Lévy metric from piecewise to the true Gaussian ROC.
    """
    tol = 10 ** -digits
    if sum(piecewise[:, 0]) > tol:
        piecewise = np.insert(piecewise, 0, [0, 0], axis=1)
    if sum(piecewise[:, -1]) < 2 - tol:
        piecewise = np.append(piecewise, np.zeros((2, 1)), axis=1)
    pfa = np.linspace(0, 1, 10 ** digits + 1)
    roc = 1 - norm.cdf(norm.ppf(1 - pfa) - 1)
    begin = 0
    max_dist = 0
    for false_alarm, detection in zip(pfa, roc):
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
    print("Calculated from estimated ROC (tol = {}): ".format(tol), dist)
    tol = 1e-4
    dist = levy(roc_est, True, tol)
    print("Calculated from estimated ROC (tol = {}): ".format(tol), dist)
    tol = 1e-6
    dist = levy(roc_est, True, tol)
    print("Calculated from estimated ROC (tol = {}): ".format(tol), dist)
    digits = 2
    dist_alt = levy_alt(roc_est, digits)
    print("Calculated from true ROC (digits = {}): ".format(digits), dist_alt)
    digits = 4
    dist_alt = levy_alt(roc_est, digits)
    print("Calculated from true ROC (digits = {}): ".format(digits), dist_alt)
    digits = 6
    dist_alt = levy_alt(roc_est, digits)
    print("Calculated from true ROC (digits = {}): ".format(digits), dist_alt)


if __name__ == "__main__":
    calc_avg_levy()
    gen_roc_examples()
    plot_avg_levy()
    plot_avg_levy_mle_only()
