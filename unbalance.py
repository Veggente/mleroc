"""MLE for likelihoods."""
import math
import json
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.style.use("ggplot")
plt.rcParams.update({"font.size": 20})


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
    tol: float = 1e-9,
    mle_only: bool = False,
) -> list[float]:
    """Gaussian hypothesis simulations.

    Args:
        n_samp: Numbers of likelihood ratio samples under H0 and H1.
        rng: Random number generator.
        suffix: Suffix for saving ROC curve plot.
        mle_only: Do MLE only.

    Returns:
        Levy metrics from MLE, empirical and CE to the true ROC.
    """
    null = np.exp(rng.normal(size=n_samp[0]) - 1 / 2)
    alt = np.exp(rng.normal(loc=1, size=n_samp[1]) - 1 / 2)
    val = np.append(null, alt)
    count = np.concatenate(([0], np.ones(sum(n_samp)), [0]))
    pmf, _ = mle(val, count)
    pfa = np.linspace(0, 1, 101)
    roc_est = get_roc(val, pmf)
    dist = [levy(roc_est, True, tol)]
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
        dist.append(levy(s_roc, False, tol))
        dist.append(levy(cs_roc, True, tol))
    else:
        dist.append(-1)
        dist.append(-1)
    if suffix:
        plt.legend()
        plt.xlabel("prob. of false alarm")
        plt.ylabel("prob. of detection")
        plt.tight_layout()
        plt.savefig(
            "/Users/veggente/Data/research/flowering/soybean-rna-seq-data/mleroc/roc-{}-{}-{}.pdf".format(  # pylint: disable=line-too-long
                *n_samp, suffix
            )
        )
    return dist


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


def get_roc(val: np.ndarray, pmf: np.ndarray) -> np.ndarray:
    """Gets ROC from F0 pmf.

    Args:
        val: Values of pmf (excluding 0).  Length is n - 1.
        pmf: Masses of pmf (including 0).  Length is n.

    Returns:
        A 2xn array of points on the ROC curve.
    """
    sorted_pmf = np.array([np.append([0], val), pmf])
    sorted_pmf = sorted_pmf[:, np.argsort(sorted_pmf[0, :])]
    points = [[1, 1]]
    for slope, h_dist in sorted_pmf.T:
        if not h_dist:
            continue
        pfa, pdet = points[-1]
        points.append([pfa - h_dist, pdet - h_dist * slope])
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
    roc = gauss_roc(left, tol)
    if left + roc[0] >= sum(point):
        return np.array([left, roc[0]]), roc[1]
    roc = gauss_roc(right, tol)
    if right + roc[0] <= sum(point):
        return np.array([right, roc[0]]), roc[1]
    while not math.isclose(left, right, abs_tol=tol):
        middle = (left + right) / 2
        if middle + gauss_roc(middle, tol)[0] > sum(point):
            right = middle
        else:
            left = middle
    roc = gauss_roc(left, tol)
    return np.array([left, roc[0]]), roc[1]


def avg_levy(
    seed: int, n_samp: list[int], n_sims: int, mle_only: bool, tol: float
) -> np.ndarray:
    """Average Levy metrics.

    Args:
        seed: RNG seed.
        n_samp: Numbers of samples.
        n_sims: Number of simulations.
        mle_only: Do MLE only.
        tol: Tolerance.

    Returns:
        Average Levy metrics.
    """
    rng = np.random.default_rng(seed)
    dist = np.zeros(3)
    dist_full = []
    for _ in tqdm(range(n_sims)):
        new_dist = gauss_sim(n_samp, rng, tol=tol, mle_only=mle_only)
        dist += np.array(new_dist)
        dist_full.append(new_dist)
    with open(
        "/Users/veggente/Data/research/flowering/soybean-rna-seq-data/mleroc/levy-all-{}-{}-n{}-mo{}.json".format(  # pylint: disable=line-too-long
            *n_samp, n_sims, mle_only
        ),
        "w",
    ) as f:
        json.dump(dist_full, f)
    return dist / n_sims


def calc_avg_levy():
    """Calculates average Levy metrics."""
    print(avg_levy(0, [10, 10], 500, False, 1e-4))
    print(avg_levy(1, [100, 100], 500, False, 1e-4))
    print(avg_levy(2, [1000, 1000], 500, False, 1e-4))
    print(avg_levy(3, [10, 100], 500, False, 1e-4))
    print(avg_levy(4, [10, 1000], 500, False, 1e-4))
    print(avg_levy(5, [100, 1000], 500, False, 1e-4))


def gen_roc_examples():
    """Generates ROC examples."""
    # rng = np.random.default_rng(0)
    # print(gauss_sim([10, 10], rng, "0", tol=1e-4))
    # rng = np.random.default_rng(1)
    # print(gauss_sim([100, 100], rng, "1", tol=1e-4))
    # rng = np.random.default_rng(2)
    # print(gauss_sim([1000, 1000], rng, "2", tol=1e-4))
    # rng = np.random.default_rng(3)
    # print(gauss_sim([10, 100], rng, "3", tol=1e-4))
    # rng = np.random.default_rng(4)
    # print(gauss_sim([10, 1000], rng, "4", tol=1e-4))
    # rng = np.random.default_rng(5)
    # print(gauss_sim([100, 1000], rng, "5", tol=1e-4))
    rng = np.random.default_rng(6)
    print(gauss_sim([0, 100], rng, "6", tol=1e-4))


def plot_avg_levy():
    """Plots average Levy metrics."""
    samples = [[10, 10], [100, 100], [1000, 1000], [10, 100], [10, 1000], [100, 1000]]
    levy_all = []
    for n_samps in samples:
        with open(
            "/Users/veggente/Data/research/flowering/soybean-rna-seq-data/mleroc/levy-all-{}-{}-n500.json".format(  # pylint: disable=line-too-long
                *n_samps
            )
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
        plt.savefig(
            "/Users/veggente/Data/research/flowering/soybean-rna-seq-data/mleroc/levy-{}-{}.pdf".format(  # pylint: disable=line-too-long
                *samples[i]
            )
        )


def plot_avg_levy_mle_only():
    """Plots average Levy metric for MLE."""
    with open(
        "/Users/veggente/Data/research/flowering/soybean-rna-seq-data/mleroc/levy-all-0-100-n500.json"  # pylint: disable=line-too-long
    ) as f:
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
    plt.savefig(
        "/Users/veggente/Data/research/flowering/soybean-rna-seq-data/mleroc/levy-0-100.pdf"  # pylint: disable=line-too-long
    )


def check_convergence():
    """Checks convergence rate of MLE."""
    print("[50, 150]:", avg_levy(0, [50, 150], 100, True, 1e-4))
    print("[500, 1500]:", avg_levy(0, [500, 1500], 100, True, 1e-4))
    print("[0, 200]:", avg_levy(0, [0, 200], 100, True, 1e-4))
    print("[0, 2000]:", avg_levy(0, [0, 2000], 100, True, 1e-4))
