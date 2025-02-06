"""Estimators for the ROC curve."""

from collections import Counter

import numpy as np
from scipy.optimize import fsolve

from .roc import ROC
from .roc import get_roc


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


def eroc(n_det: list[int], concav: bool) -> np.ndarray:
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
        if concav:
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


def zigzag(func: np.ndarray) -> np.ndarray:
    """Get zigzag piecewise linear function.

    Args:
        func: A 2xn array of points on the function.

    Returns:
        A 2x(2n-1) array of points on the zigzag function.
    """
    new_func = [list(func[:, 0])]
    for idx, point in enumerate(func[:, :-1].T):
        new_func.append([func[0, idx + 1], point[1]])
        new_func.append(list(func[:, idx + 1]))
    return np.array(new_func).T


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


def amle_lin(
    val: np.ndarray, count: np.ndarray, mix: float, lin_comb: float
) -> np.ndarray:
    """AMLE with linear combination.

    TODO: Allow ``mix`` to be 0.

    Args:
        val: Likelihood values (excluding zero and infinity).  Must be
            strictly increasing.
        count: Counts or proportional to counts for likelihood values
            (including zero and infinity).  Only the first and the
            last counts can be zero.
        mix: Mixing parameter as the fraction of H1 samples.
        mu: Linear combination parameter.  Result is mu times the
            lower-left anchored AMLE plus (1-mu) times the upper-right
            anchored AMLE.

    Returns:
        A 2xn array representing the points on the piecewise curve
        after truncation and/or concavifying.  The first point is (0,
        0), and the last point is (1, 1).  Both coordinates are
        nondecreasing.
    """
    try:
        upper_right = proc(val, count, mix)
        curve = (
            upper_right - lin_comb * upper_right[:, -1:] if lin_comb else upper_right
        )
    except ValueError as exc:
        if lin_comb != 1:
            raise ValueError(
                "Upper-right PROC is not defined, so ``mu`` must be 1."
            ) from exc
        # ``mix`` must be 1 because of the outer exception.
        curve = proc([1 / v for v in val[::-1]], count[::-1], 0)[::-1, ::-1]
    return rectify(curve, True)


def proc(val: np.ndarray, count: np.ndarray, mix: float) -> np.ndarray:
    """Pseudo ROC curve anchored at (1, 1).

    Args:
        val: Likelihood values of length n (excluding zero and
            infinity).  Must be strictly increasing.
        count: Counts or proportional to counts for likelihood values
            of length n+2 (including zero and infinity).  Only the
            first and the last counts can be zero.
        mix: Mixing parameter as the fraction of H1 samples.  Must be
            in [0, 1].

    Returns:
        A 2 x (n + 3) array of points on the pseudo ROC curve.  Both
        coordinates are nonincreasing.  The last point could be (x_{n
        + 3}, -inf).
    """
    if mix == 1 and count[0]:
        raise ValueError("Upper-right anchored PROC is not well-defined.")
    pmf = count / sum(count)
    curve = [[1, 1]]
    for i, slope in enumerate(np.insert(val, 0, 0)):
        shingle = [pmf[i] / (1 - mix + mix * slope)]
        shingle.append(shingle[0] * slope)
        curve.append([curve[-1][0] - shingle[0], curve[-1][1] - shingle[1]])
    curve.append([curve[-1][0], curve[-1][0] * (1 - 1 / mix) if mix else -np.inf])
    return np.array(curve).T


def rectify(curve: np.ndarray, truncate_first: bool) -> np.ndarray:
    """Rectifies a pseudo ROC curve into the unit square.

    Args:
        curve: A 2xm array of points on the pseudo ROC curve.  Both
            coordinates are in nonincreasing order.
        truncate_first: Truncates the curve before concavifying it.

    Returns:
        A 2xn array of points on the rectified curve.  The first point
        is (0, 0), and the last point is (1, 1).  Both coordinates are
        nondecreasing.
    """
    tall = not wide(curve)
    if tall:
        curve = flip(curve)
    if truncate_first:
        curve = truncate(curve)
        curve = concavify(curve)
    else:
        curve = concavify(curve)
        curve = truncate(curve)
    if tall:
        curve = flip(curve)
    return curve[:, ::-1]


def wide(curve: np.ndarray) -> bool:
    """Checks if a curve is wide."""
    return curve[0][0] > 1 or curve[0][-1] < 0


def truncate(curve: np.ndarray) -> np.ndarray:
    """Truncates the left part of a wide curve.

    Args:
        curve: A 2xm array of points on the pseudo ROC curve.  Both
            coordinates are in nonincreasing order.

    Returns:
        A 2xn array of truncated points on the pseudo ROC curve.  The
        last point is (0, 0).
    """
    right, left = 0, curve.shape[1] - 1
    while right < left - 1:
        middle = (left + right) // 2
        if curve[0][middle] > 0:
            right = middle
        else:
            left = middle
    ans = curve[:, :left]
    ans = np.append(
        ans,
        [
            [0],
            [
                (curve[0][right] * curve[1][left] - curve[1][right] * curve[0][left])
                / (curve[0][right] - curve[0][left])
            ],
        ],
        axis=1,
    )
    if ans[1][-1] > 0:
        ans = np.append(ans, [[0], [0]], axis=1)
    return ans


def concavify(curve: np.ndarray) -> np.ndarray:
    """Concavifies the right part a wide curve.

    Args:
        curve: A 2xm array of points on the pseudo ROC curve.  Both
            coordinates are in nonincreasing order.

    Returns:
        A 2xm array of points on the concavified pseudo ROC curve.
    """
    if curve[1][0] == 1:
        return curve
    right, left = 0, curve.shape[1] - 1
    while right < left - 1:
        middle = (left + right) // 2
        if curve[0][middle] < 1 and (curve[1][middle] - curve[1][middle + 1]) * (
            curve[0][middle] - 1
        ) <= (curve[0][middle] - curve[0][middle + 1]) * (curve[1][middle] - 1):
            left = middle
        else:
            right = middle
    ans = np.array([[1], [1]])
    ans = np.append(ans, curve[:, left:], axis=1)
    return ans


def flip(curve: np.ndarray) -> np.ndarray:
    """Flips a curve."""
    return np.array([[1], [1]]) - curve[::-1, ::-1]


def estimate_likelihood_ratios(
    pos: np.ndarray, neg: np.ndarray, bin_sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimates the likelihood ratios from positive and negative samples."""
    pos = pos.copy()
    neg = neg.copy()
    for i, size in enumerate(bin_sizes):
        pos[:, i] = pos[:, i] // size
        neg[:, i] = neg[:, i] // size
    pos = pos.astype(int)
    neg = neg.astype(int)
    counters = [Counter(tuple(row) for row in neg), Counter(tuple(row) for row in pos)]
    val = []
    count = []
    count_inf = 0
    count_0 = 0
    for key in counters[0].keys():
        if key in counters[1]:
            val.append(
                counters[1][key]
                / counters[1].total()
                / counters[0][key]
                * counters[0].total()
            )
            count.append(counters[0][key] + counters[1][key])
        else:
            count_0 += counters[0][key]
    for key in counters[1].keys():
        if key not in counters[0]:
            count_inf += counters[1][key]
    count.insert(0, count_0)
    count.append(count_inf)
    return np.array(val), np.array(count)
