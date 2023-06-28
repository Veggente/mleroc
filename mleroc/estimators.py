"""Estimators for the ROC curve."""
import numpy as np
from scipy.optimize import fsolve
from .roc import ROC, get_roc


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
