"""Receiver operating characteristic curve."""
from dataclasses import dataclass
import bisect
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats._distn_infrastructure import rv_sample


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


class DiscDist(rv_sample):
    """Discrete distribution on float numbers."""

    def __init__(self, flt_values: np.ndarray):
        super().__init__(values=(list(range(flt_values.shape[1])), list(flt_values[1])))
        self.val = flt_values[0]

    def rvs(
        self,
        *args,
        size: int | None = None,
        random_state: np.random.Generator | None = None,
        **kwds,  # pylint: disable=unused-argument
    ) -> np.ndarray:
        """Generates i.i.d. random variables."""
        indices = super().rvs(size=size, random_state=random_state)
        return self.val[indices]


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
