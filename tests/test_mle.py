"""Tests eroc."""
import unittest
import numpy as np
import unbalance


class TestMLE(unittest.TestCase):
    """Tests MLE estimators."""

    def test_levy(self):
        """Tests Lévy metric."""
        dist = unbalance.levy(np.array([[0, 0], [1, 1]]).T, True)
        print("dist =", dist)
        self.assertAlmostEqual(dist, 0.19, 2)
        dist2 = unbalance.levy(np.array([[0, 0], [1, 1]]).T, False)
        print("dist2 =", dist2)
        self.assertAlmostEqual(dist2, 0.7, 1)
        dist3 = unbalance.levy(np.array([[0, 1], [1, 1]]).T, True)
        print("dist3 =", dist3)
        self.assertAlmostEqual(dist3, 0.3, 1)

    def test_gauss_roc(self):
        """Tests true Gaussian ROC."""
        pfa = np.linspace(0.4, 0.6, 201)
        res = np.array([unbalance.gauss_roc(p) for p in pfa]).T
        tan = res[0, 100] + res[1, 100] * (pfa - 0.5)
        self.assertTrue((tan >= res[0]).all())
