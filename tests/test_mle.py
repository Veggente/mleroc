"""Tests eroc."""
import unittest
import numpy as np
import unbalance
import bhatta_bound


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


class TestBCBound(unittest.TestCase):
    """Tests BC bound."""

    @staticmethod
    def test_bc_w_true():
        """Tests BC bound against true ROC."""
        gauss = unbalance.GaussROC([0, 1], [1, 1])
        pfa = np.linspace(0, 0.7, 8)
        roc_true = unbalance.ROC(
            pfa,
            np.array([unbalance.gauss_roc(false_alarm)[0] for false_alarm in pfa]),
            "true",
        )
        roc_bc = unbalance.ROC(
            pfa, bhatta_bound.new_roc_bound([gauss.bhatta() ** 2], pfa), "BC"
        )
        np.testing.assert_array_less(roc_true.pdet, roc_bc.pdet)
