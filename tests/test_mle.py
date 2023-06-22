"""Tests eroc."""
import unittest
import numpy as np
import unbalance  # pylint: disable=import-error
from bhatta_bound import bhatta_bound  # pylint: disable=import-error


class TestMLE(unittest.TestCase):
    """Tests MLE estimators."""

    def test_levy(self):
        """Tests Lévy metric."""
        dist = unbalance.levy(np.array([[0, 0], [1, 1]]).T, True)
        self.assertAlmostEqual(dist, 0.19, 2)
        dist2 = unbalance.levy(np.array([[0, 0], [1, 1]]).T, False)
        self.assertAlmostEqual(dist2, 0.7, 1)
        dist3 = unbalance.levy(np.array([[0, 1], [1, 1]]).T, True)
        self.assertAlmostEqual(dist3, 0.3, 1)

    def test_gauss_roc(self):
        """Tests true Gaussian ROC."""
        pfa = np.linspace(0.4, 0.6, 201)
        res = np.array([unbalance.gauss_roc(p) for p in pfa]).T
        tan = res[0, 100] + res[1, 100] * (pfa - 0.5)
        self.assertTrue((tan >= res[0]).all())

    def test_oroc_levy(self):
        """Tests Levy metric for OROC objects."""
        slopes = np.array([0.5, 1, 2])
        null_prob1 = np.array([0.5, 0.25, 0.25])
        null_prob2 = np.array([1 / 3, 1 / 2, 1 / 6])
        oroc1 = unbalance.OROC(slopes, null_prob1, "oroc1")
        oroc2 = unbalance.OROC(slopes, null_prob2, "oroc2")
        self.assertAlmostEqual(oroc1.levy(oroc2), 1 / 24)
        self.assertAlmostEqual(oroc2.levy(oroc1), 1 / 24)
        alpha = np.array([1 / 6, 0])
        self.assertAlmostEqual(oroc1.sup_norm(oroc2, alpha), 1 / 8)
        self.assertAlmostEqual(oroc2.sup_norm(oroc1, alpha[::-1]), 1 / 8)

    def test_oroc_sup(self):
        """Tests OROC sup norm."""
        slope = np.array([0.62, 0.77])
        p_null = [np.array([0.997, 0.003]), np.array([0.98, 0.02])]
        alpha = np.array([0.69, 0.65])
        oroc1 = unbalance.OROC(slope, p_null[0], "oroc1")
        oroc2 = unbalance.OROC(slope, p_null[1], "oroc2")
        self.assertAlmostEqual(oroc1.sup_norm(oroc2, alpha), 0.016839500000000063)

    def test_amle(self):
        """Tests AMLE."""
        val = np.array([0.5, 1])
        count = np.array([1, 1, 1, 0])
        mix = 1 / 2
        new_val, pmf0 = unbalance.amle(val, count, mix)
        np.testing.assert_array_almost_equal(pmf0, [2 / 3, 1 / 3])
        self.assertAlmostEqual(new_val, 1 / 2)

    def test_amle_2(self):
        """Tests AMLE."""
        val = np.array([0.5, 2])
        count = np.array([0, 1, 2, 0])
        mix = 0
        new_val, pmf0 = unbalance.amle(val, count, mix)
        np.testing.assert_array_almost_equal(new_val, [1 / 2, 5 / 4])
        np.testing.assert_array_almost_equal(pmf0, [0, 1 / 3, 2 / 3])


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
