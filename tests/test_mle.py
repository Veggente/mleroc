"""Tests eroc."""
import unittest
import numpy as np
from mleroc import unbalance, estimators


class TestMLE(unittest.TestCase):
    """Tests MLE estimators."""

    def test_levy(self):
        """Tests Lévy metric."""
        sim = unbalance.GaussSim()
        self.assertAlmostEqual(sim.levy(np.array([[0, 0], [1, 1]]).T), 0.19, 2)
        self.assertAlmostEqual(sim.levy(np.array([[0, 0], [1, 0], [1, 1]]).T), 0.7, 1)
        self.assertAlmostEqual(sim.levy(np.array([[0, 1], [1, 1]]).T), 0.3, 1)

    def test_amle(self):
        """Tests AMLE."""
        val = np.array([0.5, 1])
        count = np.array([1, 1, 1, 0])
        mix = 1 / 2
        new_val, pmf0 = estimators.amle(val, count, mix)
        np.testing.assert_array_almost_equal(pmf0, [2 / 3, 1 / 3])
        self.assertAlmostEqual(new_val, 1 / 2)

    def test_amle_2(self):
        """Tests AMLE."""
        val = np.array([0.5, 2])
        count = np.array([0, 1, 2, 0])
        mix = 0
        new_val, pmf0 = estimators.amle(val, count, mix)
        np.testing.assert_array_almost_equal(new_val, [1 / 2, 5 / 4])
        np.testing.assert_array_almost_equal(pmf0, [0, 1 / 3, 2 / 3])

    def test_amle_lin(self):
        """Tests AMLE linear combination."""
        val = np.array([0.5, 1])
        count = np.array([1, 1, 1, 0])
        mix = 1 / 2
        curve = estimators.amle_lin(val, count, mix, 0)
        np.testing.assert_array_almost_equal(
            curve, [[0, 0, 1 / 3, 1], [0, 5 / 6, 1, 1]]
        )
        np.testing.assert_array_almost_equal(
            estimators.amle_lin(val, count, mix, 1 / 2),
            [[0, 0, 1 / 9, 5 / 9, 1], [0, 4 / 9, 5 / 9, 7 / 9, 1]],
        )

    def test_amle_lin_2(self):
        """Tests AMLE linear combination."""
        val = np.array([0.5, 2])
        count = np.array([0, 1, 2, 0])
        mix = 0
        np.testing.assert_array_almost_equal(
            estimators.amle_lin(val, count, mix, 0), [[0, 2 / 3, 1], [0, 5 / 6, 1]]
        )
        np.testing.assert_array_almost_equal(
            estimators.amle_lin(val, count, 1e-10, 2 / 3), [[0, 7 / 12, 1], [0, 1, 1]]
        )
