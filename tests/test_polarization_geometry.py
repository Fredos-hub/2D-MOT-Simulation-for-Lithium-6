"""
Geometric polarization decomposition: a probe with given handedness propagating
at angle theta to the local B-axis must split into (sigma-, pi, sigma+) components
that sum to 1, with the standard limiting cases at theta = 0 and theta = pi/2.
"""
import math
import numpy as np
import pytest

from src.absorption_and_emission_process import calculate_handedness_to_polarization


@pytest.mark.parametrize("handedness", [-1, 0, 1])
@pytest.mark.parametrize("angle", np.linspace(0.0, math.pi, 21))
def test_components_sum_to_one(angle, handedness):
    sq_minus, sq_pi, sq_plus = calculate_handedness_to_polarization(angle, handedness)
    assert sq_minus + sq_pi + sq_plus == pytest.approx(1.0, abs=1e-12)


def test_parallel_circular_is_pure_sigma():
    """k || B (theta = 0): handedness = +1 -> pure sigma+; -1 -> pure sigma-."""
    assert calculate_handedness_to_polarization(0.0, 1) == pytest.approx((0.0, 0.0, 1.0))
    assert calculate_handedness_to_polarization(0.0, -1) == pytest.approx((1.0, 0.0, 0.0))


def test_parallel_linear_is_pure_pi():
    """k || B with linear polarization along k -> pure pi component."""
    assert calculate_handedness_to_polarization(0.0, 0) == pytest.approx((0.0, 1.0, 0.0))


@pytest.mark.parametrize("handedness", [-1, 1])
def test_perpendicular_circular_quarter_half_quarter(handedness):
    """k perp B with circular polarization: 1/4 sigma-, 1/2 pi, 1/4 sigma+."""
    sq = calculate_handedness_to_polarization(math.pi / 2, handedness)
    assert sq == pytest.approx((0.25, 0.5, 0.25))
