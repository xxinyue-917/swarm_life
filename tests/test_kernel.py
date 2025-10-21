"""Tests for the 3-piece radial kernel."""
import math
import pytest

from backend.simulation import KernelParams, radial_kernel


def test_kernel_at_zero_distance():
    """Kernel should return max repulsion at zero distance."""
    params = KernelParams(r_rep=4.0, r_att=24.0, r_cut=36.0, a_rep=1.8, a_att=0.8)
    result = radial_kernel(0.0, params)
    assert result == pytest.approx(params.a_rep)


def test_kernel_repulsion_region():
    """Kernel should be repulsive (positive) in near region."""
    params = KernelParams(r_rep=4.0, r_att=24.0, r_cut=36.0, a_rep=1.8, a_att=0.8)

    # Test at half the repulsion radius
    r = params.r_rep / 2
    result = radial_kernel(r, params)
    assert result > 0  # Should be repulsive
    assert result < params.a_rep  # Should decrease from max


def test_kernel_attraction_region():
    """Kernel should be attractive (negative) in mid region."""
    params = KernelParams(r_rep=4.0, r_att=24.0, r_cut=36.0, a_rep=1.8, a_att=0.8)

    # Test at midpoint of attraction region
    r = (params.r_rep + params.r_att) / 2
    result = radial_kernel(r, params)
    assert result < 0  # Should be attractive (negative)


def test_kernel_zero_beyond_cutoff():
    """Kernel should be zero beyond cutoff radius."""
    params = KernelParams(r_rep=4.0, r_att=24.0, r_cut=36.0, a_rep=1.8, a_att=0.8)

    result = radial_kernel(params.r_cut, params)
    assert result == pytest.approx(0.0)

    result = radial_kernel(params.r_cut + 10.0, params)
    assert result == pytest.approx(0.0)


def test_kernel_continuity_at_r_rep():
    """Kernel should be continuous at r_rep."""
    params = KernelParams(r_rep=4.0, r_att=24.0, r_cut=36.0, a_rep=1.8, a_att=0.8)

    epsilon = 1e-6
    before = radial_kernel(params.r_rep - epsilon, params)
    at = radial_kernel(params.r_rep, params)
    after = radial_kernel(params.r_rep + epsilon, params)

    # Should be continuous (approximately equal)
    assert abs(before - at) < 0.01
    assert abs(after - at) < 0.01


def test_kernel_continuity_at_r_att():
    """Kernel should be continuous (zero) at r_att."""
    params = KernelParams(r_rep=4.0, r_att=24.0, r_cut=36.0, a_rep=1.8, a_att=0.8)

    epsilon = 1e-6
    before = radial_kernel(params.r_att - epsilon, params)
    at = radial_kernel(params.r_att, params)
    after = radial_kernel(params.r_att + epsilon, params)

    # At r_att, cosine term should be at pi, giving -0.5 * (1 + cos(pi)) = 0
    assert abs(at) < 1e-6
    assert abs(before) < 0.01
    assert abs(after) < 1e-6


def test_kernel_monotonic_repulsion():
    """Repulsion should decrease monotonically in repulsion region."""
    params = KernelParams(r_rep=4.0, r_att=24.0, r_cut=36.0, a_rep=1.8, a_att=0.8)

    r_values = [0.5, 1.0, 2.0, 3.0, params.r_rep]
    forces = [radial_kernel(r, params) for r in r_values]

    # Each force should be less than the previous (monotonic decrease)
    for i in range(1, len(forces)):
        assert forces[i] < forces[i - 1] or abs(forces[i] - forces[i - 1]) < 1e-6


def test_kernel_different_params():
    """Test kernel with different parameter values."""
    params = KernelParams(r_rep=3.0, r_att=20.0, r_cut=30.0, a_rep=2.0, a_att=1.0)

    # Zero distance
    assert radial_kernel(0.0, params) == pytest.approx(2.0)

    # In repulsion
    assert radial_kernel(1.5, params) > 0

    # In attraction
    assert radial_kernel(10.0, params) < 0

    # Beyond cutoff
    assert radial_kernel(30.0, params) == pytest.approx(0.0)


def test_kernel_symmetry():
    """Test that kernel parameters produce expected behavior."""
    params = KernelParams(r_rep=4.0, r_att=24.0, r_cut=36.0, a_rep=1.8, a_att=0.8)

    # Peak repulsion at r=0
    f_zero = radial_kernel(0.0, params)

    # Zero force at r_att
    f_att = radial_kernel(params.r_att, params)

    # Peak attraction somewhere in middle
    r_mid = (params.r_rep + params.r_att) / 2
    f_mid = radial_kernel(r_mid, params)

    assert f_zero > 0  # Repulsive
    assert abs(f_att) < 1e-6  # Near zero
    assert f_mid < 0  # Attractive
