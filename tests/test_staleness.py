import pytest

from coop.staleness import staleness_weight


def test_fresh_is_one():
    assert staleness_weight(10, 10, tau_max=8) == 1.0


def test_zero_at_and_after_tau_max():
    assert staleness_weight(2, 10, tau_max=8) == 0.0
    assert staleness_weight(0, 100, tau_max=8) == 0.0


def test_linear_in_between():
    for tau in range(9):
        assert staleness_weight(10 - tau, 10, tau_max=8) == pytest.approx(1.0 - tau / 8)


def test_future_step_clamped():
    assert staleness_weight(12, 10, tau_max=8) == 1.0
