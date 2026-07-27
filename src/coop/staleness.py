"""Staleness weighting for asynchronous pseudo-gradients."""


def staleness_weight(worker_step: int, current_step: int, tau_max: int) -> float:
    """Linear ramp, not exponential: exponential decay never reaches zero, so arbitrarily
    stale updates would keep non-zero weight and need a second cutoff knob. The linear
    ramp makes tau_max both the decay rate and the hard cutoff, and near-stale work is
    not over-punished relative to mid-stale work."""
    tau = current_step - worker_step
    return min(1.0, max(0.0, 1.0 - tau / tau_max))
