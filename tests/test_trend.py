import math

from coop import trend


def series(losses, spec="val.bin:8x4@0", step0=1, per_step_tokens=1_000_000):
    return [
        {
            "step": step0 + i,
            "val_loss": v,
            "tokens": (i + 1) * per_step_tokens,
            "spec": spec,
        }
        for i, v in enumerate(losses)
    ]


def test_spec_names_the_measurement_that_ran():
    cfg = {"eval": {"val_file": "val.bin", "batches": 8, "batch_size": 2}}
    assert trend.spec(cfg) == "val.bin:8x2@0"
    # a widened eval is a different metric, so the fingerprint has to change
    assert trend.spec({"eval": {**cfg["eval"], "batches": 64}}) != trend.spec(cfg)
    # defaults are shared with the eval call site: no eval config named, same answer
    assert trend.eval_params({})["batches"] == trend.DEFAULTS["batches"]


def test_append_is_append_only_and_survives_a_torn_line(tmp_path):
    p = tmp_path / "ledger" / trend.HISTORY
    trend.append(p, {"step": 1, "val_loss": 5.0, "tokens": 10, "spec": "s"})
    hist = trend.append(p, {"step": 2, "val_loss": 4.0, "tokens": 20, "spec": "s"})
    assert [e["step"] for e in hist] == [1, 2]

    with p.open("a") as f:
        f.write('{"step": 3, "val_l')  # tick killed mid-write
    hist = trend.append(p, {"step": 4, "val_loss": 3.0, "tokens": 40, "spec": "s"})
    assert [e["step"] for e in hist] == [1, 2, 4]  # the torn point, not the series


def test_only_comparable_points_are_compared():
    hist = series([5.0, 4.8], spec="old") + series([4.0, 3.9, 3.8], spec="new", step0=3)
    seg = trend.comparable(hist, "new")
    assert [e["step"] for e in seg] == [3, 4, 5]
    assert trend.comparable(hist, "unseen") == []
    assert trend.summarize(hist, "unseen") is None


def test_ols_recovers_a_known_line():
    xs = [1.0, 2.0, 3.0, 4.0]
    slope, stderr = trend.ols(xs, [10.0 - 2.0 * x for x in xs])
    assert slope == -2.0 and stderr < 1e-9
    assert trend.ols([1.0, 2.0], [1.0, 2.0])[1] == math.inf  # exact fit reports no error
    assert trend.ols([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])[1] == math.inf  # no x spread


def test_verdict_down_survives_a_bouncing_series():
    # noisy but descending: the up-and-down must not hide the slope
    s = trend.summarize(series([6.0, 5.6, 5.8, 5.1, 5.3, 4.6]))
    assert s["verdict"] == "down"
    assert s["slope"] < 0
    assert s["improved"] == 3 and s["worsened"] == 2
    assert s["best"] == 4.6 and s["since_best"] == 0
    assert "going down" in trend.describe(s)


def test_verdict_flat_when_noise_swamps_the_slope():
    s = trend.summarize(series([5.0, 5.4, 4.7, 5.3, 4.8, 5.2]))
    assert s["verdict"] == "flat"
    assert "too noisy" in trend.describe(s)


def test_verdict_up_is_called_out():
    s = trend.summarize(series([4.0, 4.2, 4.5, 4.7, 5.0]))
    assert s["verdict"] == "up"
    assert "RISING" in trend.describe(s)
    assert "something is wrong" in trend.headline(s)


def test_too_few_points_is_not_a_trend():
    s = trend.summarize(series([9.0, 4.0, 3.0]))
    assert s["verdict"] == "early" and "slope" not in s
    assert "too early" in trend.headline(s)


def test_steps_since_best_tracks_a_plateau():
    s = trend.summarize(series([5.0, 4.0, 3.0, 3.1, 3.2, 3.15]))
    assert s["best"] == 3.0 and s["best_step"] == 3 and s["since_best"] == 3
    assert "best 3.0000 at step 3 (3 steps ago)" in trend.describe(s)


def test_slope_is_per_token_not_per_step():
    """Two runs drop the same amount per step, but one needed 10x the tokens to do it —
    the token axis is what makes their rates comparable."""
    fast = trend.summarize(series([6.0, 5.0, 4.0, 3.0], per_step_tokens=1_000_000))
    slow = trend.summarize(series([6.0, 5.0, 4.0, 3.0], per_step_tokens=10_000_000))
    assert fast["per"] == "1M tokens"
    assert fast["slope"] < slow["slope"] < 0
    assert math.isclose(fast["slope"], 10 * slow["slope"], rel_tol=1e-6)


def test_step_axis_fallback_when_no_tokens_recorded():
    hist = [{"step": i, "val_loss": v, "spec": "s"} for i, v in enumerate([6.0, 5.0, 4.0, 3.0])]
    s = trend.summarize(hist, "s")
    assert s["per"] == "step" and s["verdict"] == "down"


def test_sparkline_plots_against_tokens():
    # a long quiet stretch then a burst: the flat span has to occupy the width it took
    hist = series([6.0, 5.0], per_step_tokens=1) + [
        {"step": 3 + i, "val_loss": v, "tokens": 100_000_000 + i, "spec": "val.bin:8x4@0"}
        for i, v in enumerate([4.0, 3.0, 2.0])
    ]
    spark = trend.summarize(hist)["spark"]
    assert len(spark) == 32
    assert spark[0] == trend.BLOCKS[-1]  # highest loss first
    assert spark[-1] == trend.BLOCKS[0]  # lowest loss last
    assert spark.count(spark[1]) > 20  # the quiet stretch, not one cell

    flat = trend.sparkline(series([3.0] * 6))
    assert set(flat) == {trend.BLOCKS[0]}  # no range: no invented wiggle


def test_chance_loss_is_where_a_run_starts():
    assert math.isclose(trend.chance_loss(8192), 9.01, abs_tol=0.01)
    assert math.isclose(trend.chance_loss(32768), 10.4, abs_tol=0.01)
