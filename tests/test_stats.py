import math

from linebal.stats import mcnemar_exact, mean_pass_hat_k, pass_hat_k, paired_bootstrap, wilson


def test_wilson_known_value():
    p, lo, hi = wilson(8, 10)
    assert p == 0.8
    assert math.isclose(lo, 0.4902, abs_tol=1e-3) and math.isclose(hi, 0.9433, abs_tol=1e-3)
    assert wilson(0, 5)[1] == 0.0


def test_pass_hat_k():
    assert pass_hat_k(3, 3, 3) == 1.0
    assert pass_hat_k(2, 3, 3) == 0.0
    assert math.isclose(pass_hat_k(2, 3, 1), 2 / 3)
    assert math.isclose(mean_pass_hat_k([(3, 3), (1, 3)], 1), (1 + 1 / 3) / 2)


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert math.isclose(mcnemar_exact(0, 6), 2 / 64)
    assert mcnemar_exact(3, 3) == 1.0


def test_paired_bootstrap_contains_mean():
    m, lo, hi = paired_bootstrap([0.1, 0.2, 0.3, 0.0, 0.4])
    assert lo <= m <= hi
