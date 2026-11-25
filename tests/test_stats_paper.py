"""Reproduce the worked examples in the papers the audit relies on.

* Bailey, D. H. & López de Prado, M. (2014). The Deflated Sharpe Ratio. Journal of Portfolio
  Management 40(5). Section "A numerical example" (p. 9-10 of the SSRN/author version):
  N = 100 trials, V[SR_n] = 1/2 (annualized), T = 1250 daily observations, skew -3, kurtosis 10,
  annualized SR 2.5  ->  SR0 ~= 0.1132 (non-annualized) and DSR ~= 0.9004.
  "after running only N = 46 independent trials ... DSR would have been 0.9505";
  with Normal returns (skew 0, kurtosis 3), "DSR = 0.9505 after N = 88 independent trials".

* Bailey, D. H. & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. Journal of Risk
  15(2). Section 5 and appendix A.3: annualized SR 2 vs benchmark 1 at 95% confidence needs
  2.73 years (daily, Normal), 2.83 (weekly), 3.24 (monthly), and 4.99 years = 59.895 months with
  monthly skew -0.72 and kurtosis 5.78; PSR at T = 59.895 is 0.95.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nba_auditor.stats import (deflated_sharpe, expected_max_sharpe, gaussian_top_share, lo_sharpe, max_drawdown,
                               min_track_record_length, moments, newey_west_tstat, probabilistic_sharpe)

SR_DAILY = 2.5 / math.sqrt(250)
V_DAILY = 0.5 / 250


def test_dsr_paper_sr0():
    assert expected_max_sharpe(100, V_DAILY) == pytest.approx(0.1132, abs=5e-5)


def test_dsr_paper_main_example():
    out = deflated_sharpe(SR_DAILY, 1250, -3, 10, 100, V_DAILY)
    assert out["dsr"] == pytest.approx(0.9004, abs=5e-4)


@pytest.mark.parametrize("n,skew,kurt", [(46, -3, 10), (88, 0, 3)])
def test_dsr_paper_threshold_examples(n, skew, kurt):
    assert deflated_sharpe(SR_DAILY, 1250, skew, kurt, n, V_DAILY)["dsr"] == pytest.approx(0.9505, abs=5e-4)


def test_dsr_decreases_with_trials():
    vals = [deflated_sharpe(SR_DAILY, 1250, -3, 10, n, V_DAILY)["dsr"] for n in (1, 10, 100, 1000)]
    assert vals == sorted(vals, reverse=True)
    assert deflated_sharpe(SR_DAILY, 1250, -3, 10, 1, V_DAILY)["sr0"] == 0.0


def test_min_trl_paper_four_moments():
    sr, ref = 2 / math.sqrt(12), 1 / math.sqrt(12)
    assert min_track_record_length(sr, ref, -0.72, 5.78) == pytest.approx(59.895, abs=1e-3)
    assert probabilistic_sharpe(sr, ref, 59.895, -0.72, 5.78) == pytest.approx(0.95, abs=1e-6)


@pytest.mark.parametrize("ppy,years", [(252, 2.73), (52, 2.83), (12, 3.24)])
def test_min_trl_paper_normal(ppy, years):
    sr, ref = 2 / math.sqrt(ppy), 1 / math.sqrt(ppy)
    assert min_track_record_length(sr, ref, 0, 3) / ppy == pytest.approx(years, abs=0.005)


def test_min_trl_infinite_when_not_above_benchmark():
    assert math.isinf(min_track_record_length(0.01, 0.02, 0, 3))


def test_expected_max_matches_simulation():
    rng = np.random.default_rng(0)
    sims = rng.standard_normal((20000, 100)).max(axis=1).mean()
    assert expected_max_sharpe(100, 1.0) == pytest.approx(sims, abs=0.03)


def test_moments_normal():
    x = np.random.default_rng(1).standard_normal(200_000)
    mu, sd, sk, ku = moments(x)
    assert abs(mu) < 0.01 and sd == pytest.approx(1, abs=0.01) and abs(sk) < 0.03 and ku == pytest.approx(3, abs=0.05)


def test_newey_west_equals_naive_for_iid_and_shrinks_for_positive_autocorrelation():
    rng = np.random.default_rng(2)
    x = rng.normal(0.05, 1, 5000)
    naive = x.mean() / (x.std(ddof=0) / math.sqrt(len(x)))
    assert newey_west_tstat(x, lags=0) == pytest.approx(naive, rel=1e-9)
    ma = np.convolve(rng.normal(0.05, 1, 5004), np.ones(5) / 5, mode="valid")  # overlapping 5-period returns
    assert newey_west_tstat(ma, lags=10) < 0.6 * newey_west_tstat(ma, lags=0)


def test_lo_adjustment_penalizes_positive_autocorrelation():
    rng = np.random.default_rng(3)
    e = rng.normal(0.0005, 0.01, 3000)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, len(e)):
        x[i] = 0.4 * x[i - 1] + e[i]
    lo = lo_sharpe(x, 252)
    assert lo["sharpe_ann_lo"] < lo["sharpe_ann_naive"]
    assert lo["t_lo"] < lo["t_iid"]
    iid = lo_sharpe(e, 252)
    assert iid["sharpe_ann_lo"] == pytest.approx(iid["sharpe_ann_naive"], rel=0.15)


def test_max_drawdown_hand_example():
    d = max_drawdown(np.array([0.1, -0.5, 0.2, 1.0]))
    assert d["max_drawdown"] == pytest.approx(-0.5)
    assert (d["peak_index"], d["trough_index"]) == (0, 1)
    assert d["longest_underwater_periods"] == 2 and not d["currently_underwater"]
    d2 = max_drawdown(np.array([-0.1, 0.05]))
    assert d2["peak_index"] == -1 and d2["currently_underwater"]


def test_gaussian_top_share_matches_simulation():
    rng = np.random.default_rng(4)
    shares = []
    for _ in range(200):
        x = rng.normal(0.0008, 0.01, 2520)
        top = np.sort(x)[::-1][:126].sum()
        shares.append(top / x.sum())
    assert gaussian_top_share(0.0008, 0.01, 2520) == pytest.approx(np.median(shares), rel=0.15)
