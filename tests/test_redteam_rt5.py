"""Red-team round 5 (review of PR #9, lint v0.3 at bef5683): regression cases.

tests/corpus_rt5/*.txt hold the red team's repro blocks (`#### NAME` then the code). The 19 cases of the red team's
test_rt5.py failed on bef5683 and passed on v0.2; they are the required set below. The files also contain clean
look-alikes, LA05 mask cases and some leaks that neither v0.2 nor v0.3 catches (documented misses).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nba_auditor import check_code

RT5 = Path(__file__).parent / "corpus_rt5"


def _blocks(*names: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for fn in names:
        txt = (RT5 / fn).read_text(encoding="utf-8")
        out.update(dict(b.split("\n", 1) for b in re.split(r"^#### ", txt, flags=re.M)[1:]))
    return out


def _hm(src: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for f in check_code(src).findings:
        if f.severity in ("high", "medium"):
            out.setdefault(f.rule_id, set()).add(f.severity)
    return out


L = _blocks("leaky1.txt", "leaky2.txt", "leaky3.txt", "leaky4.txt")
C = _blocks("clean1.txt", "clean2.txt", "clean3.txt")
M = _blocks("la05.txt")

# The red team's required cases (test_rt5.py): all failed on bef5683.
REQUIRED_LEAKS = ["L04_market_timing_from_groupby_mean", "L05_rolling_sharpe_of_forward_pnl_as_weight",
                  "L08_target_encoding_loc", "L19_position_times_fwd_then_equity_filter",
                  "N03_mean_fwd_per_ticker_as_weight", "N04_fill_return_feeds_size", "P01_neutral_tilt_from_groupby",
                  "P02_booked_then_neutral_rolling", "P08_concat_feature_matrix_named_data",
                  "P09_dataframe_dict_feature_matrix_named_tbl", "Q01_concat_forgot_drop",
                  "Q02_concat_then_X_select_includes_fwd", "Q04_dataframe_np_column_stack",
                  "M04_fit_datemask_all_after_start"]
REQUIRED_CLEAN = ["C01_const_horizon_realign", "C02_func_param_horizon_realign", "C08_shift_neg_HORIZON_kw",
                  "D02_const_H", "D04_cfg_attr"]
# Correct code in the red team's files (clean1-3; D05-D10 in clean2 are leaky look-alikes, see LEAKY_IN_CLEAN_FILES)
CLEAN_LOOKALIKES = ["C03_pct_change_neg_realign", "C04_np_roll_label_masked", "C05_scaler_on_train_mask_then_model",
                    "C06_fwd_used_in_hit_rate_and_ic", "C07_entry_exit_fill_pnl", "C09_bucket_spread_on_past",
                    "C10_rolling_sharpe_lagged", "C11_lgb_with_label_and_valid", "C12_fwd_shift_neg_then_two_lags",
                    "C13_dmatrix_label_neutral", "D01_literal5", "D03_const_H_lag_bigger", "K01_ic_frame",
                    "K02_concat_then_drop"]
LEAKY_IN_CLEAN_FILES = ["D05_leaky_const_lag_smaller", "D06_leaky_lead_via_shift_neg_mul",
                        "D07_leaky_rolling_lead_window", "D08_leaky_shift_neg5_then_pct_change",
                        "D09_leaky_lead_via_kw_and_lag_kw", "D10_leaky_lag_via_diff"]
# LA05: the scaler is fitted on every row, or on rows that are only bounded below (a start date or a test window)
LA05_LEAKS = ["all", "datemask_start", "year_mask", "test_window_mask", "notna_mask", "loc_start_str"]


# Leaks in the red team's files that v0.3 still misses (documented in README "Known limitations"). P05 (a drop list
# the lint can't resolve) is "possible" by design; the others are not caught by v0.2 either.
KNOWN_MISSES = {"L10_fit_on_iloc_minus1", "L12_fit_on_test_slice", "L13_overlap_train_test",
                "L17_masked_selection_feeds_signal", "M02_fit_all_but_last_row_before_split",
                "M06_model_fit_all_minus1_predict", "P05_drop_list_reassigned_after", "P06_drop_inplace_wrong_frame",
                "P07_drop_errors_ignore_typo"}


@pytest.mark.parametrize("name", REQUIRED_LEAKS)
def test_rt5_required_leak_is_flagged(name):
    assert _hm(L[name]), name


@pytest.mark.parametrize("name", sorted(set(L) - KNOWN_MISSES))
def test_rt5_every_other_leak_stays_flagged(name):
    assert _hm(L[name]), name


def test_rt5_known_misses_are_still_misses():
    """If one of these starts being caught, move it out of KNOWN_MISSES and update the README."""
    now_caught = sorted(n for n in KNOWN_MISSES if _hm(L[n]))
    assert not now_caught, now_caught


def test_rt5_unresolvable_drop_list_is_possible():
    got = {f.severity for f in check_code(L["P05_drop_list_reassigned_after"]).findings if f.rule_id == "LA10"}
    assert got == {"possible"}, got


@pytest.mark.parametrize("name", REQUIRED_CLEAN + CLEAN_LOOKALIKES)
def test_rt5_correct_code_has_no_high_or_medium(name):
    assert not _hm(C[name]), (name, _hm(C[name]))


@pytest.mark.parametrize("name", LEAKY_IN_CLEAN_FILES)
def test_rt5_leaky_lookalike_is_flagged(name):
    assert _hm(C[name]), name


@pytest.mark.parametrize("name", LA05_LEAKS)
def test_rt5_la05_full_or_start_bounded_fit(name):
    assert "LA05" in _hm(M[name]), (name, _hm(M[name]))
