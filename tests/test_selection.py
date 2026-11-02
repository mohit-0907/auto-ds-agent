"""Tests for the paired fold-by-fold winner rule. Hand-made results, no training."""

import pytest

from autods.arena.selection import (
    baseline_lift,
    clearly_beats,
    explain_selection,
    fold_wins,
    rank,
    select_best,
)
from autods.state import ModelResult


def _result(model_id: str, folds: list[float], metric: str = "roc_auc") -> ModelResult:
    return ModelResult(
        model_id=model_id,
        model_name=model_id.upper(),
        status="success",
        metrics={metric: sum(folds) / len(folds)},
        metrics_std={metric: 0.0},
        primary_metric=metric,
        fold_scores=folds,
    )


def _failed(model_id: str) -> ModelResult:
    return ModelResult(
        model_id=model_id,
        model_name=model_id.upper(),
        status="failed",
        primary_metric="roc_auc",
        error="ValueError: boom",
    )


DUMMY = _result("dummy", [0.5] * 5)


# --- fold_wins / clearly_beats -------------------------------------------------------


def test_fold_wins_counts_strictly_better_folds() -> None:
    a = _result("hgb", [0.90, 0.85, 0.80, 0.70, 0.95])
    b = _result("rf", [0.88, 0.85, 0.82, 0.60, 0.90])
    assert fold_wins(a, b) == 3  # fold 2 is a tie: not a win


def test_four_of_five_folds_is_clearly_better() -> None:
    a = _result("hgb", [0.90, 0.86, 0.80, 0.71, 0.95])
    b = _result("rf", [0.88, 0.85, 0.82, 0.70, 0.90])
    assert fold_wins(a, b) == 4
    assert clearly_beats(a, b)


def test_three_of_five_is_a_coin_flip_not_evidence() -> None:
    a = _result("hgb", [0.90, 0.86, 0.80, 0.69, 0.95])
    b = _result("rf", [0.88, 0.85, 0.82, 0.70, 0.90])
    assert fold_wins(a, b) == 3
    assert not clearly_beats(a, b)


def test_winning_most_folds_but_losing_on_average_is_not_enough() -> None:
    a = _result("hgb", [0.81, 0.81, 0.81, 0.81, 0.50])
    b = _result("rf", [0.80, 0.80, 0.80, 0.80, 0.90])
    assert fold_wins(a, b) == 4
    assert not clearly_beats(a, b)


def test_lower_is_better_metrics_compare_the_other_way() -> None:
    cheap = _result("ridge", [30_000, 31_000, 29_000, 32_000, 30_500], metric="rmse")
    good = _result("hgb", [25_000, 26_000, 24_000, 27_000, 25_500], metric="rmse")
    assert clearly_beats(good, cheap)
    assert not clearly_beats(cheap, good)


# --- select_best --------------------------------------------------------------------


def test_top_model_wins_when_it_clearly_beats_every_simpler_one() -> None:
    logreg = _result("logreg", [0.84, 0.85, 0.83, 0.86, 0.85])
    hgb = _result("hgb", [0.89, 0.88, 0.87, 0.90, 0.88])
    best, notes = select_best([DUMMY, logreg, hgb])

    assert best == "hgb"
    assert notes == []


def test_simpler_model_wins_a_tie() -> None:
    """hgb's higher mean comes from one lucky fold: logreg (simpler) is chosen."""
    logreg = _result("logreg", [0.85, 0.85, 0.85, 0.85, 0.85])
    hgb = _result("hgb", [0.84, 0.84, 0.86, 0.84, 0.95])
    best, notes = select_best([DUMMY, logreg, hgb])

    assert best == "logreg"
    assert "held on only 2 of 5 folds" in notes[0]


def test_tie_note_for_rmse_says_better_not_higher() -> None:
    """Found on the mpg run: the note called a lower (better) RMSE "higher"."""
    ridge = _result("ridge", [3.0, 3.0, 3.0, 3.0, 3.0], metric="rmse")
    rf = _result("rf", [2.9, 3.1, 2.9, 3.1, 2.7], metric="rmse")  # better mean, wins 3/5
    best, notes = select_best([_result("dummy", [8.0] * 5, metric="rmse"), ridge, rf])

    assert best == "ridge"
    assert "RF's better mean rmse (2.9400 vs 3.0000, lower is better)" in notes[0]
    assert "higher mean" not in notes[0]


def test_the_simplest_untied_model_is_chosen() -> None:
    """lgbm tops the mean; it clearly beats logreg but not hgb, so hgb wins."""
    logreg = _result("logreg", [0.80, 0.80, 0.80, 0.80, 0.80])
    hgb = _result("hgb", [0.88, 0.87, 0.89, 0.88, 0.86])
    lgbm = _result("lgbm", [0.89, 0.86, 0.90, 0.87, 0.89])
    best, _ = select_best([DUMMY, logreg, hgb, lgbm])
    assert best == "hgb"


def test_dummy_is_never_the_winner() -> None:
    weak = _result("logreg", [0.50, 0.49, 0.51, 0.50, 0.50])
    best, notes = select_best([DUMMY, weak])

    assert best == "logreg"
    assert "does not clearly beat the DUMMY baseline" in notes[-1]


def test_failed_models_cannot_win() -> None:
    logreg = _result("logreg", [0.80] * 5)
    best, _ = select_best([DUMMY, logreg, _failed("hgb")])
    assert best == "logreg"


def test_no_real_model_means_no_winner() -> None:
    best, notes = select_best([DUMMY, _failed("logreg"), _failed("hgb")])

    assert best is None
    assert "no best model" in notes[0]


def test_selection_ignores_input_order() -> None:
    results = [
        DUMMY,
        _result("logreg", [0.80] * 5),
        _result("hgb", [0.88, 0.87, 0.89, 0.88, 0.86]),
        _result("rf", [0.89, 0.86, 0.90, 0.87, 0.89]),
    ]
    assert select_best(results) == select_best(list(reversed(results)))


# --- rank ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "expected"),
    [("roc_auc", ["hgb", "logreg", "dummy", "rf"]), ("rmse", ["dummy", "logreg", "hgb", "rf"])],
)
def test_rank_orders_by_metric_direction_with_failures_last(
    metric: str, expected: list[str]
) -> None:
    results = [
        _result("logreg", [0.7] * 5, metric),
        _failed("rf"),
        _result("hgb", [0.9] * 5, metric),
        _result("dummy", [0.5] * 5, metric),
    ]
    assert [r.model_id for r in rank(results)] == expected


# --- explain_selection ----------------------------------------------------------------


def _rows(results: list[ModelResult]) -> dict[str, tuple[bool, int, bool, bool]]:
    """explain_selection's rows as {model_id: (better_mean, folds_won, clearly, tested)}."""
    best, _ = select_best(results)
    selection = explain_selection(results, best)
    assert selection is not None and selection.chosen_model_id == best
    return {
        c.model_id: (c.better_mean, c.folds_won, c.clearly, c.tested) for c in selection.comparisons
    }


def test_explains_a_top_model_that_beat_every_simpler_one() -> None:
    logreg = _result("logreg", [0.84, 0.85, 0.83, 0.86, 0.85])
    hgb = _result("hgb", [0.89, 0.88, 0.87, 0.90, 0.88])
    rf = _result("rf", [0.88, 0.89, 0.86, 0.89, 0.87])
    results = [DUMMY, logreg, hgb, rf]
    selection = explain_selection(results, "hgb")

    assert (selection.top_model_id, selection.n_folds, selection.min_fold_wins) == ("hgb", 5, 4)
    assert [c.model_id for c in selection.comparisons] == ["rf", "logreg"]  # leaderboard order
    assert _rows(results) == {
        "rf": (False, 4, True, False),  # more complex: the rule never needed to test it
        "logreg": (False, 5, True, True),  # simpler: tested, and hgb won 5 of 5
    }


def test_explains_a_simpler_model_kept_in_a_tie_and_shows_an_untested_pair() -> None:
    """rf tops the mean but beats hgb on only 3 folds, so hgb is kept.

    lgbm is the case the rule does not look at: it clearly beats hgb (4 folds and
    a better mean), but select_best only tests the top model, so hgb still wins.
    The row says so with tested=False instead of implying it was checked.
    """
    ridge = _result("ridge", [0.80] * 5)
    hgb = _result("hgb", [0.85] * 5)
    lgbm = _result("lgbm", [0.86, 0.86, 0.86, 0.86, 0.84])
    rf = _result("rf", [0.84, 0.84, 0.90, 0.90, 0.90])
    results = [DUMMY, ridge, hgb, lgbm, rf]

    assert explain_selection(results, "hgb").top_model_id == "rf"
    assert _rows(results) == {
        "rf": (True, 3, False, True),  # the test that decided the tie
        "lgbm": (True, 4, True, False),
        "ridge": (False, 5, True, False),  # rf was tested against ridge, not hgb
    }


def test_tested_rows_agree_with_the_decision() -> None:
    """Every row select_best actually ran must match its choice."""
    cases = [
        [DUMMY, _result("logreg", [0.85] * 5), _result("hgb", [0.84, 0.84, 0.86, 0.84, 0.95])],
        [
            DUMMY,
            _result("logreg", [0.80] * 5),
            _result("hgb", [0.88, 0.87, 0.89, 0.88, 0.86]),
            _result("lgbm", [0.89, 0.86, 0.90, 0.87, 0.89]),
        ],
    ]
    for results in cases:
        best, _ = select_best(results)
        selection = explain_selection(results, best)
        for row in (c for c in selection.comparisons if c.tested):
            # The chosen model kept its place: either it was the top model and clearly beat
            # this simpler one, or this row is the top model that failed to beat it.
            assert row.clearly == (selection.chosen_model_id == selection.top_model_id)


def test_no_chosen_model_means_no_explanation() -> None:
    assert explain_selection([DUMMY, _failed("logreg")], None) is None


# --- baseline_lift ---------------------------------------------------------------------


def test_lift_is_a_difference_for_higher_is_better_metrics() -> None:
    lift = baseline_lift([DUMMY, _result("hgb", [0.88] * 5)], "hgb")
    assert (lift.kind, lift.baseline_name, lift.baseline_score) == ("difference", "DUMMY", 0.5)
    assert lift.value == pytest.approx(0.38)


def test_lift_is_an_error_reduction_for_error_metrics() -> None:
    dummy = _result("dummy", [8.0] * 5, metric="rmse")
    lift = baseline_lift([dummy, _result("hgb", [2.0] * 5, metric="rmse")], "hgb")
    assert (lift.kind, lift.value) == ("percent", pytest.approx(75.0))  # 8 -> 2: 75% less error


def test_no_lift_without_a_baseline_or_an_error_to_reduce() -> None:
    assert baseline_lift([_result("hgb", [0.88] * 5)], "hgb") is None
    perfect = _result("dummy", [0.0] * 5, metric="rmse")
    assert baseline_lift([perfect, _result("hgb", [0.0] * 5, metric="rmse")], "hgb") is None
