"""Tests for the paired fold-by-fold winner rule. Hand-made results, no training."""

import pytest

from autods.arena.selection import clearly_beats, fold_wins, rank, select_best
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
