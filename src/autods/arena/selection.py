"""Pick the best model with a paired, fold-by-fold comparison. Deterministic; no LLM.

The rule: the model with the best mean wins -- unless it fails to clearly beat
a simpler model, in which case the simplest model it fails to beat wins.
"Clearly beats" means better on at least SELECTION_MIN_FOLD_WIN_RATE of the
folds AND better on average.

Why paired: every model was scored on the same CV folds, so some folds are
hard for every model and some are easy. Comparing fold by fold cancels that
shared difficulty out; comparing means +- std does not, which makes it far
too pessimistic about real differences.

Ties are broken by a fixed simplicity order, never by measured time: timings
change slightly between runs, and the winner must not (invariant 8).
"""

from __future__ import annotations

import math

from autods.arena.scoring import LOWER_IS_BETTER
from autods.config import SELECTION_MIN_FOLD_WIN_RATE
from autods.state import BaselineLift, ModelResult, PairComparison, Selection

BASELINE_ID = "dummy"

# Simplest first. Linear models are the most interpretable; hgb ships with
# scikit-learn (nothing extra to deploy); lgbm and xgb need their own native
# libraries; rf is the largest artifact and slowest to predict (74 ms vs 21 ms
# per fold for hgb on Titanic). Unknown IDs sort last, in arrival order.
SIMPLICITY_ORDER: tuple[str, ...] = ("logreg", "ridge", "hgb", "lgbm", "xgb", "rf")


def _simplicity(result: ModelResult) -> int:
    model_id = result.model_id
    return (
        SIMPLICITY_ORDER.index(model_id) if model_id in SIMPLICITY_ORDER else len(SIMPLICITY_ORDER)
    )


def _better(a: float, b: float, metric: str) -> bool:
    return a < b if metric in LOWER_IS_BETTER else a > b


def _mean(result: ModelResult) -> float:
    return result.metrics[result.primary_metric]


def _min_fold_wins(n_folds: int) -> int:
    return math.ceil(SELECTION_MIN_FOLD_WIN_RATE * n_folds)


def _contenders(results: list[ModelResult]) -> list[ModelResult]:
    """Successful non-baseline models, simplest first: who competes to be chosen."""
    return sorted(
        (r for r in results if r.status == "success" and r.model_id != BASELINE_ID),
        key=_simplicity,
    )


def fold_wins(a: ModelResult, b: ModelResult) -> int:
    """How many shared folds a scored strictly better than b on."""
    metric = a.primary_metric
    return sum(_better(x, y, metric) for x, y in zip(a.fold_scores, b.fold_scores, strict=True))


def clearly_beats(a: ModelResult, b: ModelResult) -> bool:
    """True if a wins on enough folds AND on average to rule out luck.

    Args:
        a: The challenger.
        b: The model it is compared against, scored on the same folds.

    Returns:
        Whether a's advantage over b holds up fold by fold.
    """
    needed = _min_fold_wins(len(a.fold_scores))
    return fold_wins(a, b) >= needed and _better(_mean(a), _mean(b), a.primary_metric)


def rank(results: list[ModelResult]) -> list[ModelResult]:
    """Leaderboard order: successful models by mean primary metric, best first, then failures.

    Equal means are ordered by simplicity, so the order is fully deterministic.
    """
    ok = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status != "success"]
    if not ok:
        return failed
    sign = 1 if ok[0].primary_metric in LOWER_IS_BETTER else -1
    return sorted(ok, key=lambda r: (sign * _mean(r), _simplicity(r))) + failed


def select_best(results: list[ModelResult]) -> tuple[str | None, list[str]]:
    """Apply the paired rule to a run's results.

    The dummy is the yardstick, never the winner. If the chosen model does not
    clearly beat it, the choice stands but a warning says so.

    Args:
        results: Every candidate's ModelResult from one run (same folds).

    Returns:
        (best_model_id, notes). best_model_id is None when no real model
        trained successfully. Notes explain the choice for the report.
    """
    contenders = _contenders(results)
    if not contenders:
        return None, ["No model besides the baseline trained successfully; no best model."]

    top = rank(contenders)[0]
    chosen, notes = top, []
    for simpler in contenders:
        if simpler is top:
            break
        if not clearly_beats(top, simpler):
            chosen = simpler
            wins, n = fold_wins(top, simpler), len(top.fold_scores)
            # "better", not "higher": for RMSE the better mean is the lower one.
            direction = "lower" if top.primary_metric in LOWER_IS_BETTER else "higher"
            notes.append(
                f"Chose {simpler.model_name} over {top.model_name}: {top.model_name}'s "
                f"better mean {top.primary_metric} ({_mean(top):.4f} vs {_mean(simpler):.4f}, "
                f"{direction} is better) held on only {wins} of {n} folds, so the simpler "
                "model wins the tie."
            )
            break

    baseline = next(
        (r for r in results if r.model_id == BASELINE_ID and r.status == "success"), None
    )
    if baseline is not None and not clearly_beats(chosen, baseline):
        notes.append(
            f"{chosen.model_name} does not clearly beat the {baseline.model_name} baseline "
            f"({_mean(chosen):.4f} vs {_mean(baseline):.4f}); the features may carry "
            "little signal for this target."
        )
    return chosen.model_id, notes


def baseline_lift(results: list[ModelResult], best_model_id: str | None) -> BaselineLift | None:
    """How far the chosen model is ahead of the dummy baseline.

    Higher-is-better metrics give the plain difference (+0.379 ROC-AUC); error
    metrics give the percentage of the baseline's error removed (64.0% lower
    RMSE), which reads the same whatever the target's units.

    Args:
        results: Every candidate's ModelResult from one run.
        best_model_id: The chosen model.

    Returns:
        The lift, or None without a chosen model, a scored baseline, or (for an
        error metric) a baseline error above zero to take a percentage of.
    """
    ok = {r.model_id: r for r in results if r.status == "success"}
    best, baseline = ok.get(best_model_id or ""), ok.get(BASELINE_ID)
    if best is None or baseline is None:
        return None
    base, value = _mean(baseline), _mean(best)
    if best.primary_metric in LOWER_IS_BETTER:
        if base <= 0:
            return None
        kind, lift = "percent", 100 * (base - value) / base
    else:
        kind, lift = "difference", value - base
    return BaselineLift(
        baseline_name=baseline.model_name, baseline_score=base, kind=kind, value=lift
    )


def explain_selection(results: list[ModelResult], best_model_id: str | None) -> Selection | None:
    """The evidence behind select_best's choice, for the 'Why this model' panel.

    One comparison per other contender, each asking the rule's question of that
    pair: did the better-mean model win enough folds? Only some of these are
    tests select_best actually ran (``tested``): it compares the top-mean model
    with the simpler models, nothing else.

    Args:
        results: Every candidate's ModelResult from one run (same folds).
        best_model_id: What select_best chose for these results.

    Returns:
        The selection record, or None when no model was chosen.
    """
    contenders = _contenders(results)
    chosen = next((r for r in contenders if r.model_id == best_model_id), None)
    if chosen is None:
        return None
    top = rank(contenders)[0]
    metric = chosen.primary_metric

    comparisons = []
    for other in rank(contenders):
        if other is chosen:
            continue
        other_better = _better(_mean(other), _mean(chosen), metric)
        leader, trailer = (other, chosen) if other_better else (chosen, other)
        # select_best tests the top model against the simpler ones, in order. A chosen top
        # model passed every one of those; otherwise the top model's failed test against
        # the chosen one is the only test that involved the chosen model.
        tested = contenders.index(other) < contenders.index(top) if chosen is top else other is top
        comparisons.append(
            PairComparison(
                model_id=other.model_id,
                model_name=other.model_name,
                better_mean=other_better,
                folds_won=fold_wins(leader, trailer),
                clearly=clearly_beats(leader, trailer),
                tested=tested,
            )
        )
    return Selection(
        chosen_model_id=chosen.model_id,
        top_model_id=top.model_id,
        n_folds=len(chosen.fold_scores),
        min_fold_wins=_min_fold_wins(len(chosen.fold_scores)),
        comparisons=comparisons,
        baseline=baseline_lift(results, chosen.model_id),
    )
