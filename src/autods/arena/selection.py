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
from autods.state import ModelResult

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
    needed = math.ceil(SELECTION_MIN_FOLD_WIN_RATE * len(a.fold_scores))
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
    contenders = sorted(
        (r for r in results if r.status == "success" and r.model_id != BASELINE_ID),
        key=_simplicity,
    )
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
            notes.append(
                f"Chose {simpler.model_name} over {top.model_name}: {top.model_name}'s "
                f"higher mean {top.primary_metric} ({_mean(top):.4f} vs {_mean(simpler):.4f}) "
                f"held on only {wins} of {n} folds, so the simpler model wins the tie."
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
