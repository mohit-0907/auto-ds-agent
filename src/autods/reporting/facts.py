"""The facts sheet: every number the report may mention, computed in Python.

Invariant 1: the reporter LLM never does arithmetic. Derived numbers -- how much
better than the baseline, fold-by-fold win counts, how many times slower --
are computed here and handed over as finished facts. The same text is then the
reference the number verifier checks the LLM's prose against.
"""

from __future__ import annotations

from pathlib import Path

from autods.arena.scoring import LOWER_IS_BETTER
from autods.arena.selection import BASELINE_ID, fold_wins
from autods.state import AgentState, ModelResult

METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "f1_macro": "macro-F1",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "precision_macro": "macro-precision",
    "recall_macro": "macro-recall",
}


def fmt(value: float) -> str:
    """Format a metric for humans, consistently everywhere in the report.

    Scores below 10 (AUC, F1, R², accuracy) get 3 decimals; larger values
    (RMSE in dollars or grams) get thousands separators and no decimals.
    """
    if abs(value) >= 10:
        return f"{value:,.0f}"
    return f"{value:.3f}"


def leaderboard_table(leaderboard: list[ModelResult], best_model_id: str | None) -> str:
    """The leaderboard as a Markdown table, built by Python -- never typed by the LLM."""
    if not leaderboard:
        return "_No models were trained._"
    first_ok = next((r for r in leaderboard if r.status == "success"), None)
    metrics = list(first_ok.metrics) if first_ok else []
    header = [
        "Rank",
        "Model",
        *[METRIC_LABELS.get(m, m) for m in metrics],
        "Train (s)",
        "Predict (ms/fold)",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for position, r in enumerate(leaderboard, start=1):
        name = f"**{r.model_name}** ✅" if r.model_id == best_model_id else r.model_name
        if r.status != "success":
            cells = [
                "–",
                f"{name} (failed: {r.error})",
                *["–"] * len(metrics),
                "–",
                "–",
            ]
        else:
            cells = [
                str(position),
                name,
                *[f"{fmt(r.metrics[m])} ± {fmt(r.metrics_std[m])}" for m in metrics],
                f"{r.training_time_s:.2f}",
                f"{r.predict_time_s * 1000:.1f}",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _arena_facts(leaderboard: list[ModelResult], best_model_id: str | None) -> list[str]:
    ok = {r.model_id: r for r in leaderboard if r.status == "success"}
    best = ok.get(best_model_id or "")
    if best is None:
        return ["- No best model was chosen."]

    metric = best.primary_metric
    label = METRIC_LABELS.get(metric, metric)
    lower = metric in LOWER_IS_BETTER
    n_folds = len(best.fold_scores)
    facts = [
        f"- Ranking metric: {label} ({'lower' if lower else 'higher'} is better), "
        f"{n_folds}-fold cross-validation; every model was scored on the same folds.",
        f"- Best model (chosen by the selection rule): {best.model_name}, "
        f"{label} {fmt(best.metrics[metric])} ± {fmt(best.metrics_std[metric])}.",
    ]

    top = (
        min(ok.values(), key=lambda r: r.metrics[metric])
        if lower
        else max(ok.values(), key=lambda r: r.metrics[metric])
    )
    if top.model_id != best.model_id:
        facts.append(
            f"- {top.model_name} had the best mean {label} "
            f"({fmt(top.metrics[metric])}, {'lower' if lower else 'higher'} is better) "
            f"but beat {best.model_name} on only "
            f"{fold_wins(top, best)} of {n_folds} folds, so the simpler {best.model_name} "
            "was chosen."
        )

    baseline = ok.get(BASELINE_ID)
    if baseline is not None:
        base, value = baseline.metrics[metric], best.metrics[metric]
        if lower and base > 0:
            facts.append(
                f"- Baseline ({baseline.model_name}): {label} {fmt(base)}. The best model's "
                f"{label} is {100 * (base - value) / base:.1f}% lower than the baseline's."
            )
        else:
            facts.append(
                f"- Baseline ({baseline.model_name}): {label} {fmt(base)}. The best model is "
                f"{fmt(value - base)} higher."
            )

    for other in ok.values():
        if other.model_id in {best.model_id, BASELINE_ID}:
            continue
        facts.append(
            f"- {best.model_name} vs {other.model_name}: {label} {fmt(best.metrics[metric])} vs "
            f"{fmt(other.metrics[metric])}; {best.model_name} was better on "
            f"{fold_wins(best, other)} of {n_folds} folds."
        )

    # Against the fastest *real* model: the dummy does no work, so it is always
    # fastest and would make every real model look slow ("2.9x the baseline").
    real = [r for r in ok.values() if r.model_id != BASELINE_ID]
    fastest = min(real, key=lambda r: r.predict_time_s)
    if fastest.model_id != best.model_id and fastest.predict_time_s > 0:
        ratio = best.predict_time_s / fastest.predict_time_s
        facts.append(
            f"- Prediction cost: {best.model_name} takes {best.predict_time_s * 1000:.1f} ms per "
            f"fold vs {fastest.predict_time_s * 1000:.1f} ms for {fastest.model_name} "
            f"({ratio:.1f}× as long)."
        )
    for r in leaderboard:
        if r.status != "success":
            facts.append(f"- {r.model_name} failed to train: {r.error}")
    return facts


def build_facts(state: AgentState) -> str:
    """Render everything the report may say as a plain-text facts sheet.

    Args:
        state: The run's final state. Any stage may be missing (failed runs
            get a report too), so every section degrades to what exists.

    Returns:
        Markdown-ish text: the LLM's only source, and the verifier's reference.
    """
    lines = [f"# Facts for the run on {Path(state['dataset_path']).name}", ""]

    profile = state["data_profile"]
    lines.append("## Dataset")
    if profile is None:
        lines.append("- The dataset could not be profiled.")
    else:
        lines += [
            f"- {profile.n_rows:,} rows, {profile.n_cols} columns, "
            f"{profile.n_duplicate_rows:,} duplicate rows.",
            f"- Constant columns: {', '.join(profile.constant_columns) or 'none'}.",
        ]
        if profile.columns_truncated:
            lines.append(
                f"- The task router saw {len(profile.columns)} of the {profile.n_cols} "
                "columns in detail (profile truncated to fit the prompt budget)."
            )
    lines.append(f"- User's description: {state['user_description'] or '(none)'}")

    decision = state["task_decision"]
    lines += ["", "## Task decision"]
    if decision is None:
        lines.append("- No task decision was made.")
    else:
        target = decision.target_column or "(none)"
        lines += [
            f"- Task type: {decision.task_type}; target column: {target}.",
            f"- Router confidence: {decision.confidence:.2f}.",
            f"- Router reasoning: {decision.reasoning}",
        ]
        if decision.unsupported_reason:
            lines.append(f"- Why unsupported: {decision.unsupported_reason}")

    lines += ["", "## Feature engineering"]
    history = state["fe_code_history"]
    for i, explanation in enumerate(state["fe_explanations"], start=1):
        outcome = (
            state["tracebacks"][i - 1].splitlines()[0]
            if i <= len(state["tracebacks"])
            else "passed validation"
        )
        lines.append(f"- Attempt {i}: {explanation} Outcome: {outcome}")
    processed = state["processed_data_path"] or ""
    if "fallback" in processed:
        lines.append(
            "- Final features: the deterministic fallback recipe "
            "(median imputation, one-hot encoding, ID and free-text columns dropped)."
        )
    elif processed:
        lines.append(f"- Final features: generated code from attempt {len(history)}.")
    else:
        lines.append("- No features were produced.")

    lines += ["", "## Model arena", *_arena_facts(state["leaderboard"], state["best_model_id"])]
    lines += ["", "## Warnings recorded during the run"]
    lines += [f"- {w}" for w in state["warnings"]] or ["- none"]
    lines += ["", f"## Cost so far\n- LLM cost before this report: ${state['token_cost_usd']:.4f}."]
    return "\n".join(lines)
