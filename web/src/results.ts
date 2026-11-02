// Pure functions behind the Results tab: they turn numbers the server computed (lift, fold
// wins, fold scores, file sizes) into text and positions. None of them computes a metric.

import { formatMetric, METRIC_LABELS } from './stages'
import type { BaselineLift, ModelResult, PairComparison, RunSummary, Selection } from './types'

/** Mirrors BASELINE_ID in src/autods/arena/selection.py. */
export const BASELINE_ID = 'dummy'

/** Mirrors LOWER_IS_BETTER in src/autods/arena/scoring.py: error metrics, smaller wins. */
const LOWER_IS_BETTER = new Set(['rmse', 'mae'])

export function lowerIsBetter(metric: string): boolean {
  return LOWER_IS_BETTER.has(metric)
}

export function metricLabel(metric: string): string {
  return METRIC_LABELS[metric] ?? metric
}

// --- Recommendation card -----------------------------------------------------------------

export interface LiftText {
  headline: string // "+0.382" or "−63.7% error"
  sub: string // "over 0.500 baseline (Dummy (class priors))"
  better: boolean // false if the chosen model is no better than the baseline
}

/** The lift line beside the score. Signs use a real minus sign (−), as the design does. */
export function liftText(lift: BaselineLift): LiftText {
  const base = `${formatMetric(lift.baseline_score)} baseline (${lift.baseline_name})`
  const better = lift.value > 0
  if (lift.kind === 'percent') {
    // value = % of the baseline's error removed: 63.7 means "63.7% less error".
    const sign = better ? '−' : '+'
    return { headline: `${sign}${Math.abs(lift.value).toFixed(1)}% error`, sub: `vs ${base}`, better }
  }
  const sign = lift.value >= 0 ? '+' : '−'
  return { headline: `${sign}${formatMetric(Math.abs(lift.value))}`, sub: `over ${base}`, better }
}

// --- Why this model ----------------------------------------------------------------------

/** The sentence above the comparison rows, from the selection record. */
export function whySentence(selection: Selection, leaderboard: ModelResult[], metric: string): string {
  const name = (id: string) => leaderboard.find((r) => r.model_id === id)?.model_name ?? id
  const chosen = name(selection.chosen_model_id)
  const label = metricLabel(metric)
  const { min_fold_wins: need, n_folds: n } = selection
  if (selection.chosen_model_id !== selection.top_model_id) {
    const top = selection.comparisons.find((c) => c.model_id === selection.top_model_id)
    const won = top ? `only ${top.folds_won} of ${n}` : `fewer than ${need} of ${n}`
    return (
      `${name(selection.top_model_id)} had a better mean ${label} but beat ${chosen} on ${won} ` +
      `folds, so the simpler ${chosen} wins the tie.`
    )
  }
  if (selection.comparisons.some((c) => c.tested)) {
    return (
      `It had the best mean ${label} and beat every simpler model on at least ${need} of ${n} ` +
      'folds, so the rule kept it.'
    )
  }
  return `It had the best mean ${label}, and no simpler model competed with it.`
}

export type Square = 'chosen' | 'other' | 'empty'

export interface ComparisonView {
  squares: Square[] // one per fold
  squaresLabel: string // what the squares say, for screen readers
  verdict: string
  tone: 'ok' | 'muted'
}

/**
 * One "vs model" row. Its squares count folds for whichever model had the better mean
 * (the one the rule asks to prove itself): sage when that is the chosen model, faint when
 * it is the other one. An untested pair the other model would have won is labelled as such,
 * rather than implying the rule checked it.
 */
export function comparisonView(row: PairComparison, selection: Selection, chosenName: string): ComparisonView {
  const n = selection.n_folds
  const k = row.folds_won
  const filled: Square = row.better_mean ? 'other' : 'chosen'
  const squares = Array.from({ length: n }, (_, i): Square => (i < k ? filled : 'empty'))
  const squaresLabel = `${row.better_mean ? row.model_name : chosenName} won ${k} of ${n} folds`

  if (!row.better_mean) {
    return { squares, squaresLabel, verdict: `won ${k}/${n}`, tone: row.clearly ? 'ok' : 'muted' }
  }
  if (row.clearly) {
    // Only possible for an untested pair: a tested one that clearly won would have been chosen.
    return { squares, squaresLabel, verdict: `won ${k}/${n} · untested`, tone: 'muted' }
  }
  return { squares, squaresLabel, verdict: row.tested ? `only ${k}/${n} · not enough` : `only ${k}/${n}`, tone: 'muted' }
}

// --- Leaderboard fold dots ---------------------------------------------------------------

export interface FoldScale {
  lo: number
  hi: number
}

/**
 * One shared scale for every real model's fold scores, padded by 8% on each side so no dot
 * sits on the edge. Baselines are left out: a dummy at 0.500 would squash every real model
 * into one corner. Null when there are no fold scores at all.
 */
export function foldScale(leaderboard: ModelResult[]): FoldScale | null {
  const scores = leaderboard
    .filter((r) => r.status === 'success' && r.model_id !== BASELINE_ID)
    .flatMap((r) => r.fold_scores)
  if (scores.length === 0) return null
  const lo = Math.min(...scores)
  const hi = Math.max(...scores)
  // Every score equal: give the scale some width so the dots land in the middle.
  const pad = hi > lo ? (hi - lo) * 0.08 : Math.abs(lo) * 0.01 || 1
  return { lo: lo - pad, hi: hi + pad }
}

/** Where a fold score sits on the scale, as a percentage from the left (0–100). */
export function dotPosition(score: number, scale: FoldScale): number {
  const t = (score - scale.lo) / (scale.hi - scale.lo)
  return Math.max(0, Math.min(1, t)) * 100
}

// --- Runs without a model ----------------------------------------------------------------

export interface Ending {
  kicker: string
  title: string
  text: string
  tone: 'bad' | 'warn'
}

/** What the Results tab says about a run that ended without a recommended model. */
export function ending(summary: RunSummary): Ending {
  if (summary.status === 'failed') {
    // error.txt holds a traceback; its last line names the exception.
    const lines = (summary.error ?? '').trim().split('\n')
    return {
      kicker: 'The run crashed',
      title: 'Something went wrong inside the agent',
      text: lines[lines.length - 1] || 'No error message was saved.',
      tone: 'bad',
    }
  }
  if (summary.status === 'interrupted') {
    return {
      kicker: 'Stopped before finishing',
      title: 'This run never finished',
      text:
        'The server stopped while it was running (for example, it was restarted). ' +
        'Start the run again to get results.',
      tone: 'warn',
    }
  }
  return {
    kicker: 'Ended without a model',
    title: summary.task_type === 'unsupported' ? 'This task is out of scope' : 'No model trained successfully',
    text: summary.warnings[0] ?? 'The report explains why.',
    tone: 'bad',
  }
}

// --- Artifacts ---------------------------------------------------------------------------

/**
 * "13 B", "2.0 KB", "412 KB", "46.7 MB": about three significant figures, 1 KB = 1024 bytes
 * as file managers show.
 */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value < 100 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`
}
