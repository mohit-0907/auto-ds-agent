// Turning the stream of stage events into what the timeline draws. Plain functions, no React.

import type { StageEvent } from './types'

/** Friendly names for the graph's node names. */
export const STAGE_LABELS: Record<string, string> = {
  profiler: 'Profile data',
  router: 'Choose task',
  sanity_check: 'Check decision',
  fe_codegen: 'Write features',
  sandbox: 'Run in sandbox',
  fallback_fe: 'Fallback features',
  arena: 'Benchmark models',
  artifacts: 'Save model',
  reporter: 'Write report',
  finalize: 'Finish',
}

export function stageLabel(node: string): string {
  return STAGE_LABELS[node] ?? node
}

/** One row of the timeline: a single stage, or a feature-engineering attempt block. */
export type TimelineItem =
  | { kind: 'stage'; event: StageEvent }
  | { kind: 'attempt'; attempt: number; events: StageEvent[] }

const ATTEMPT_NODES = new Set(['fe_codegen', 'sandbox'])

/**
 * Group the corrective loop's stages into "Attempt N" blocks so retries read clearly.
 *
 * Events are expected in index order. fe_codegen/sandbox events with attempt >= 1 join the
 * block for their attempt; everything else (including a codegen call that failed before
 * producing any code, attempt 0) is a top-level row.
 */
export function groupTimeline(events: StageEvent[]): TimelineItem[] {
  const items: TimelineItem[] = []
  for (const event of events) {
    const last = items[items.length - 1]
    if (ATTEMPT_NODES.has(event.node) && event.attempt >= 1) {
      if (last?.kind === 'attempt' && last.attempt === event.attempt) {
        last.events.push(event)
      } else {
        items.push({ kind: 'attempt', attempt: event.attempt, events: [event] })
      }
    } else {
      items.push({ kind: 'stage', event })
    }
  }
  return items
}

/** Format a metric like the Python reports do: 3 decimals below 10, else whole numbers. */
export function formatMetric(value: number): string {
  return Math.abs(value) >= 10
    ? Math.round(value).toLocaleString('en-US')
    : value.toFixed(3)
}

/** Display names for metric keys (mirrors METRIC_LABELS in src/autods/reporting/facts.py). */
export const METRIC_LABELS: Record<string, string> = {
  roc_auc: 'ROC-AUC',
  f1_macro: 'macro-F1',
  rmse: 'RMSE',
  mae: 'MAE',
  r2: 'R²',
  accuracy: 'accuracy',
}
