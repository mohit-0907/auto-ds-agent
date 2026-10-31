// Turning the stream of stage events into what the timeline draws. Plain functions, no React.

import type { RunListItem, RunOutcome, RunStatus, StageEvent } from './types'

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

// --- Status, progress and labels for the app shell (Task 7.6) ---------------------------

/** Display names for task types. */
export const TASK_LABELS: Record<string, string> = {
  binary_classification: 'Binary classification',
  multiclass_classification: 'Multiclass',
  regression: 'Regression',
  unsupported: 'Out of scope',
}

/** Which colour family a status tag uses; each maps to the design's tag colours. */
export type Tone = 'running' | 'ok' | 'warn' | 'bad'

/** The status tag for a run: label + tone, from its status and (once finished) outcome. */
export function statusTag(
  status: RunStatus,
  outcome: RunOutcome | null,
): { label: string; tone: Tone } {
  if (status === 'running') return { label: 'Running', tone: 'running' }
  if (status === 'failed') return { label: 'Failed', tone: 'bad' }
  if (status === 'interrupted') return { label: 'Interrupted', tone: 'warn' }
  if (outcome === 'no_model') return { label: 'No model', tone: 'bad' }
  if (outcome === 'fallback') return { label: 'Finished · fallback', tone: 'warn' }
  return { label: 'Finished', tone: 'ok' }
}

const NEXT_STAGE: Record<string, string | null> = {
  profiler: 'router',
  router: 'sanity_check',
  sanity_check: 'fe_codegen',
  arena: 'artifacts',
  artifacts: 'reporter',
  reporter: 'finalize',
  finalize: null,
}

/**
 * The stage the agent is working on now, given the last stage it finished.
 * The corrective loop branches: a failed sandbox run goes back to codegen, or to the
 * fallback after the last attempt; a codegen call that produced nothing goes to the fallback.
 */
export function nextStage(last: StageEvent | null | undefined, maxAttempts = 3): string | null {
  if (!last) return 'profiler'
  if (last.node === 'sandbox') {
    if (last.ok) return 'arena'
    return last.attempt >= maxAttempts ? 'fallback_fe' : 'fe_codegen'
  }
  if (last.node === 'fe_codegen') return last.ok ? 'sandbox' : 'fallback_fe'
  if (last.node === 'fallback_fe') return last.ok ? 'arena' : 'reporter'
  return NEXT_STAGE[last.node] ?? null
}

/** "now", "5m", "3h", "2d": how long ago a run started. */
export function relativeTime(iso: string | null, now: Date = new Date()): string {
  if (!iso) return ''
  const minutes = Math.floor((now.getTime() - new Date(iso).getTime()) / 60_000)
  if (minutes < 1) return 'now'
  if (minutes < 60) return `${minutes}m`
  if (minutes < 60 * 24) return `${Math.floor(minutes / 60)}h`
  return `${Math.floor(minutes / (60 * 24))}d`
}

/** The second line of a sidebar row: what this run is doing, or how it ended. */
export function runSubline(run: RunListItem): string {
  if (run.status === 'running') {
    const stage = nextStage(run.last_event)
    return stage ? `Running · ${stageLabel(stage)}` : 'Running'
  }
  if (run.status === 'failed') return 'Failed'
  if (run.status === 'interrupted') return 'Interrupted'
  if (run.best_model_name && run.primary_metric && run.best_score !== null) {
    const metric = METRIC_LABELS[run.primary_metric] ?? run.primary_metric
    return `${run.best_model_name} · ${metric} ${formatMetric(run.best_score)}`
  }
  const task = run.task_type ? TASK_LABELS[run.task_type] ?? run.task_type : null
  return task ? `No model · ${task}` : 'No model'
}
