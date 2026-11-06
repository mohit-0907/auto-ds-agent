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

/** Mirrors MAX_FE_ATTEMPTS in src/autods/config.py: change both together. */
export const MAX_FE_ATTEMPTS = 3

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
export function nextStage(
  last: StageEvent | null | undefined,
  maxAttempts = MAX_FE_ATTEMPTS,
): string | null {
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

// --- Run screen: header facts, pipeline graph, captions (Task 7.7) ----------------------

/** "00:38", "01:05", "12:30": a duration in seconds as mm:ss (minutes keep growing past 59). */
export function clock(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds))
  const m = Math.floor(whole / 60)
  const s = whole % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

/** Seconds from `startIso` to `iso` (both from the server's clock). */
export function secondsBetween(startIso: string, iso: string): number {
  return (Date.parse(iso) - Date.parse(startIso)) / 1000
}

export interface RunFacts {
  taskType: string | null
  targetColumn: string | null
  nRows: number | null
  nCols: number | null
}

/**
 * The header's facts, read from the typed fields of the events that carry them. The
 * latest value wins, so a task type corrected by the sanity check replaces the router's.
 */
export function runFacts(events: StageEvent[]): RunFacts {
  const facts: RunFacts = { taskType: null, targetColumn: null, nRows: null, nCols: null }
  for (const event of events) {
    if (event.task_type != null) facts.taskType = event.task_type
    if (event.target_column != null) facts.targetColumn = event.target_column
    if (event.n_rows != null) facts.nRows = event.n_rows
    if (event.n_cols != null) facts.nCols = event.n_cols
  }
  return facts
}

/** The nodes drawn in the pipeline graph, in order (finalize is not drawn). */
export const GRAPH_NODES = [
  'profiler',
  'router',
  'sanity_check',
  'fe_codegen',
  'sandbox',
  'arena',
  'artifacts',
  'reporter',
] as const
export type GraphNode = (typeof GRAPH_NODES)[number]

export type NodeState = 'done' | 'fail' | 'active' | 'pending' | 'skipped'

export interface PipelineState {
  nodes: Record<GraphNode, NodeState>
  /** The corrective loop's latest attempt (0 = no code written yet). */
  attempt: number
  /** The fallback pill: used (warn), failed, active (running now), or idle. */
  fallback: 'used' | 'failed' | 'active' | 'idle'
}

/**
 * Every node's state in the pipeline graph, derived from the events alone.
 *
 * The node the agent is on now (live only) is `active`; a node with events is `done` or
 * `fail` by its latest event; a node with none is `pending` while live, and `skipped`
 * once the run is over (the graph never went there).
 */
export function pipelineState(events: StageEvent[], live: boolean): PipelineState {
  const active = live ? nextStage(events[events.length - 1]) : null
  const nodes = {} as Record<GraphNode, NodeState>
  for (const node of GRAPH_NODES) {
    const own = events.filter((e) => e.node === node)
    if (node === active) nodes[node] = 'active'
    else if (own.length === 0) nodes[node] = live ? 'pending' : 'skipped'
    else nodes[node] = own[own.length - 1].ok ? 'done' : 'fail'
  }
  const fallbackEvent = events.find((e) => e.node === 'fallback_fe')
  let fallback: PipelineState['fallback'] = 'idle'
  if (fallbackEvent) fallback = fallbackEvent.ok ? 'used' : 'failed'
  else if (active === 'fallback_fe') fallback = 'active'
  const attempt = Math.max(0, ...events.map((e) => e.attempt))
  return { nodes, attempt, fallback }
}

// How many stages are left on the straight path to the end, counting the node itself:
// lets the live caption estimate the total ("5 of ~9") even after retries add stages.
const STAGES_LEFT: Record<string, number> = {
  profiler: 9,
  router: 8,
  sanity_check: 7,
  fe_codegen: 6,
  sandbox: 5,
  fallback_fe: 5,
  arena: 4,
  artifacts: 3,
  reporter: 2,
  finalize: 1,
}

/** The caption beside "PIPELINE": live progress, or how the run ended. */
export function graphCaption(
  events: StageEvent[],
  status: RunStatus,
  outcome: RunOutcome | null,
  live: boolean,
  durationSeconds: number | null,
): string {
  if (live) {
    const next = nextStage(events[events.length - 1])
    const total = events.length + (next ? STAGES_LEFT[next] ?? 0 : 0)
    return `Live · ${events.length} of ~${total} stages`
  }
  if (status === 'failed') return 'The run crashed'
  if (status === 'interrupted') return 'Stopped before finishing'
  if (outcome === 'no_model') {
    // Where it stopped: the last stage before the report-writing tail.
    const stop = [...events].reverse().find((e) => !['reporter', 'finalize'].includes(e.node))
    return stop ? `Stopped at ${stageLabel(stop.node)}` : 'Stopped without a model'
  }
  if (outcome === 'fallback') return 'Recovered with the fallback recipe'
  return durationSeconds === null ? 'Completed' : `Completed in ${clock(durationSeconds)}`
}

export type AttemptStatus = 'passed' | 'failed' | 'in progress'

/** An attempt block's tag: its sandbox verdict, or in progress while the agent is on it. */
export function attemptStatus(events: StageEvent[], live: boolean): AttemptStatus {
  const sandbox = events.find((e) => e.node === 'sandbox')
  if (sandbox) return sandbox.ok ? 'passed' : 'failed'
  return live ? 'in progress' : 'failed'
}

// --- Feature code tab and the All runs page (Task 7.9) ------------------------------------

/** The Feature code tab's tag: which attempt produced the code, or the fallback recipe. */
export function codeTag(events: StageEvent[], outcome: RunOutcome | null): { label: string; tone: Tone } | null {
  if (outcome === 'fallback' || events.some((e) => e.node === 'fallback_fe' && e.ok)) {
    return { label: 'fallback recipe', tone: 'warn' }
  }
  const passed = [...events].reverse().find((e) => e.node === 'sandbox' && e.ok)
  return passed ? { label: `attempt ${passed.attempt} · passed validation`, tone: 'ok' } : null
}

export type RunFilter = 'all' | 'running' | 'finished' | 'no_model'

/** Which status filter a run falls under. "No model" is every run that ended without one. */
export function runFilterOf(run: RunListItem): Exclude<RunFilter, 'all'> {
  if (run.status === 'running') return 'running'
  if (run.status === 'finished' && run.outcome !== 'no_model') return 'finished'
  return 'no_model' // no model chosen, crashed, or interrupted
}

/** The All runs table's rows: matching the search (dataset or description) and the filter. */
export function filterRuns(runs: RunListItem[], query: string, filter: RunFilter): RunListItem[] {
  const q = query.trim().toLowerCase()
  return runs.filter(
    (run) =>
      (filter === 'all' || runFilterOf(run) === filter) &&
      (q === '' || run.dataset.toLowerCase().includes(q) || run.description.toLowerCase().includes(q)),
  )
}
