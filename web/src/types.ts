// The shapes of the API's JSON replies. Hand-written mirrors of the Pydantic models in
// src/autods (progress.py StageEvent, state.py ModelResult and Selection, api/runs.py
// RunSummary, DownloadFile and RunListItem, api/app.py RunCreated). If a model changes in Python, change it here too.

export type RunStatus = 'running' | 'finished' | 'failed' | 'interrupted'

/** How a finished run ended (null until it finishes). */
export type RunOutcome = 'model' | 'fallback' | 'no_model'

/** One finished stage of the agent, as shown in the live timeline. */
export interface StageEvent {
  index: number // 0, 1, 2, ... in the order stages finished
  node: string // e.g. "sandbox"
  ok: boolean
  text: string // e.g. "attempt 1 failed: KeyError: 'Deck'"
  attempt: number
  at: string // ISO timestamp
  detail: string | null // what an expanded step shows: code, a decision, a traceback
  detail_kind: 'code' | 'error' | null
  // Header facts, set only by the stage that learns them (profiler; router / sanity check)
  n_rows: number | null
  n_cols: number | null
  task_type: string | null
  target_column: string | null
}

/** One model's cross-validated scores from the arena. */
export interface ModelResult {
  model_id: string
  model_name: string
  status: 'success' | 'failed'
  metrics: Record<string, number> // e.g. { roc_auc: 0.881, accuracy: 0.839 }
  metrics_std: Record<string, number>
  primary_metric: string
  fold_scores: number[]
  training_time_s: number
  predict_time_s: number
  error: string | null
}

/** The selection rule's test on the chosen model and one other (state.py PairComparison). */
export interface PairComparison {
  model_id: string // the other model
  model_name: string
  better_mean: boolean // the other model's mean beat the chosen one's
  folds_won: number // folds won by whichever of the two had the better mean
  clearly: boolean // that better-mean model won enough folds AND on average
  tested: boolean // select_best itself ran this comparison (only top vs simpler ones)
}

/** The chosen model's lead over the dummy baseline (state.py BaselineLift). */
export interface BaselineLift {
  baseline_name: string
  baseline_score: number
  kind: 'difference' | 'percent' // difference: +0.379 ROC-AUC; percent: 64.0% less error
  value: number
}

/** Why the arena chose its model, recorded when it chose (state.py Selection). */
export interface Selection {
  chosen_model_id: string
  top_model_id: string // the best mean
  n_folds: number
  min_fold_wins: number
  comparisons: PairComparison[] // leaderboard order
  baseline: BaselineLift | null
}

/** A file the server will serve via /files/{name} (api/runs.py DownloadFile). */
export interface DownloadFile {
  name: string
  size_bytes: number
}

/** Reply to POST /api/runs. */
export interface RunCreated {
  run_id: string
  status: string
}

/** Reply to GET /api/runs/{id}: everything about one run. */
export interface RunSummary {
  run_id: string
  status: RunStatus
  outcome: RunOutcome | null
  dataset: string
  description: string
  target_column: string | null
  created_at: string | null
  events: StageEvent[]
  task_type: string | null
  best_model_id: string | null
  leaderboard: ModelResult[]
  selection: Selection | null // null for runs saved before Task 7.8
  report_markdown: string | null
  token_cost_usd: number
  warnings: string[]
  error: string | null
  files: string[]
  downloads: DownloadFile[]
}

/** One row of GET /api/runs (history page and sidebar). */
export interface RunListItem {
  run_id: string
  status: RunStatus
  outcome: RunOutcome | null
  dataset: string
  description: string
  target_column: string | null
  created_at: string | null
  task_type: string | null
  best_model_id: string | null
  best_model_name: string | null
  primary_metric: string | null
  best_score: number | null // computed by the arena, displayed as-is
  last_event: StageEvent | null // only while running
}

/** One row of GET /api/samples. */
export interface SampleDataset {
  name: string
  rows: number
  kind: string // "binary" | "multiclass" | "regression"
  description: string
  target: string // "" = let the agent infer it
}
