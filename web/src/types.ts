// The shapes of the API's JSON replies. Hand-written mirrors of the Pydantic models in
// src/autods (progress.py StageEvent, state.py ModelResult, api/runs.py RunSummary and
// RunListItem, api/app.py RunCreated). If a model changes in Python, change it here too.

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
  report_markdown: string | null
  token_cost_usd: number
  warnings: string[]
  error: string | null
  files: string[]
  downloads: string[] // the files the server will serve via /files/{name}
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
