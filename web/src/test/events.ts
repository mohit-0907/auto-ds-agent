// Build a StageEvent for tests: every field the server sends, with the ones under test overridden.

import type { StageEvent } from '../types'

export function stageEvent(overrides: Partial<StageEvent> & { node: string }): StageEvent {
  return {
    index: 0,
    ok: true,
    text: overrides.node,
    attempt: 0,
    at: '2026-10-18T10:00:00',
    detail: null,
    detail_kind: null,
    n_rows: null,
    n_cols: null,
    task_type: null,
    target_column: null,
    ...overrides,
  }
}
