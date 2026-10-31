import { describe, expect, it } from 'vitest'
import { formatMetric, groupTimeline, nextStage, relativeTime, runSubline, statusTag } from './stages'
import type { RunListItem, StageEvent } from './types'

let next = 0
function ev(node: string, attempt = 0, ok = true): StageEvent {
  return { index: next++, node, ok, text: node, attempt, at: '2026-10-18T10:00:00' }
}

describe('groupTimeline', () => {
  it('groups each feature-engineering attempt into its own block', () => {
    next = 0
    const items = groupTimeline([
      ev('profiler'),
      ev('router'),
      ev('fe_codegen', 1),
      ev('sandbox', 1, false), // attempt 1 failed
      ev('fe_codegen', 2),
      ev('sandbox', 2), // attempt 2 passed
      ev('arena', 2),
    ])

    expect(items.map((i) => (i.kind === 'stage' ? i.event.node : `attempt ${i.attempt}`))).toEqual(
      ['profiler', 'router', 'attempt 1', 'attempt 2', 'arena'],
    )
    const firstAttempt = items[2]
    expect(firstAttempt.kind === 'attempt' && firstAttempt.events.map((e) => e.ok)).toEqual([
      true,
      false,
    ])
  })

  it('keeps a codegen call that never produced code (attempt 0) as a plain row', () => {
    next = 0
    const items = groupTimeline([ev('fe_codegen', 0, false), ev('fallback_fe')])
    expect(items.every((i) => i.kind === 'stage')).toBe(true)
  })
})

describe('formatMetric', () => {
  it('matches the Python report formatting', () => {
    expect(formatMetric(0.88153)).toBe('0.882')
    expect(formatMetric(28284.98)).toBe('28,285')
  })
})

describe('statusTag', () => {
  it('names the status, and the outcome once finished', () => {
    expect(statusTag('running', null)).toEqual({ label: 'Running', tone: 'running' })
    expect(statusTag('failed', null)).toEqual({ label: 'Failed', tone: 'bad' })
    expect(statusTag('interrupted', null)).toEqual({ label: 'Interrupted', tone: 'warn' })
    expect(statusTag('finished', 'model')).toEqual({ label: 'Finished', tone: 'ok' })
    expect(statusTag('finished', 'fallback')).toEqual({ label: 'Finished · fallback', tone: 'warn' })
    expect(statusTag('finished', 'no_model')).toEqual({ label: 'No model', tone: 'bad' })
  })
})

describe('nextStage', () => {
  it('starts at the profiler and follows the straight edges', () => {
    expect(nextStage(null)).toBe('profiler')
    expect(nextStage(ev('router'))).toBe('sanity_check')
    expect(nextStage(ev('arena'))).toBe('artifacts')
    expect(nextStage(ev('finalize'))).toBeNull()
  })

  it('follows the corrective loop like the graph does', () => {
    expect(nextStage(ev('fe_codegen', 1))).toBe('sandbox')
    expect(nextStage(ev('fe_codegen', 0, false))).toBe('fallback_fe') // no code produced
    expect(nextStage(ev('sandbox', 1, true))).toBe('arena')
    expect(nextStage(ev('sandbox', 1, false))).toBe('fe_codegen') // retry
    expect(nextStage(ev('sandbox', 3, false))).toBe('fallback_fe') // out of attempts
    expect(nextStage(ev('fallback_fe', 3, true))).toBe('arena')
    expect(nextStage(ev('fallback_fe', 3, false))).toBe('reporter') // early exit
  })
})

describe('relativeTime', () => {
  const now = new Date('2026-10-30T12:00:00')
  it('rounds down to the largest whole unit', () => {
    expect(relativeTime('2026-10-30T11:59:30', now)).toBe('now')
    expect(relativeTime('2026-10-30T11:48:00', now)).toBe('12m')
    expect(relativeTime('2026-10-30T09:01:00', now)).toBe('2h')
    expect(relativeTime('2026-10-27T12:00:00', now)).toBe('3d')
    expect(relativeTime(null, now)).toBe('')
  })
})

describe('runSubline', () => {
  const base: RunListItem = {
    run_id: 'r',
    status: 'finished',
    outcome: 'model',
    dataset: 'mpg.csv',
    description: 'd',
    target_column: null,
    created_at: null,
    task_type: 'regression',
    best_model_id: 'hgb',
    best_model_name: 'Hist Gradient Boosting',
    primary_metric: 'rmse',
    best_score: 2.8712,
    last_event: null,
  }

  it('shows the winner and its score for a finished run', () => {
    expect(runSubline(base)).toBe('Hist Gradient Boosting · RMSE 2.871')
  })

  it('shows the stage being worked on for a running run', () => {
    const running = { ...base, status: 'running' as const, last_event: ev('sandbox', 1, false) }
    expect(runSubline(running)).toBe('Running · Write features')
  })

  it('falls back to the task when there is no model', () => {
    const none = { ...base, outcome: 'no_model' as const, best_model_id: null, best_model_name: null, best_score: null }
    expect(runSubline(none)).toBe('No model · Regression')
  })
})
