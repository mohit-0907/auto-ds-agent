import { describe, expect, it } from 'vitest'
import {
  attemptStatus,
  clock,
  formatMetric,
  graphCaption,
  groupTimeline,
  nextStage,
  pipelineState,
  relativeTime,
  runFacts,
  runSubline,
  statusTag,
} from './stages'
import { stageEvent } from './test/events'
import type { RunListItem, StageEvent } from './types'

let next = 0
function ev(node: string, attempt = 0, ok = true): StageEvent {
  return stageEvent({ index: next++, node, ok, attempt })
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

describe('clock', () => {
  it('formats seconds as mm:ss, with minutes past an hour', () => {
    expect(clock(0)).toBe('00:00')
    expect(clock(38.9)).toBe('00:38')
    expect(clock(65)).toBe('01:05')
    expect(clock(3725)).toBe('62:05')
    expect(clock(-3)).toBe('00:00') // clocks a hair apart never show negative time
  })
})

describe('runFacts', () => {
  it('reads the typed fields, the latest value winning', () => {
    const facts = runFacts([
      stageEvent({ node: 'profiler', n_rows: 891, n_cols: 12 }),
      stageEvent({ node: 'router', task_type: 'regression', target_column: 'Survived' }),
      stageEvent({ node: 'sanity_check', task_type: 'binary_classification', target_column: 'Survived' }),
      stageEvent({ node: 'arena' }), // no facts: leaves the others alone
    ])
    expect(facts).toEqual({
      taskType: 'binary_classification',
      targetColumn: 'Survived',
      nRows: 891,
      nCols: 12,
    })
  })

  it('is all null before the profiler finishes', () => {
    expect(runFacts([])).toEqual({ taskType: null, targetColumn: null, nRows: null, nCols: null })
  })
})

describe('pipelineState', () => {
  const head = () => [ev('profiler'), ev('router'), ev('sanity_check')]

  it('marks the stage in progress active and the rest pending while live', () => {
    next = 0
    const { nodes, attempt, fallback } = pipelineState(head(), true)
    expect(nodes.profiler).toBe('done')
    expect(nodes.sanity_check).toBe('done')
    expect(nodes.fe_codegen).toBe('active')
    expect(nodes.sandbox).toBe('pending')
    expect(nodes.reporter).toBe('pending')
    expect(attempt).toBe(0)
    expect(fallback).toBe('idle')
  })

  it('shows a retry: the failed sandbox stays red while codegen tries again', () => {
    next = 0
    const events = [...head(), ev('fe_codegen', 1), ev('sandbox', 1, false)]
    const { nodes, attempt } = pipelineState(events, true)
    expect(nodes.sandbox).toBe('fail')
    expect(nodes.fe_codegen).toBe('active')
    expect(attempt).toBe(1)
  })

  it('a repaired run ends with every node done', () => {
    next = 0
    const events = [
      ...head(),
      ev('fe_codegen', 1),
      ev('sandbox', 1, false),
      ev('fe_codegen', 2),
      ev('sandbox', 2),
      ev('arena', 2),
      ev('artifacts', 2),
      ev('reporter', 2),
      ev('finalize', 2),
    ]
    const { nodes, attempt, fallback } = pipelineState(events, false)
    expect(Object.values(nodes).every((state) => state === 'done')).toBe(true)
    expect(attempt).toBe(2)
    expect(fallback).toBe('idle')
  })

  it('three failures: the fallback becomes active, then used', () => {
    next = 0
    const failures = [1, 2, 3].flatMap((n) => [ev('fe_codegen', n), ev('sandbox', n, false)])
    expect(pipelineState([...head(), ...failures], true).fallback).toBe('active')

    const after = [...head(), ...failures, ev('fallback_fe', 3), ev('arena', 3)]
    const state = pipelineState(after, true)
    expect(state.fallback).toBe('used')
    expect(state.nodes.sandbox).toBe('fail')
    expect(state.nodes.artifacts).toBe('active')
  })

  it('an early exit leaves the unvisited nodes skipped', () => {
    next = 0
    const events = [...head(), ev('reporter'), ev('finalize')]
    const { nodes } = pipelineState(events, false)
    expect(nodes.fe_codegen).toBe('skipped')
    expect(nodes.arena).toBe('skipped')
    expect(nodes.reporter).toBe('done')
  })
})

describe('graphCaption', () => {
  it('estimates the total while live, growing with retries', () => {
    next = 0
    const five = [ev('profiler'), ev('router'), ev('sanity_check'), ev('fe_codegen', 1), ev('sandbox', 1)]
    expect(graphCaption(five, 'running', null, true, null)).toBe('Live · 5 of ~9 stages')
    // A failed sandbox means another codegen + sandbox: the estimate grows to 11.
    const retry = [...five.slice(0, 4), ev('sandbox', 1, false)]
    expect(graphCaption(retry, 'running', null, true, null)).toBe('Live · 5 of ~11 stages')
  })

  it('says how a finished run ended', () => {
    next = 0
    expect(graphCaption([ev('finalize')], 'finished', 'model', false, 60)).toBe('Completed in 01:00')
    expect(graphCaption([], 'finished', 'fallback', false, 60)).toBe(
      'Recovered with the fallback recipe',
    )
    const stopped = [ev('profiler'), ev('router'), ev('sanity_check'), ev('reporter'), ev('finalize')]
    expect(graphCaption(stopped, 'finished', 'no_model', false, 9)).toBe('Stopped at Check decision')
    expect(graphCaption([], 'failed', null, false, null)).toBe('The run crashed')
  })
})

describe('attemptStatus', () => {
  it("uses the attempt's sandbox verdict, or in progress while live", () => {
    next = 0
    expect(attemptStatus([ev('fe_codegen', 1), ev('sandbox', 1)], false)).toBe('passed')
    expect(attemptStatus([ev('fe_codegen', 1), ev('sandbox', 1, false)], false)).toBe('failed')
    expect(attemptStatus([ev('fe_codegen', 2)], true)).toBe('in progress')
  })
})
