import { describe, expect, it } from 'vitest'
import { formatMetric, groupTimeline } from './stages'
import type { StageEvent } from './types'

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
