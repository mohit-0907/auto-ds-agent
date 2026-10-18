// The live stage timeline: one row per finished stage, retries grouped into attempt blocks.

import { groupTimeline, stageLabel } from '../stages'
import type { StageEvent } from '../types'

interface Props {
  events: StageEvent[]
  /** True while the run is still going: show a "working" row for the next stage. */
  live: boolean
}

function elapsed(first: StageEvent | undefined, event: StageEvent): string {
  if (!first) return ''
  const seconds = Math.max(0, (Date.parse(event.at) - Date.parse(first.at)) / 1000)
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function StageRow({ event, start }: { event: StageEvent; start: StageEvent | undefined }) {
  return (
    <li className="flex items-start gap-3 py-1.5" data-testid={`stage-${event.index}`}>
      <span
        aria-label={event.ok ? 'succeeded' : 'failed'}
        className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs font-bold text-white ${
          event.ok ? 'bg-emerald-500' : 'bg-red-500'
        }`}
      >
        {event.ok ? '✓' : '✗'}
      </span>
      <span className="w-36 shrink-0 font-medium">{stageLabel(event.node)}</span>
      <span className={`min-w-0 flex-1 break-words ${event.ok ? 'text-slate-600' : 'text-red-700'}`}>
        {event.text}
      </span>
      <span className="shrink-0 font-mono text-xs text-slate-400">{elapsed(start, event)}</span>
    </li>
  )
}

export function Timeline({ events, live }: Props) {
  const start = events[0]
  return (
    <ol className="text-sm">
      {groupTimeline(events).map((item) =>
        item.kind === 'stage' ? (
          // `key` lets React match rows between redraws, so existing rows are kept, not rebuilt.
          <StageRow key={item.event.index} event={item.event} start={start} />
        ) : (
          <li key={`attempt-${item.attempt}`} className="my-1.5 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Attempt {item.attempt}
            </p>
            <ol>
              {item.events.map((event) => (
                <StageRow key={event.index} event={event} start={start} />
              ))}
            </ol>
          </li>
        ),
      )}
      {live && (
        <li className="flex items-center gap-3 py-1.5 text-slate-500" aria-live="polite">
          <span className="h-5 w-5 shrink-0 animate-spin rounded-full border-2 border-slate-300 border-t-indigo-500" />
          Working…
        </li>
      )}
    </ol>
  )
}
