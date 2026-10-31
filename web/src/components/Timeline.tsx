// The Steps tab: one row per finished stage, retries grouped into attempt blocks. A row with
// a detail (code, decision, traceback) is a button that opens it underneath.

import { Check, ChevronRight, X } from 'lucide-react'
import {
  attemptStatus,
  clock,
  groupTimeline,
  nextStage,
  secondsBetween,
  stageLabel,
  type AttemptStatus,
} from '../stages'
import type { StageEvent } from '../types'
import { Spinner } from './Spinner'

interface Props {
  events: StageEvent[]
  /** True while the run is still going: show a "working…" row for the next stage. */
  live: boolean
  /** When the run started (server clock); step times count from here. */
  startIso: string | null
  /** Which rows are open, by event index. Kept by the page, so switching tabs keeps it. */
  open: Set<number>
  onToggle: (index: number) => void
}

const GRID = 'grid w-full grid-cols-[24px_var(--label-col)_minmax(0,1fr)_auto_18px] items-center gap-3 rounded-2xl px-2.5 py-[9px] text-left text-[13.5px]'

function StatusCircle({ event }: { event: StageEvent }) {
  return (
    <span
      aria-label={event.ok ? 'succeeded' : 'failed'}
      className={`flex h-6 w-6 items-center justify-center rounded-full ${
        event.ok ? 'bg-ok-bg text-ok-fg' : 'bg-bad-bg text-bad-fg'
      }`}
    >
      {event.ok ? <Check size={13} strokeWidth={3} /> : <X size={13} strokeWidth={3} />}
    </span>
  )
}

interface RowProps {
  event: StageEvent
  startIso: string | null
  isOpen: boolean
  onToggle: (index: number) => void
}

function StepRow({ event, startIso, isOpen, onToggle }: RowProps) {
  const time = startIso ? clock(secondsBetween(startIso, event.at)) : ''
  const detailId = `step-detail-${event.index}`
  const cells = (
    <>
      <StatusCircle event={event} />
      <span className="font-semibold">{stageLabel(event.node)}</span>
      <span className={`[overflow-wrap:anywhere] ${event.ok ? 'text-muted' : 'text-bad-fg'}`}>
        {event.text}
      </span>
      <span className="font-mono text-xs font-medium text-muted">{time}</span>
      <ChevronRight
        size={16}
        strokeWidth={2.75}
        aria-hidden="true"
        className={`text-muted transition-transform duration-150 ${isOpen ? 'rotate-90' : ''} ${
          event.detail ? '' : 'invisible'
        }`}
      />
    </>
  )

  return (
    <li className="flex flex-col" data-testid={`stage-${event.index}`}>
      {event.detail ? (
        // A real <button>: reachable with Tab, opens with Enter/Space, and aria-expanded
        // tells a screen reader whether the detail below is showing.
        <button
          type="button"
          aria-expanded={isOpen}
          aria-controls={detailId}
          onClick={() => onToggle(event.index)}
          className={`${GRID} cursor-pointer hover:bg-sel`}
        >
          {cells}
        </button>
      ) : (
        <div className={GRID}>{cells}</div>
      )}
      {event.detail && isOpen && (
        <pre
          id={detailId}
          className={`mx-2.5 mt-1 mb-2.5 ml-[46px] overflow-x-auto rounded-2xl bg-code px-4 py-3.5 font-mono text-xs leading-[1.6] whitespace-pre-wrap ${
            event.detail_kind === 'error' ? 'text-bad-fg' : 'text-ink'
          }`}
        >
          {event.detail}
        </pre>
      )}
    </li>
  )
}

const ATTEMPT_TAG: Record<AttemptStatus, string> = {
  passed: 'bg-ok-bg text-ok-fg',
  failed: 'bg-bad-bg text-bad-fg',
  'in progress': 'bg-accent-bg text-accent-fg',
}

/** The row for the stage the agent is on now. */
function WorkingRow({ node }: { node: string | null }) {
  return (
    <li className={GRID}>
      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-accent-bg">
        <Spinner />
      </span>
      <span className="font-semibold">{node ? stageLabel(node) : 'Working'}</span>
      <span className="text-muted">working…</span>
      <span />
      <span />
    </li>
  )
}

export function Timeline({ events, live, startIso, open, onToggle }: Props) {
  const items = groupTimeline(events)
  const row = (event: StageEvent) => (
    // `key` lets React match rows between redraws, so existing rows are kept, not rebuilt.
    <StepRow
      key={event.index}
      event={event}
      startIso={startIso}
      isOpen={open.has(event.index)}
      onToggle={onToggle}
    />
  )
  // The block for the attempt in progress is the last item while the loop is running.
  const lastItem = items[items.length - 1]

  return (
    <section aria-label="Steps" className="rounded-[28px] bg-card px-4 py-3.5 shadow-card">
      {/* The label column is 150 px, 140 px inside an attempt block (it is indented). */}
      <ol className="m-0 flex list-none flex-col gap-0.5 p-0 [--label-col:150px]">
        {items.map((item) => {
          if (item.kind === 'stage') return row(item.event)
          const status = attemptStatus(item.events, live && item === lastItem)
          return (
            <li
              key={`attempt-${item.attempt}`}
              className={`my-1.5 rounded-[22px] border-[1.5px] bg-page p-1.5 [--label-col:140px] ${
                status === 'in progress' ? 'border-accent' : 'border-divider'
              }`}
            >
              <div className="flex items-center gap-2.5 px-2.5 pt-1.5 pb-1">
                <span className="text-[11px] font-bold tracking-[0.08em] text-muted uppercase">
                  Attempt {item.attempt}
                </span>
                <span className={`rounded-full px-[9px] py-px text-[11px] font-semibold ${ATTEMPT_TAG[status]}`}>
                  {status}
                </span>
              </div>
              <ol className="m-0 flex list-none flex-col p-0">{item.events.map(row)}</ol>
            </li>
          )
        })}
        {live && <WorkingRow node={nextStage(events[events.length - 1])} />}
      </ol>
      {live && (
        <p className="mx-2.5 my-1.5 text-xs text-muted" aria-live="polite">
          Streaming live from the agent. Results appear here when the run finishes.
        </p>
      )}
    </section>
  )
}
