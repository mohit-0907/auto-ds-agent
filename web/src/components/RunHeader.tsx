// The top of the run screen: dataset, the request in quotes, and fact chips.

import { TASK_LABELS, clock, type RunFacts } from '../stages'

interface Props {
  dataset: string
  description: string | null
  facts: RunFacts
  /** The target the person asked for; null means the router chose it. */
  requestedTarget: string | null
  live: boolean
  /** Seconds since the run started (live), or its total duration (finished); null if unknown. */
  seconds: number | null
}

function shape(facts: RunFacts): string | null {
  if (facts.nRows === null || facts.nCols === null) return null
  return `${facts.nRows.toLocaleString('en-US')} × ${facts.nCols}`
}

function target(facts: RunFacts, requested: string | null): string | null {
  if (facts.targetColumn === null) return requested
  if (facts.targetColumn === '') return '—' // an out-of-scope decision names no target
  return requested ? facts.targetColumn : `${facts.targetColumn} (inferred)`
}

export function RunHeader({ dataset, description, facts, requestedTarget, live, seconds }: Props) {
  const time = seconds === null ? null : clock(seconds)
  const chips: [string, string | null][] = [
    ['task', facts.taskType ? TASK_LABELS[facts.taskType] ?? facts.taskType : null],
    ['target', target(facts, requestedTarget)],
    ['shape', shape(facts)],
    ['duration', time],
  ]

  return (
    <header className="flex flex-wrap items-end justify-between gap-6">
      <div className="flex min-w-0 flex-col gap-1.5">
        <h1 className="m-0 font-display text-[32px] font-normal tracking-[-0.015em] [overflow-wrap:anywhere]">
          {dataset}
        </h1>
        {description && <p className="m-0 text-[15px] text-muted">“{description}”</p>}
        <ul aria-label="Run facts" className="m-0 mt-2 flex list-none flex-wrap gap-2 p-0">
          {chips.map(([key, value]) => (
            <li
              key={key}
              className="inline-flex items-center gap-1.5 rounded-full border border-divider px-3 py-1 text-xs"
            >
              <span className="text-muted">{key}</span>
              {/* "…" until the stage that knows the value has finished */}
              <span className="font-mono text-[11.5px] font-medium">{value ?? '…'}</span>
            </li>
          ))}
        </ul>
      </div>
      {live && time && (
        <span className="font-mono text-[13px] font-medium text-muted">elapsed {time}</span>
      )}
    </header>
  )
}
