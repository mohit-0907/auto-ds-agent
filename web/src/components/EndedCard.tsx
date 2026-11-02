// The Results tab when there is no model to show: out of scope, nothing trained, crashed,
// or interrupted. One card, worded from the run's status and its own warnings / error.

import { TriangleAlert } from 'lucide-react'
import { Link } from 'react-router'
import { ending } from '../results'
import type { RunSummary } from '../types'

const TONE = {
  bad: { circle: 'bg-bad-bg text-bad-fg', kicker: 'text-bad-fg' },
  warn: { circle: 'bg-warn-bg text-warn-fg', kicker: 'text-warn-fg' },
}

interface Props {
  summary: RunSummary
  onOpenReport: () => void
}

export function EndedCard({ summary, onOpenReport }: Props) {
  const { kicker, title, text, tone } = ending(summary)
  return (
    <section aria-label="Result" className="flex items-start gap-5 rounded-[28px] bg-card p-7 shadow-card">
      <span className={`flex h-12 w-12 flex-none items-center justify-center rounded-full ${TONE[tone].circle}`}>
        <TriangleAlert size={22} strokeWidth={2.75} aria-hidden="true" />
      </span>
      <div className="flex min-w-0 flex-col gap-2">
        <span className={`text-[11px] font-semibold tracking-[0.08em] uppercase ${TONE[tone].kicker}`}>{kicker}</span>
        <h2 className="m-0 font-display text-2xl font-normal">{title}</h2>
        <p className="m-0 max-w-[68ch] text-muted text-pretty [overflow-wrap:anywhere]">{text}</p>
        <div className="mt-2 flex flex-wrap gap-2.5">
          {summary.report_markdown && (
            <button
              type="button"
              onClick={onOpenReport}
              className="cursor-pointer rounded-full border border-divider bg-transparent px-[18px] py-[9px] font-display text-sm text-ink hover:bg-sel"
            >
              Read the report
            </button>
          )}
          <Link
            to="/"
            className="rounded-full bg-accent px-[18px] py-[9px] font-display text-sm text-on-accent no-underline hover:brightness-[.93]"
          >
            Try another dataset
          </Link>
        </div>
      </div>
    </section>
  )
}
