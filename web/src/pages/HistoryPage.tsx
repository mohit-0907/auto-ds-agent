// URL "/runs": every run, newest first, with a search box and a status filter.
// Each row opens the run's own page.

import { Search } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router'
import { listRuns } from '../api'
import { StatusTag } from '../components/StatusTag'
import { metricLabel } from '../results'
import { TASK_LABELS, filterRuns, formatMetric, type RunFilter } from '../stages'
import type { RunListItem } from '../types'

const FILTERS: { id: RunFilter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'running', label: 'Running' },
  { id: 'finished', label: 'Finished' },
  { id: 'no_model', label: 'No model' },
]

const TH = 'border-b border-divider px-3 py-2.5 text-left text-[11px] font-semibold tracking-[0.08em] whitespace-nowrap text-muted uppercase'
const TD = 'border-b border-divider px-3 py-3'

/** "Oct 18, 23:23", in the browser's time zone. */
function started(created: string | null): string {
  if (!created) return '–'
  return new Date(created).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
}

export function HistoryPage() {
  const [runs, setRuns] = useState<RunListItem[] | null>(null) // null = still loading
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<RunFilter>('all')
  const navigate = useNavigate()

  useEffect(() => {
    let cancelled = false
    listRuns()
      .then((result) => {
        if (!cancelled) setRuns(result)
      })
      .catch(() => {
        if (!cancelled) setError('Could not load the run history.')
      })
    return () => {
      cancelled = true
    }
  }, []) // [] = run once, when the page opens

  // Worked out on every render from the list + what the person typed; nothing stored twice.
  const shown = runs === null ? [] : filterRuns(runs, query, filter)

  return (
    <div className="flex max-w-[1180px] flex-col gap-6 px-4 pt-8 pb-16 min-[760px]:px-9">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h1 className="m-0 font-display text-[32px] font-normal tracking-[-0.015em]">All runs</h1>
          <p className="m-0 text-muted">Every run keeps its report, leaderboard, feature code and saved model.</p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 rounded-full border border-divider bg-card px-3.5 py-2 focus-within:outline-2 focus-within:outline-accent">
            <Search size={15} strokeWidth={2.75} className="text-muted" aria-hidden="true" />
            <input
              type="search"
              aria-label="Search datasets"
              placeholder="Search datasets"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-44 border-none bg-transparent text-[13px] text-ink outline-none placeholder:text-muted"
            />
          </label>
          <div role="group" aria-label="Status filter" className="flex gap-0.5 rounded-full border border-divider p-[3px]">
            {FILTERS.map(({ id, label }) => (
              <button
                key={id}
                type="button"
                aria-pressed={filter === id}
                onClick={() => setFilter(id)}
                className={`cursor-pointer rounded-full border-none px-3 py-[5px] text-xs font-semibold ${
                  filter === id ? 'bg-accent text-on-accent' : 'bg-transparent text-muted hover:bg-sel'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {error && <p role="alert" className="m-0 text-bad-fg">{error}</p>}
      {!error && runs === null && <p className="m-0 text-muted">Loading…</p>}
      {runs?.length === 0 && (
        <p className="m-0 text-muted">
          No runs yet.{' '}
          <Link to="/" className="font-semibold text-accent-fg">
            Start one
          </Link>
          .
        </p>
      )}

      {runs && runs.length > 0 && (
        <section aria-label="Runs" className="overflow-x-auto rounded-[28px] bg-card px-3 py-2 shadow-card">
          <table className="w-full min-w-[720px] border-collapse text-[13px]">
            <thead>
              <tr>
                <th className={TH}>Dataset</th>
                <th className={TH}>Task</th>
                <th className={TH}>Target</th>
                <th className={TH}>Best model</th>
                <th className={TH}>Score</th>
                <th className={TH}>Status</th>
                <th className={TH}>Started</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((run) => (
                // The whole row opens the run; the dataset link is there for keyboards and
                // screen readers (a table row itself cannot be focused or announced as a link).
                <tr key={run.run_id} onClick={() => navigate(`/runs/${run.run_id}`)} className="cursor-pointer hover:bg-sel">
                  <td className={`${TD} max-w-[320px]`}>
                    <Link
                      to={`/runs/${run.run_id}`}
                      // Without this the click also reaches the row, which navigates a second
                      // time: two history entries, so Back would seem to do nothing.
                      onClick={(e) => e.stopPropagation()}
                      className="block truncate font-semibold text-ink no-underline"
                    >
                      {run.dataset}
                    </Link>
                    <span className="block truncate text-xs text-muted">{run.description}</span>
                  </td>
                  <td className={`${TD} whitespace-nowrap`}>{run.task_type ? (TASK_LABELS[run.task_type] ?? run.task_type) : '–'}</td>
                  <td className={`${TD} font-mono text-[12.5px]`}>
                    {run.target_column ?? <span className="font-sans text-muted">inferred</span>}
                  </td>
                  <td className={`${TD} whitespace-nowrap`}>{run.best_model_name ?? '–'}</td>
                  <td className={`${TD} whitespace-nowrap`}>
                    {run.best_score !== null && run.primary_metric ? (
                      <>
                        <span className="text-muted">{metricLabel(run.primary_metric)}</span>{' '}
                        <span className="font-mono">{formatMetric(run.best_score)}</span>
                      </>
                    ) : (
                      '–'
                    )}
                  </td>
                  <td className={TD}>
                    <StatusTag status={run.status} outcome={run.outcome} />
                  </td>
                  <td className={`${TD} whitespace-nowrap text-muted`}>{started(run.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {shown.length === 0 && <p className="m-0 px-3 py-6 text-center text-muted">No runs match.</p>}
        </section>
      )}
    </div>
  )
}
