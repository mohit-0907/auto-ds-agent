// URL "/runs": every run, newest first. Each row links to the run's own page.

import { useEffect, useState } from 'react'
import { Link } from 'react-router'
import { listRuns } from '../api'
import type { RunListItem } from '../types'
import { StatusPill } from '../components/StatusPill'

function when(created: string | null): string {
  return created ? new Date(created).toLocaleString() : '–'
}

export function HistoryPage() {
  const [runs, setRuns] = useState<RunListItem[] | null>(null) // null = still loading
  const [error, setError] = useState<string | null>(null)

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

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="mb-4 text-base font-semibold">Run history</h2>
      {error && <p role="alert" className="text-sm text-red-700">{error}</p>}
      {!error && runs === null && <p className="text-sm text-slate-500">Loading…</p>}
      {runs?.length === 0 && (
        <p className="text-sm text-slate-500">
          No runs yet. <Link to="/" className="text-indigo-600 hover:underline">Start one</Link>.
        </p>
      )}
      {runs && runs.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
                <th className="py-2 pr-3 font-medium">Started</th>
                <th className="py-2 pr-3 font-medium">Dataset</th>
                <th className="py-2 pr-3 font-medium">Task</th>
                <th className="py-2 pr-3 font-medium">Best model</th>
                <th className="py-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run_id} className="border-b border-slate-100 hover:bg-slate-50">
                  <td className="py-2 pr-3 text-slate-500">{when(run.created_at)}</td>
                  <td className="py-2 pr-3">
                    <Link to={`/runs/${run.run_id}`} className="font-medium text-indigo-700 hover:underline">
                      {run.dataset}
                    </Link>
                  </td>
                  <td className="py-2 pr-3 text-slate-600">{run.task_type ?? '–'}</td>
                  <td className="py-2 pr-3 text-slate-600">{run.best_model_id ?? '–'}</td>
                  <td className="py-2">
                    <StatusPill status={run.status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
