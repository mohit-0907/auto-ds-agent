// URL "/runs/:runId": one run, live while it runs, with a result card once it ends.

import { useEffect, useState, type ReactNode } from 'react'
import { Link, useParams } from 'react-router'
import { ApiError, getRun } from '../api'
import { Downloads } from '../components/Downloads'
import { Leaderboard } from '../components/Leaderboard'
import { ReportView } from '../components/ReportView'
import { StatusTag } from '../components/StatusTag'
import { Timeline } from '../components/Timeline'
import { useRunEvents } from '../hooks/useRunEvents'
import { METRIC_LABELS, formatMetric } from '../stages'
import type { RunStatus, RunSummary } from '../types'

export function RunPage() {
  const { runId = '' } = useParams() // the ":runId" part of the URL
  // key={runId}: moving to a different run gives a brand-new RunView with fresh state,
  // instead of the old run's events lingering while the new ones arrive.
  return <RunView key={runId} runId={runId} />
}

function RunView({ runId }: { runId: string }) {
  const { events, stream, endStatus } = useRunEvents(runId)
  const [summary, setSummary] = useState<RunSummary | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const ended = stream === 'ended'

  // Fetch the run's details when the page opens, and once more when the stream ends --
  // by then run.json and report.md exist, so the summary includes the results.
  useEffect(() => {
    let cancelled = false // ignore a reply that arrives after we have left this page
    getRun(runId)
      .then((result) => {
        if (!cancelled) setSummary(result)
      })
      .catch((err) => {
        if (!cancelled) {
          setLoadError(
            err instanceof ApiError && err.status === 404
              ? 'There is no run with this ID.'
              : 'Could not load this run.',
          )
        }
      })
    return () => {
      cancelled = true
    }
  }, [runId, ended])

  if (loadError) {
    return (
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <p role="alert" className="text-red-700">{loadError}</p>
        <Link to="/" className="mt-3 inline-block text-sm text-indigo-600 hover:underline">
          Start a new run
        </Link>
      </section>
    )
  }

  const status: RunStatus = endStatus ?? summary?.status ?? 'running'
  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h2 className="truncate text-base font-semibold">{summary?.dataset ?? runId}</h2>
            {summary && <p className="mt-0.5 text-sm text-slate-500">“{summary.description}”</p>}
          </div>
          <StatusTag status={status} outcome={summary?.outcome ?? null} />
        </div>
        <Timeline events={events} live={!ended && stream !== 'error'} />
        {stream === 'error' && (
          <p role="alert" className="mt-3 text-sm text-red-700">
            Lost the connection to the live progress. Refresh the page to reconnect.
          </p>
        )}
      </section>

      {ended && summary && summary.status !== 'running' && (
        <>
          <ResultCard summary={summary} />
          {summary.leaderboard.length > 0 && (
            <Card title="Leaderboard">
              <Leaderboard leaderboard={summary.leaderboard} bestModelId={summary.best_model_id} />
            </Card>
          )}
          <Card title="Downloads">
            <Downloads runId={runId} files={summary.downloads} />
          </Card>
          {summary.report_markdown && (
            <Card title="Report">
              <ReportView markdown={summary.report_markdown} />
            </Card>
          )}
        </>
      )}
    </div>
  )
}

function ResultCard({ summary }: { summary: RunSummary }) {
  const best = summary.leaderboard.find((r) => r.model_id === summary.best_model_id)
  let headline: string
  if (summary.status === 'failed') {
    headline = `The run crashed: ${summary.error ?? 'unknown error'}`
  } else if (summary.status === 'interrupted') {
    headline = 'The run stopped before finishing (for example, the server was restarted).'
  } else if (best) {
    const metric = best.primary_metric
    headline = `Recommended model: ${best.model_name} · ${METRIC_LABELS[metric] ?? metric} ${formatMetric(best.metrics[metric])}`
  } else {
    headline = 'The run ended without a model. The report explains why.'
  }

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm" aria-label="Result">
      <p className="font-medium">{headline}</p>
      <p className="mt-1 text-sm text-slate-500">
        LLM cost ${summary.token_cost_usd.toFixed(4)}
      </p>
    </section>
  )
}

function Card({ title, children }: { title: string; children: ReactNode }) {
  // `children` is whatever is written between <Card> and </Card>.
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm" aria-label={title}>
      <h3 className="mb-4 text-base font-semibold">{title}</h3>
      {children}
    </section>
  )
}
