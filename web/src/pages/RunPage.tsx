// URL "/runs/:runId": one run -- header, live pipeline graph, and tabs for its steps and results.

import { useEffect, useState, type ReactNode } from 'react'
import { Link, useParams } from 'react-router'
import { ApiError, getRun } from '../api'
import { Downloads } from '../components/Downloads'
import { Leaderboard } from '../components/Leaderboard'
import { PipelineGraph } from '../components/PipelineGraph'
import { ReportView } from '../components/ReportView'
import { RunHeader } from '../components/RunHeader'
import { RunTabs, type TabSpec } from '../components/RunTabs'
import { Timeline } from '../components/Timeline'
import { useNow } from '../hooks/useNow'
import { useRunEvents } from '../hooks/useRunEvents'
import {
  METRIC_LABELS,
  formatMetric,
  graphCaption,
  pipelineState,
  runFacts,
  secondsBetween,
} from '../stages'
import type { RunStatus, RunSummary } from '../types'

type TabId = 'results' | 'steps' | 'report'

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
  // Which step rows are open. Lives here, not in Timeline, so it survives switching tabs.
  const [open, setOpen] = useState<Set<number>>(() => new Set())
  // A tab the person clicked, remembered together with whether the run was live then.
  const [choice, setChoice] = useState<{ tab: TabId; live: boolean } | null>(null)
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

  const status: RunStatus = endStatus ?? summary?.status ?? 'running'
  const live = status === 'running' && !ended && stream !== 'error'
  const now = useNow(live) // re-renders every second while live, for the elapsed clock

  if (loadError) {
    return (
      <div className="px-9 pt-8">
        <section className="rounded-[28px] bg-card p-6 shadow-card">
          <p role="alert" className="m-0 text-bad-fg">{loadError}</p>
          <Link to="/" className="mt-3 inline-block text-sm font-semibold text-accent-fg">
            Start a new run
          </Link>
        </section>
      </div>
    )
  }

  // Everything below is derived from events + summary on each render; nothing is stored twice.
  const startIso = summary?.created_at ?? events[0]?.at ?? null
  const lastAt = events[events.length - 1]?.at ?? null
  let seconds: number | null = null
  if (startIso && live) seconds = (now.getTime() - Date.parse(startIso)) / 1000
  else if (startIso && lastAt) seconds = secondsBetween(startIso, lastAt)

  const outcome = summary?.outcome ?? null
  const resultsReady = summary !== null && summary.status !== 'running'
  const tabs: TabSpec<TabId>[] = [
    { id: 'results', label: 'Results', disabled: !resultsReady },
    { id: 'steps', label: 'Steps', count: events.length },
    { id: 'report', label: 'Report', disabled: !resultsReady || !summary?.report_markdown },
  ]
  // A choice made while the run was live expires when it ends, so the page moves to Results
  // by itself -- without an effect that sets state after every render.
  const defaultTab: TabId = resultsReady ? 'results' : 'steps'
  const chosen = choice && choice.live === live ? choice.tab : null
  const tab = chosen && !tabs.find((t) => t.id === chosen)?.disabled ? chosen : defaultTab

  const toggle = (index: number) =>
    setOpen((previous) => {
      const next = new Set(previous) // a new Set: React only re-renders on a new value
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })

  return (
    <div className="flex max-w-[1180px] flex-col gap-6 px-9 pt-8 pb-16">
      <RunHeader
        dataset={summary?.dataset ?? runId}
        description={summary?.description ?? null}
        facts={runFacts(events)}
        requestedTarget={summary?.target_column ?? null}
        live={live}
        seconds={seconds}
      />
      <PipelineGraph
        pipeline={pipelineState(events, live)}
        caption={graphCaption(events, status, outcome, live, seconds)}
      />
      {stream === 'error' && (
        <p role="alert" className="m-0 text-sm text-bad-fg">
          Lost the connection to the live progress. Refresh the page to reconnect.
        </p>
      )}

      <RunTabs tabs={tabs} active={tab} onSelect={(id) => setChoice({ tab: id, live })} />
      <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} className="flex flex-col gap-6">
        {tab === 'steps' && (
          <Timeline events={events} live={live} startIso={startIso} open={open} onToggle={toggle} />
        )}
        {tab === 'results' && summary && <Results summary={summary} />}
        {tab === 'report' && summary?.report_markdown && (
          <Card title="Report">
            <ReportView markdown={summary.report_markdown} />
          </Card>
        )}
      </div>
    </div>
  )
}

// The Results tab as in 7.5; Task 7.8 replaces it with the redesigned results.
function Results({ summary }: { summary: RunSummary }) {
  return (
    <>
      <ResultCard summary={summary} />
      {summary.leaderboard.length > 0 && (
        <Card title="Leaderboard">
          <Leaderboard leaderboard={summary.leaderboard} bestModelId={summary.best_model_id} />
        </Card>
      )}
      <Card title="Downloads">
        <Downloads runId={summary.run_id} files={summary.downloads} />
      </Card>
    </>
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
    <section className="rounded-[28px] bg-card p-6 shadow-card" aria-label="Result">
      <p className="m-0 font-semibold">{headline}</p>
      <p className="mt-1 mb-0 text-sm text-muted">LLM cost ${summary.token_cost_usd.toFixed(4)}</p>
    </section>
  )
}

function Card({ title, children }: { title: string; children: ReactNode }) {
  // `children` is whatever is written between <Card> and </Card>.
  return (
    <section className="rounded-[28px] bg-card p-6 shadow-card" aria-label={title}>
      <h3 className="mt-0 mb-4 text-base font-semibold">{title}</h3>
      {children}
    </section>
  )
}
