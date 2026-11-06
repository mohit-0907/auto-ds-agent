// URL "/runs/:runId": one run -- header, live pipeline graph, and tabs for its steps and results.

import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router'
import { ApiError, getRun } from '../api'
import { CodeView } from '../components/CodeView'
import { PipelineGraph } from '../components/PipelineGraph'
import { ReportView } from '../components/ReportView'
import { ResultsTab } from '../components/ResultsTab'
import { RunHeader } from '../components/RunHeader'
import { RunTabs, type TabSpec } from '../components/RunTabs'
import { Timeline } from '../components/Timeline'
import { useNow } from '../hooks/useNow'
import { useRunEvents } from '../hooks/useRunEvents'
import { codeTag, graphCaption, pipelineState, runFacts, secondsBetween } from '../stages'
import type { RunStatus, RunSummary } from '../types'

type TabId = 'results' | 'steps' | 'report' | 'code'

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
  // One tab-switching function for the tabs and for buttons inside them ("Read the report",
  // "open in Results"). useCallback keeps it the same function between renders unless
  // `live` changes, so ReportView can build its Markdown elements once (see useMemo there).
  const openTab = useCallback((tab: TabId) => setChoice({ tab, live }), [live])
  const showResults = useCallback(() => openTab('results'), [openTab])

  if (loadError) {
    return (
      <div className="px-4 pt-8 min-[760px]:px-9">
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
    { id: 'code', label: 'Feature code', disabled: !resultsReady || !summary?.fe_code },
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
    <div className="flex max-w-[1180px] flex-col gap-6 px-4 pt-8 pb-16 min-[760px]:px-9">
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

      <RunTabs tabs={tabs} active={tab} onSelect={openTab} />
      <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} className="flex flex-col gap-6">
        {tab === 'steps' && (
          <Timeline events={events} live={live} startIso={startIso} open={open} onToggle={toggle} />
        )}
        {tab === 'results' && summary && (
          <ResultsTab summary={summary} onOpenReport={() => openTab('report')} />
        )}
        {tab === 'report' && summary?.report_markdown && (
          <section aria-label="Report" className="rounded-[28px] bg-card px-5 py-7 shadow-card min-[760px]:px-11 min-[760px]:py-9">
            <ReportView markdown={summary.report_markdown} onOpenLeaderboard={showResults} />
          </section>
        )}
        {tab === 'code' && summary?.fe_code && <CodeView code={summary.fe_code} tag={codeTag(events, outcome)} />}
      </div>
    </div>
  )
}
