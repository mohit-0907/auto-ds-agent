// The whole run screen: URL -> fetch the run -> stream its events -> graph, steps, results.

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { stageEvent } from '../test/events'
import { FakeEventSource } from '../test/fakeEventSource'
import { fakeFetch } from '../test/fakeFetch'
import type { RunSummary, StageEvent } from '../types'
import { NewRunPage } from './NewRunPage'
import { RunPage } from './RunPage'

const RUN_ID = '20261018_120000_titanic'

function summary(overrides: Partial<RunSummary> = {}): RunSummary {
  return {
    run_id: RUN_ID,
    status: 'running',
    dataset: 'Titanic-Dataset.csv',
    description: 'Predict who survived',
    target_column: null,
    created_at: '2026-10-18T12:00:00',
    events: [],
    task_type: null,
    best_model_id: null,
    leaderboard: [],
    report_markdown: null,
    token_cost_usd: 0,
    warnings: [],
    error: null,
    files: [],
    downloads: [],
    selection: null,
    outcome: null,
    ...overrides,
  }
}

const FINISHED = summary({
  status: 'finished',
  outcome: 'model',
  best_model_id: 'hgb',
  token_cost_usd: 0.0255,
  report_markdown: '# Run report: Titanic\n\nHist Gradient Boosting is recommended.',
  downloads: [
    { name: 'report.md', size_bytes: 2048 },
    { name: 'best_pipeline.joblib', size_bytes: 412_000 },
  ],
  leaderboard: [
    {
      model_id: 'hgb',
      model_name: 'Hist Gradient Boosting',
      status: 'success',
      // 0.88153, not 0.8815: 0.8815 is stored in binary as 0.88149999... and correctly
      // rounds to 0.881 (Python agrees) -- an earlier version of this test expected 0.882.
      metrics: { roc_auc: 0.88153 },
      metrics_std: { roc_auc: 0.024 },
      primary_metric: 'roc_auc',
      fold_scores: [],
      training_time_s: 1,
      predict_time_s: 0.01,
      error: null,
    },
  ],
})

/** A stage event `index * 5` seconds after the run started at 12:00:00. */
function stage(
  index: number,
  node: string,
  text: string,
  attempt = 0,
  ok = true,
  extra: Partial<StageEvent> = {},
): StageEvent {
  const seconds = index * 5
  const mm = String(Math.floor(seconds / 60)).padStart(2, '0')
  const ss = String(seconds % 60).padStart(2, '0')
  const at = `2026-10-18T12:${mm}:${ss}`
  return stageEvent({ index, node, ok, text, attempt, at, ...extra })
}

/** fetch answers GET /api/runs/{id} with each summary in turn (the last one repeats). */
function serveSummaries(...replies: RunSummary[]) {
  fakeFetch({ [`GET /api/runs/${RUN_ID}`]: replies.map((body) => ({ status: 200, body })) })
}

function renderAt(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/" element={<NewRunPage />} />
        <Route path="/runs/:runId" element={<RunPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

beforeEach(() => {
  FakeEventSource.instances = []
  vi.stubGlobal('EventSource', FakeEventSource)
})
afterEach(() => vi.unstubAllGlobals())

describe('RunPage', () => {
  it('shows the corrective loop live, then moves to the results', async () => {
    serveSummaries(summary(), FINISHED)
    renderAt(`/runs/${RUN_ID}`)
    expect(await screen.findByRole('heading', { name: 'Titanic-Dataset.csv' })).toBeInTheDocument()

    // While live: Steps is the tab; Results and Report wait for the end.
    expect(screen.getByRole('tab', { name: /Steps/ })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: 'Results' })).toBeDisabled()
    expect(screen.getByRole('tab', { name: 'Report' })).toBeDisabled()

    const source = FakeEventSource.latest()
    const shape = { n_rows: 891, n_cols: 12 }
    source.emit('stage', stage(0, 'profiler', '891 rows × 12 columns', 0, true, shape))
    source.emit(
      'stage',
      stage(1, 'router', 'binary_classification → Survived', 0, true, {
        task_type: 'binary_classification',
        target_column: 'Survived',
      }),
    )
    source.emit('stage', stage(2, 'sanity_check', 'no changes'))
    source.emit('stage', stage(3, 'fe_codegen', 'attempt 1 written', 1))
    source.emit('stage', stage(4, 'sandbox', "attempt 1 failed: KeyError: 'Deck'", 1, false))

    // The header facts arrived with the events that know them.
    const facts = screen.getByRole('list', { name: 'Run facts' })
    expect(facts).toHaveTextContent('Binary classification')
    expect(facts).toHaveTextContent('Survived (inferred)')
    expect(facts).toHaveTextContent('891 × 12')

    // The graph: the failed sandbox is red while codegen runs again for attempt 2.
    const graph = screen.getByRole('region', { name: 'Pipeline' })
    expect(within(graph).getByText(/Run in sandbox/)).toHaveTextContent('(fail)')
    expect(within(graph).getByText(/Write features/)).toHaveTextContent('(active)')
    expect(within(graph).getByText('attempt 1 / 3')).toBeInTheDocument()
    expect(within(graph).getByText('Live · 5 of ~11 stages')).toBeInTheDocument()

    // The retry is visible in the steps, with the next stage "working…".
    const attempt1 = screen.getByText('Attempt 1').closest('li')!
    expect(within(attempt1).getByText('failed')).toBeInTheDocument()
    expect(within(attempt1).getByLabelText('failed')).toBeInTheDocument()
    expect(screen.getByText('working…')).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /Steps/ })).toHaveTextContent('5')

    source.emit('stage', stage(5, 'fe_codegen', 'attempt 2 written', 2))
    source.emit('stage', stage(6, 'sandbox', 'attempt 2 passed validation', 2))
    expect(screen.getByText('Attempt 2').closest('li')).toHaveTextContent('passed')

    source.emit('end', { status: 'finished' })
    // The run ended: the page switches to Results by itself.
    const result = await screen.findByRole('region', { name: 'Result' })
    expect(screen.getByRole('tab', { name: 'Results' })).toHaveAttribute('aria-selected', 'true')
    expect(result).toHaveTextContent('Recommended model: Hist Gradient Boosting · ROC-AUC 0.882')
    expect(result).toHaveTextContent('LLM cost $0.0255')
    expect(screen.queryByText('working…')).not.toBeInTheDocument()

    const leaderboard = screen.getByRole('region', { name: 'Leaderboard' })
    expect(within(leaderboard).getByText('chosen')).toBeInTheDocument()
    const downloads = screen.getByRole('region', { name: 'Downloads' })
    expect(within(downloads).getByRole('link', { name: /best_pipeline\.joblib/ })).toHaveAttribute(
      'href',
      `/api/runs/${RUN_ID}/files/best_pipeline.joblib`,
    )

    // The report has its own tab.
    await userEvent.setup().click(screen.getByRole('tab', { name: 'Report' }))
    const report = screen.getByRole('region', { name: 'Report' })
    expect(within(report).getByRole('heading', { name: 'Run report: Titanic' })).toBeInTheDocument()
  })

  it('opens a step to show its detail, and remembers it across tabs', async () => {
    serveSummaries(FINISHED)
    renderAt(`/runs/${RUN_ID}`)
    const user = userEvent.setup()
    const source = FakeEventSource.latest()
    source.emit('stage', stage(0, 'profiler', '891 rows × 12 columns'))
    source.emit(
      'stage',
      stage(1, 'sandbox', "attempt 1 failed: KeyError: 'Deck'", 1, false, {
        detail: "KeyError: 'Deck'\n\n→ error sent back to the LLM for attempt 2",
        detail_kind: 'error',
      }),
    )
    source.emit('end', { status: 'finished' })
    await screen.findByRole('region', { name: 'Result' })
    await user.click(screen.getByRole('tab', { name: /Steps/ }))

    // A step without a detail is not a button; one with a detail is, closed at first.
    expect(within(screen.getByTestId('stage-0')).queryByRole('button')).toBeNull()
    const row = within(screen.getByTestId('stage-1')).getByRole('button')
    expect(row).toHaveAttribute('aria-expanded', 'false')

    await user.click(row)
    expect(row).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText(/error sent back to the LLM/)).toHaveClass('text-bad-fg')

    await user.click(screen.getByRole('tab', { name: 'Results' }))
    await user.click(screen.getByRole('tab', { name: /Steps/ }))
    expect(within(screen.getByTestId('stage-1')).getByRole('button')).toHaveAttribute(
      'aria-expanded',
      'true',
    )
  })

  it('a finished run shows its duration and a completed graph', async () => {
    serveSummaries(FINISHED)
    renderAt(`/runs/${RUN_ID}`)
    const source = FakeEventSource.latest()
    source.emit('stage', stage(0, 'profiler', 'p'))
    source.emit('stage', stage(12, 'finalize', 'run.json written'))
    source.emit('end', { status: 'finished' })
    await screen.findByRole('region', { name: 'Result' })

    // The last event is at 12:01:00, one minute after created_at.
    expect(screen.getByRole('list', { name: 'Run facts' })).toHaveTextContent('duration01:00')
    expect(screen.getByText('Completed in 01:00')).toBeInTheDocument()
    expect(screen.queryByText(/elapsed/)).not.toBeInTheDocument()
  })

  it('says so when the run does not exist', async () => {
    fakeFetch({ 'GET /api/runs/no-such-run': { status: 404, body: { detail: 'nope' } } })
    renderAt('/runs/no-such-run')
    expect(await screen.findByRole('alert')).toHaveTextContent('There is no run with this ID.')
  })
})

describe('NewRunPage', () => {
  it('goes to the new run’s page after a successful upload', async () => {
    fakeFetch({
      'GET /api/samples': { status: 200, body: [] },
      'POST /api/runs': { status: 202, body: { run_id: RUN_ID, status: 'running' } },
      [`GET /api/runs/${RUN_ID}`]: { status: 200, body: summary() },
    })
    renderAt('/')

    const user = userEvent.setup()
    await user.upload(screen.getByLabelText('Dataset file'), new File(['a\n1\n'], 'titanic.csv'))
    await user.type(screen.getByLabelText('What do you want to predict?'), 'Predict survival')
    await user.click(screen.getByRole('button', { name: 'Start run' }))

    expect(await screen.findByRole('heading', { name: 'Titanic-Dataset.csv' })).toBeInTheDocument()
    expect(FakeEventSource.latest().url).toBe(`/api/runs/${RUN_ID}/events`)
  })
})
