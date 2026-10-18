// The whole live page: URL -> fetch the run -> stream its events -> result card.

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { FakeEventSource } from '../test/fakeEventSource'
import type { RunSummary } from '../types'
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
    created_at: null,
    events: [],
    task_type: null,
    best_model_id: null,
    leaderboard: [],
    report_markdown: null,
    token_cost_usd: 0,
    warnings: [],
    error: null,
    files: [],
    ...overrides,
  }
}

const FINISHED = summary({
  status: 'finished',
  best_model_id: 'hgb',
  token_cost_usd: 0.0255,
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

function stage(index: number, node: string, text: string, attempt = 0, ok = true) {
  return { index, node, ok, text, attempt, at: `2026-10-18T12:00:${String(index).padStart(2, '0')}` }
}

/** fetch answers GET /api/runs/{id} with each summary in turn. */
function serveSummaries(...replies: RunSummary[]) {
  const fetchMock = vi.fn()
  for (const reply of replies) {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(reply), { status: 200 }))
  }
  vi.stubGlobal('fetch', fetchMock)
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
  it('shows the corrective loop live, then the result', async () => {
    serveSummaries(summary(), FINISHED)
    renderAt(`/runs/${RUN_ID}`)
    expect(await screen.findByText('Titanic-Dataset.csv')).toBeInTheDocument()

    const source = FakeEventSource.latest()
    source.emit('stage', stage(0, 'profiler', '891 rows × 12 columns'))
    source.emit('stage', stage(1, 'fe_codegen', 'attempt 1 written', 1))
    source.emit('stage', stage(2, 'sandbox', "attempt 1 failed: KeyError: 'Deck'", 1, false))
    expect(screen.getByText('Working…')).toBeInTheDocument()

    source.emit('stage', stage(3, 'fe_codegen', 'attempt 2 written', 2))
    source.emit('stage', stage(4, 'sandbox', 'attempt 2 passed validation', 2))

    // The retry is visible: two attempt blocks, the first with a failed sandbox run.
    const attempt1 = screen.getByText('Attempt 1').closest('li')!
    expect(within(attempt1).getByText("attempt 1 failed: KeyError: 'Deck'")).toBeInTheDocument()
    expect(within(attempt1).getByLabelText('failed')).toBeInTheDocument()
    expect(screen.getByText('Attempt 2')).toBeInTheDocument()

    source.emit('end', { status: 'finished' })
    const result = await screen.findByRole('region', { name: 'Result' })
    expect(result).toHaveTextContent('Recommended model: Hist Gradient Boosting · ROC-AUC 0.882')
    expect(result).toHaveTextContent('LLM cost $0.0255')
    expect(screen.queryByText('Working…')).not.toBeInTheDocument()
  })

  it('says so when the run does not exist', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: 'nope' }), { status: 404 })),
    )
    renderAt('/runs/no-such-run')
    expect(await screen.findByRole('alert')).toHaveTextContent('There is no run with this ID.')
  })
})

describe('NewRunPage', () => {
  it('goes to the new run’s page after a successful upload', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ run_id: RUN_ID, status: 'running' }), { status: 202 }),
      )
      .mockResolvedValueOnce(new Response(JSON.stringify(summary()), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    renderAt('/')

    const user = userEvent.setup()
    await user.upload(screen.getByLabelText('Dataset file'), new File(['a\n1\n'], 'titanic.csv'))
    await user.type(screen.getByLabelText('What do you want to predict?'), 'Predict survival')
    await user.click(screen.getByRole('button', { name: 'Start run' }))

    expect(await screen.findByText('Titanic-Dataset.csv')).toBeInTheDocument() // now on the run page
    expect(FakeEventSource.latest().url).toBe(`/api/runs/${RUN_ID}/events`)
  })
})
