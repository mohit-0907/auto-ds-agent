// The All runs page: the table, search, the status filter, and opening a run.

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useParams } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { fakeFetch } from '../test/fakeFetch'
import type { RunListItem } from '../types'
import { HistoryPage } from './HistoryPage'

function run(overrides: Partial<RunListItem>): RunListItem {
  return {
    run_id: 'mpg',
    status: 'finished',
    outcome: 'model',
    dataset: 'mpg.csv',
    description: 'Predict fuel efficiency',
    target_column: null,
    created_at: '2026-10-18T23:23:00',
    task_type: 'regression',
    best_model_id: 'hgb',
    best_model_name: 'Hist Gradient Boosting',
    primary_metric: 'rmse',
    best_score: 2.8109,
    last_event: null,
    ...overrides,
  }
}

const RUNS = [
  run({}),
  run({
    run_id: 'titanic',
    dataset: 'Titanic-Dataset.csv',
    description: 'Predict who survived',
    target_column: 'Survived',
    task_type: 'binary_classification',
    primary_metric: 'roc_auc',
    best_score: 0.88153, // not 0.8815: that is 0.88149999... in binary and rounds to 0.881
  }),
  run({
    run_id: 'flights',
    dataset: 'flights.csv',
    description: 'Forecast passengers',
    outcome: 'no_model',
    task_type: 'unsupported',
    best_model_id: null,
    best_model_name: null,
    primary_metric: null,
    best_score: null,
  }),
]

/** Stands in for the run page, to show which run a click opened. */
function OpenedRun() {
  return <p>opened {useParams().runId}</p>
}

function renderPage() {
  fakeFetch({ 'GET /api/runs': { status: 200, body: RUNS } })
  render(
    <MemoryRouter initialEntries={['/runs']}>
      <Routes>
        <Route path="/runs" element={<HistoryPage />} />
        <Route path="/runs/:runId" element={<OpenedRun />} />
      </Routes>
    </MemoryRouter>,
  )
}

afterEach(() => vi.unstubAllGlobals())

const datasets = () =>
  within(screen.getByRole('region', { name: 'Runs' }))
    .queryAllByRole('link')
    .map((a) => a.textContent)

describe('HistoryPage', () => {
  it('lists every run with its task, target, best model and score', async () => {
    renderPage()
    const titanic = (await screen.findByRole('link', { name: 'Titanic-Dataset.csv' })).closest('tr')!

    expect(titanic).toHaveTextContent('Predict who survived')
    expect(titanic).toHaveTextContent('Binary classification')
    expect(titanic).toHaveTextContent('Survived')
    expect(titanic).toHaveTextContent('ROC-AUC 0.882')
    expect(titanic).toHaveTextContent('Oct 18, 23:23')
    // No requested target: the agent worked it out.
    expect(screen.getByRole('link', { name: 'mpg.csv' }).closest('tr')).toHaveTextContent('inferred')
  })

  it('searches and filters, and says when nothing matches', async () => {
    renderPage()
    await screen.findByRole('link', { name: 'mpg.csv' })
    const user = userEvent.setup()

    await user.click(screen.getByRole('button', { name: 'No model' }))
    expect(screen.getByRole('button', { name: 'No model' })).toHaveAttribute('aria-pressed', 'true')
    expect(datasets()).toEqual(['flights.csv'])

    await user.click(screen.getByRole('button', { name: 'All' }))
    await user.type(screen.getByRole('searchbox', { name: 'Search datasets' }), 'SURVIVED')
    expect(datasets()).toEqual(['Titanic-Dataset.csv']) // matched on the description

    await user.click(screen.getByRole('button', { name: 'Running' }))
    expect(datasets()).toEqual([])
    expect(screen.getByText('No runs match.')).toBeInTheDocument()
  })

  it('clicking anywhere on a row opens that run', async () => {
    renderPage()
    const row = (await screen.findByRole('link', { name: 'flights.csv' })).closest('tr')!

    await userEvent.setup().click(within(row).getByText('Out of scope'))
    expect(screen.getByText('opened flights')).toBeInTheDocument()
  })
})
