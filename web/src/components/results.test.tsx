// Tests for the Results tab components: recommendation + why, artifacts, leaderboard,
// the no-model / crashed cards, and the rendered report.

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router'
import { describe, expect, it, vi } from 'vitest'
import type { ModelResult, RunSummary, Selection } from '../types'
import { ArtifactsCard } from './ArtifactsCard'
import { Leaderboard } from './Leaderboard'
import { ReportView } from './ReportView'
import { ResultsTab } from './ResultsTab'

function model(id: string, name: string, folds: number[], status: 'success' | 'failed' = 'success'): ModelResult {
  const mean = folds.length ? folds.reduce((a, b) => a + b, 0) / folds.length : 0
  return {
    model_id: id,
    model_name: name,
    status,
    metrics: status === 'success' ? { rmse: mean, r2: 0.8 } : {},
    metrics_std: status === 'success' ? { rmse: 0.4, r2: 0.03 } : {},
    primary_metric: 'rmse',
    fold_scores: folds,
    training_time_s: 1.5,
    predict_time_s: 0.0058,
    error: status === 'failed' ? 'ValueError: boom' : null,
  }
}

// As on the real mpg run: Random Forest has the best mean, HGB is chosen by the fold rule.
const MPG = [
  model('rf', 'Random Forest', [2.5, 2.7, 2.7, 3.0, 2.9]),
  model('hgb', 'Hist Gradient Boosting', [2.6, 2.8, 2.8, 3.1, 2.9]),
  model('dummy', 'Dummy (mean)', [7.8, 7.8, 7.8, 7.8, 7.8]),
  model('xgb', 'XGBoost', [], 'failed'),
]

const SELECTION: Selection = {
  chosen_model_id: 'hgb',
  top_model_id: 'rf',
  n_folds: 5,
  min_fold_wins: 4,
  comparisons: [
    { model_id: 'rf', model_name: 'Random Forest', better_mean: true, folds_won: 3, clearly: false, tested: true },
  ],
  baseline: { baseline_name: 'Dummy (mean)', baseline_score: 7.8, kind: 'percent', value: 63.7 },
}

function summary(overrides: Partial<RunSummary> = {}): RunSummary {
  return {
    run_id: 'run-1',
    status: 'finished',
    outcome: 'model',
    dataset: 'mpg.csv',
    description: 'Predict mpg',
    target_column: null,
    created_at: null,
    events: [],
    task_type: 'regression',
    best_model_id: 'hgb',
    leaderboard: MPG,
    selection: SELECTION,
    report_markdown: '# Report',
    token_cost_usd: 0.0231,
    warnings: [],
    error: null,
    files: [],
    downloads: [
      { name: 'best_pipeline.joblib', size_bytes: 421_888 },
      { name: 'fe_code.py', size_bytes: 2048 },
    ],
    ...overrides,
  }
}

function renderTab(run: RunSummary, onOpenReport = () => {}) {
  render(
    <MemoryRouter>
      <ResultsTab summary={run} onOpenReport={onOpenReport} />
    </MemoryRouter>,
  )
}

describe('ResultsTab with a model', () => {
  it('shows the recommendation, its lift, and the deciding comparison', () => {
    renderTab(summary())

    const card = screen.getByRole('region', { name: 'Recommendation' })
    expect(within(card).getByRole('heading', { name: 'Hist Gradient Boosting' })).toBeInTheDocument()
    expect(card).toHaveTextContent('2.840') // the mean RMSE
    expect(card).toHaveTextContent('RMSE ± 0.400 · 5-fold CV')
    expect(card).toHaveTextContent('−63.7% error')
    expect(card).toHaveTextContent('vs 7.800 baseline (Dummy (mean))')

    expect(card).toHaveTextContent(
      'Random Forest had a better mean RMSE but beat Hist Gradient Boosting on only 3 of 5',
    )
    const rf = within(card).getByText('Random Forest').closest('li')!
    expect(within(rf).getByRole('img', { name: 'Random Forest won 3 of 5 folds' })).toBeInTheDocument()
    expect(rf).toHaveTextContent('only 3/5 · not enough')
    expect(card).toHaveTextContent('at least 4 of 5 folds')
  })

  it('shows the run warnings above everything else', () => {
    renderTab(summary({ outcome: 'fallback', warnings: ['Feature engineering failed 3 times.'] }))
    expect(screen.getByRole('region', { name: 'Warnings' })).toHaveTextContent('Feature engineering failed 3 times.')
    // A fallback run's feature file is the fixed recipe, and says so.
    expect(screen.getByRole('region', { name: 'Artifacts' })).toHaveTextContent('Fallback feature recipe')
  })

  it('a run saved before selection records existed hides "Why this model" and explains in the table', () => {
    renderTab(summary({ selection: null }))
    expect(screen.queryByText('Why this model')).not.toBeInTheDocument()
    expect(screen.queryByText(/% error/)).not.toBeInTheDocument() // the lift came with the record
    expect(screen.getByText(/Why isn’t the top row selected/)).toBeInTheDocument()
  })
})

describe('ResultsTab without a model', () => {
  it('an out-of-scope task says why and offers the report', async () => {
    const onOpenReport = vi.fn()
    renderTab(
      summary({
        outcome: 'no_model',
        task_type: 'unsupported',
        best_model_id: null,
        leaderboard: [],
        selection: null,
        warnings: ['Forecasting future values is out of scope.'],
      }),
      onOpenReport,
    )
    const card = screen.getByRole('region', { name: 'Result' })
    expect(within(card).getByRole('heading', { name: 'This task is out of scope' })).toBeInTheDocument()
    expect(card).toHaveTextContent('Forecasting future values is out of scope.')
    expect(within(card).getByRole('link', { name: 'Try another dataset' })).toHaveAttribute('href', '/')

    await userEvent.setup().click(within(card).getByRole('button', { name: 'Read the report' }))
    expect(onOpenReport).toHaveBeenCalledOnce()
  })

  it('a crash shows the error, and no report button when there is no report', () => {
    renderTab(summary({ status: 'failed', outcome: null, report_markdown: null, error: 'Traceback...\nMemoryError' }))
    const card = screen.getByRole('region', { name: 'Result' })
    expect(card).toHaveTextContent('The run crashed')
    expect(card).toHaveTextContent('MemoryError')
    expect(within(card).queryByRole('button', { name: 'Read the report' })).toBeNull()
  })
})

describe('ArtifactsCard', () => {
  it('links each file to its download URL with its size, and shows the cost', () => {
    render(<ArtifactsCard runId="run-1" files={summary().downloads} costUsd={0.0231} fallback={false} />)

    const links = screen.getAllByRole('link')
    expect(links.map((a) => a.getAttribute('href'))).toEqual([
      '/api/runs/run-1/files/best_pipeline.joblib',
      '/api/runs/run-1/files/fe_code.py',
    ])
    expect(links[0]).toHaveAttribute('download')
    expect(links[0]).toHaveTextContent('412 KB')
    expect(links[1]).toHaveTextContent('Generated feature code')
    expect(screen.getByText('$0.0231')).toBeInTheDocument()
  })
})

describe('Leaderboard', () => {
  it('shows every metric as mean ± std, with the primary first and its direction', () => {
    render(<Leaderboard leaderboard={MPG} bestModelId="hgb" explainTie={false} />)

    const headers = screen.getAllByRole('columnheader').map((h) => h.textContent)
    expect(headers).toEqual(['#', 'Model', 'RMSE ↓', 'R²', 'Folds', 'Train s', 'Predict ms'])
    expect(screen.getByText(/ranked by RMSE, lower is better/)).toBeInTheDocument()
    const rf = screen.getByText('Random Forest').closest('tr')!
    expect(rf).toHaveTextContent('2.760 ±0.400')
    expect(rf).toHaveTextContent('5.8') // predict ms
  })

  it('tags the selected model and the baseline; only real models get fold dots', () => {
    render(<Leaderboard leaderboard={MPG} bestModelId="hgb" explainTie={false} />)

    const hgb = screen.getByText('selected').closest('tr')!
    expect(within(hgb).getByText('Hist Gradient Boosting')).toBeInTheDocument()
    expect(
      within(hgb).getByRole('img', { name: 'Fold scores: 2.600, 2.800, 2.800, 3.100, 2.900' }),
    ).toBeInTheDocument()
    const dummy = screen.getByText('baseline').closest('tr')!
    expect(within(dummy).queryByRole('img')).toBeNull()
    // The scale spans the real models' folds (2.5 .. 3.1) plus 8% padding, not the dummy's 7.8.
    expect(screen.getByText(/shared scale \(2\.452 – 3\.148 RMSE\)/)).toBeInTheDocument()
  })

  it('lists failed models with their error', () => {
    render(<Leaderboard leaderboard={MPG} bestModelId="hgb" explainTie={false} />)
    expect(screen.getByText('XGBoost: failed (ValueError: boom)')).toBeInTheDocument()
  })
})

describe('ReportView', () => {
  it('renders headings and GitHub-style tables', () => {
    render(<ReportView markdown={'## Leaderboard\n\n| Model | RMSE |\n|---|---|\n| HGB | 2,811 |\n'} />)

    expect(screen.getByRole('heading', { name: 'Leaderboard' })).toBeInTheDocument()
    expect(screen.getByRole('table')).toHaveTextContent('2,811')
  })

  it('never runs HTML hidden in the LLM-written report (XSS)', () => {
    const hostile = [
      'Hist Gradient Boosting is recommended.',
      '<script>window.hacked = true</script>',
      '<img src="x" onerror="window.hacked = true">',
      '[click me](javascript:window.hacked=true)',
    ].join('\n\n')
    const { container } = render(<ReportView markdown={hostile} />)

    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('img')).toBeNull()
    const link = container.querySelector('a')
    expect(link?.getAttribute('href') ?? '').not.toMatch(/^javascript:/i)
    expect((window as unknown as { hacked?: boolean }).hacked).toBeUndefined()
    expect(screen.getByText('Hist Gradient Boosting is recommended.')).toBeInTheDocument()
  })
})
