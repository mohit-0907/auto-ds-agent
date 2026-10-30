// Tests for the results components: leaderboard, downloads, rendered report.

import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { ModelResult } from '../types'
import { Downloads } from './Downloads'
import { Leaderboard } from './Leaderboard'
import { ReportView } from './ReportView'

function model(id: string, name: string, rmse: number, status: 'success' | 'failed' = 'success'): ModelResult {
  return {
    model_id: id,
    model_name: name,
    status,
    metrics: status === 'success' ? { rmse, r2: 0.8 } : {},
    metrics_std: status === 'success' ? { rmse: 400, r2: 0.03 } : {},
    primary_metric: 'rmse',
    fold_scores: [],
    training_time_s: 1.5,
    predict_time_s: 0.0058,
    error: status === 'failed' ? 'ValueError: boom' : null,
  }
}

// As on the real mpg run: Random Forest has the best mean, HGB is chosen by the fold rule.
const MPG = [
  model('rf', 'Random Forest', 2772),
  model('hgb', 'Hist Gradient Boosting', 2811),
  model('ridge', 'Ridge Regression', 3136),
  model('xgb', 'XGBoost', 0, 'failed'),
]

describe('Leaderboard', () => {
  it('shows every metric as mean ± std, with columns taken from the run', () => {
    render(<Leaderboard leaderboard={MPG} bestModelId="hgb" />)

    const headers = screen.getAllByRole('columnheader').map((h) => h.textContent)
    expect(headers).toEqual(['#', 'Model', 'RMSE', 'R²', 'Train (s)', 'Predict (ms)'])
    const rf = screen.getByText('Random Forest').closest('tr')!
    expect(rf).toHaveTextContent('2,772 ± 400')
    expect(rf).toHaveTextContent('5.8') // predict ms
  })

  it('marks the chosen model and explains why it is not the top row', () => {
    render(<Leaderboard leaderboard={MPG} bestModelId="hgb" />)

    const chosen = screen.getByText('chosen').closest('tr')!
    expect(within(chosen).getByText('Hist Gradient Boosting')).toBeInTheDocument()
    expect(screen.getByText(/selection rule/)).toHaveTextContent('at least 4 of the 5')
  })

  it('needs no explanation when the top row is chosen', () => {
    render(<Leaderboard leaderboard={MPG} bestModelId="rf" />)
    expect(screen.queryByText(/selection rule/)).not.toBeInTheDocument()
  })

  it('lists failed models with their error', () => {
    render(<Leaderboard leaderboard={MPG} bestModelId="hgb" />)
    expect(screen.getByText('XGBoost: failed (ValueError: boom)')).toBeInTheDocument()
  })
})

describe('Downloads', () => {
  it('links each available file to its download URL', () => {
    render(<Downloads runId="run-1" files={['report.md', 'fe_code.py']} />)

    const links = screen.getAllByRole('link')
    expect(links.map((a) => a.getAttribute('href'))).toEqual([
      '/api/runs/run-1/files/report.md',
      '/api/runs/run-1/files/fe_code.py',
    ])
    expect(links[0]).toHaveAttribute('download')
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
