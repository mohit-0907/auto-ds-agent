// Tests for the Results tab's pure functions: lift text, the "Why this model" sentence and
// rows, the fold-dot scale, file sizes, and what a run without a model says.

import { describe, expect, it } from 'vitest'
import {
  comparisonView,
  dotPosition,
  ending,
  foldScale,
  formatBytes,
  liftText,
  whySentence,
} from './results'
import type { ModelResult, PairComparison, RunSummary, Selection } from './types'

function model(id: string, name: string, folds: number[], metric = 'rmse'): ModelResult {
  const mean = folds.reduce((a, b) => a + b, 0) / folds.length
  return {
    model_id: id,
    model_name: name,
    status: 'success',
    metrics: { [metric]: mean },
    metrics_std: { [metric]: 0.4 },
    primary_metric: metric,
    fold_scores: folds,
    training_time_s: 1,
    predict_time_s: 0.01,
    error: null,
  }
}

function row(id: string, name: string, over: Partial<PairComparison>): PairComparison {
  return { model_id: id, model_name: name, better_mean: false, folds_won: 0, clearly: false, tested: false, ...over }
}

// The real mpg run (fallback features, 7.8a offline check): rf has the best mean RMSE but
// beats hgb on only 3 of 5 folds, so the simpler hgb is kept.
const MPG: Selection = {
  chosen_model_id: 'hgb',
  top_model_id: 'rf',
  n_folds: 5,
  min_fold_wins: 4,
  comparisons: [
    row('rf', 'Random Forest', { better_mean: true, folds_won: 3, tested: true }),
    row('lgbm', 'LightGBM', { better_mean: true, folds_won: 2 }),
    row('xgb', 'XGBoost', { folds_won: 4, clearly: true }),
    row('ridge', 'Ridge Regression', { folds_won: 4, clearly: true }),
  ],
  baseline: { baseline_name: 'Dummy (mean)', baseline_score: 7.8047, kind: 'percent', value: 63.69 },
}
const BOARD = [
  model('rf', 'Random Forest', [2.5, 2.9, 2.7, 3.0, 2.9]),
  model('hgb', 'Hist Gradient Boosting', [2.6, 2.8, 2.8, 3.1, 2.9]),
  model('dummy', 'Dummy (mean)', [7.8, 7.8, 7.8, 7.8, 7.8]),
]

describe('liftText', () => {
  it('shows error metrics as the share of the baseline error removed', () => {
    expect(liftText(MPG.baseline!)).toEqual({
      headline: '−63.7% error',
      sub: 'vs 7.805 baseline (Dummy (mean))',
      better: true,
    })
  })

  it('shows higher-is-better metrics as a plain difference', () => {
    const lift = liftText({ baseline_name: 'Dummy (class priors)', baseline_score: 0.5, kind: 'difference', value: 0.3815 })
    expect(lift.headline).toBe('+0.382')
    expect(lift.sub).toBe('over 0.500 baseline (Dummy (class priors))')
  })

  it('does not dress up a model that is worse than the baseline', () => {
    expect(liftText({ baseline_name: 'D', baseline_score: 0.5, kind: 'difference', value: -0.01 })).toMatchObject({
      headline: '−0.010',
      better: false,
    })
    expect(liftText({ baseline_name: 'D', baseline_score: 8, kind: 'percent', value: -5 }).headline).toBe('+5.0% error')
  })
})

describe('whySentence', () => {
  it('explains a simpler model kept in a tie, with the deciding fold count', () => {
    expect(whySentence(MPG, BOARD, 'rmse')).toBe(
      'Random Forest had a better mean RMSE but beat Hist Gradient Boosting on only 3 of 5 folds, ' +
        'so the simpler Hist Gradient Boosting wins the tie.',
    )
  })

  it('explains a top model that beat every simpler one', () => {
    const top: Selection = {
      ...MPG,
      top_model_id: 'hgb',
      comparisons: [row('logreg', 'Logistic Regression', { folds_won: 5, clearly: true, tested: true })],
    }
    expect(whySentence(top, BOARD, 'roc_auc')).toBe(
      'It had the best mean ROC-AUC and beat every simpler model on at least 4 of 5 folds, so the rule kept it.',
    )
  })

  it('says when there was nothing simpler to beat', () => {
    const simplest: Selection = { ...MPG, top_model_id: 'hgb', comparisons: [row('rf', 'Random Forest', { folds_won: 3 })] }
    expect(whySentence(simplest, BOARD, 'f1_macro')).toBe(
      'It had the best mean macro-F1, and no simpler model competed with it.',
    )
  })
})

describe('comparisonView', () => {
  const view = (r: PairComparison) => comparisonView(r, MPG, 'Hist Gradient Boosting')

  it('counts folds for the better-mean side and names whose wins they are', () => {
    const rf = view(MPG.comparisons[0])
    expect(rf.squares).toEqual(['other', 'other', 'other', 'empty', 'empty'])
    expect(rf.squaresLabel).toBe('Random Forest won 3 of 5 folds')
    expect(rf.verdict).toBe('only 3/5 · not enough') // the test that decided it

    const ridge = view(MPG.comparisons[3])
    expect(ridge.squares).toEqual(['chosen', 'chosen', 'chosen', 'chosen', 'empty'])
    expect(ridge.squaresLabel).toBe('Hist Gradient Boosting won 4 of 5 folds')
    expect([ridge.verdict, ridge.tone]).toEqual(['won 4/5', 'ok'])
  })

  it('an untested better-mean model is not presented as checked', () => {
    expect(view(MPG.comparisons[1]).verdict).toBe('only 2/5')
    // The gap in select_best: a mid-ranked model that clearly beats the chosen one.
    const gap = row('lgbm', 'LightGBM', { better_mean: true, folds_won: 4, clearly: true })
    expect([view(gap).verdict, view(gap).tone]).toEqual(['won 4/5 · untested', 'muted'])
  })

  it('a narrow lead over a worse-mean model is not shown as a clear win', () => {
    expect(view(row('xgb', 'XGBoost', { folds_won: 3 }))).toMatchObject({ verdict: 'won 3/5', tone: 'muted' })
  })
})

describe('foldScale / dotPosition', () => {
  it('shares one scale across real models, leaves the baseline out, pads 8% each side', () => {
    const scale = foldScale(BOARD)! // real folds span 2.5 .. 3.1; the dummy's 7.8 is ignored
    expect(scale.lo).toBeCloseTo(2.5 - 0.048)
    expect(scale.hi).toBeCloseTo(3.1 + 0.048)
    // The lowest fold sits just inside the left edge: 0.048 of a 0.696-wide scale = 6.9%.
    expect(dotPosition(2.5, scale)).toBeCloseTo(6.9, 1)
    expect(dotPosition(2.8, scale)).toBeCloseTo(50)
  })

  it('clamps scores outside the scale to the track', () => {
    const scale = { lo: 0, hi: 1 }
    expect(dotPosition(-1, scale)).toBe(0)
    expect(dotPosition(2, scale)).toBe(100)
  })

  it('gives identical scores a middle position instead of dividing by zero', () => {
    const scale = foldScale([model('ridge', 'Ridge', [3, 3, 3, 3, 3])])!
    expect(dotPosition(3, scale)).toBe(50)
  })

  it('has no scale without fold scores', () => {
    expect(foldScale([model('dummy', 'Dummy', [0.5, 0.5])])).toBeNull()
  })
})

describe('formatBytes', () => {
  it.each([
    [13, '13 B'],
    [2048, '2.0 KB'],
    [421_888, '412 KB'],
    [48_968_499, '46.7 MB'],
  ])('%i bytes -> %s', (bytes, text) => {
    expect(formatBytes(bytes)).toBe(text)
  })
})

describe('ending', () => {
  const base = { status: 'finished', task_type: 'regression', warnings: [], error: null } as unknown as RunSummary

  it('names an out-of-scope task and gives the reason from the warnings', () => {
    const run = { ...base, task_type: 'unsupported', warnings: ['Time-series forecasting is not supported.'] }
    expect(ending(run)).toMatchObject({
      kicker: 'Ended without a model',
      title: 'This task is out of scope',
      text: 'Time-series forecasting is not supported.',
    })
  })

  it('shows the last line of a crash traceback', () => {
    const run = { ...base, status: 'failed', error: 'Traceback (most recent call last):\n  ...\nMemoryError\n' } as RunSummary
    expect(ending(run)).toMatchObject({ kicker: 'The run crashed', text: 'MemoryError', tone: 'bad' })
  })

  it('an interrupted run is a warning, not a failure', () => {
    expect(ending({ ...base, status: 'interrupted' }).tone).toBe('warn')
  })
})
