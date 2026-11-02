// The headline result: the chosen model, its cross-validated score, its lift over the
// baseline, and (for runs recorded since Task 7.8) the evidence for choosing it.

import { formatMetric } from '../stages'
import { liftText, metricLabel } from '../results'
import type { ModelResult, Selection } from '../types'
import { WhyThisModel } from './WhyThisModel'

interface Props {
  best: ModelResult
  selection: Selection | null
  leaderboard: ModelResult[]
}

export function RecommendationCard({ best, selection, leaderboard }: Props) {
  const metric = best.primary_metric
  const lift = selection?.baseline ? liftText(selection.baseline) : null
  return (
    <section aria-label="Recommendation" className="flex flex-col gap-[18px] rounded-[28px] bg-card px-7 py-[26px] shadow-card">
      <div className="flex flex-col gap-1">
        <span className="text-[11px] font-semibold tracking-[0.08em] text-accent-fg uppercase">Recommended model</span>
        <h2 className="m-0 font-display text-[30px] font-normal">{best.model_name}</h2>
      </div>
      <div className="flex flex-wrap items-end gap-7">
        <div className="flex flex-col">
          <span className="font-mono text-[44px] leading-none font-medium tracking-[-0.03em]">
            {formatMetric(best.metrics[metric])}
          </span>
          <span className="mt-1.5 text-[12.5px] text-muted">
            {metricLabel(metric)}{' '}
            <span className="font-mono">± {formatMetric(best.metrics_std[metric])}</span> ·{' '}
            {best.fold_scores.length || 5}-fold CV
          </span>
        </div>
        {lift && (
          <div className="flex flex-col gap-0.5 pb-0.5">
            <span className={`font-mono text-[15px] font-medium ${lift.better ? 'text-ok-fg' : 'text-muted'}`}>
              {lift.headline}
            </span>
            <span className="text-[12.5px] text-muted">{lift.sub}</span>
          </div>
        )}
      </div>
      {selection && (
        <WhyThisModel selection={selection} leaderboard={leaderboard} chosenName={best.model_name} metric={metric} />
      )}
    </section>
  )
}
