// The arena's leaderboard: every model, every metric as mean ± std, each fold as a dot on one
// shared scale, timings, and which model was selected.

import { formatMetric } from '../stages'
import { BASELINE_ID, dotPosition, foldScale, lowerIsBetter, metricLabel, type FoldScale } from '../results'
import type { ModelResult } from '../types'

const TH = 'border-b border-divider px-2 py-2.5 text-[11px] font-semibold tracking-[0.08em] whitespace-nowrap text-muted uppercase'
const TD = 'border-b border-divider px-2 py-[11px]'

/** One model's fold scores as dots on a 120 px track. */
function FoldDots({ scores, scale, selected }: { scores: number[]; scale: FoldScale; selected: boolean }) {
  return (
    <span
      role="img"
      aria-label={`Fold scores: ${scores.map(formatMetric).join(', ')}`}
      className="relative block h-3.5 w-[120px]"
    >
      <span className="absolute inset-x-0 top-1.5 h-0.5 rounded-sm bg-divider" />
      {scores.map((score, i) => (
        <span
          key={i}
          className={`absolute top-[3px] -ml-1 h-2 w-2 rounded-full opacity-85 ${selected ? 'bg-ok-dot' : 'bg-muted'}`}
          style={{ left: `${dotPosition(score, scale)}%` }} // a computed position: inline style
        />
      ))}
    </span>
  )
}

interface Props {
  leaderboard: ModelResult[] // already ranked by the server, failed models last
  bestModelId: string | null
  // Runs saved before Task 7.8 have no "Why this model" panel; say why here instead.
  explainTie: boolean
}

export function Leaderboard({ leaderboard, bestModelId, explainTie }: Props) {
  const ranked = leaderboard.filter((r) => r.status === 'success')
  if (ranked.length === 0) return <p className="m-0 text-sm text-muted">No models were trained.</p>

  // Columns come from the run itself: ROC-AUC... for Titanic, RMSE... for Ames.
  const primary = ranked[0].primary_metric
  const metrics = [primary, ...Object.keys(ranked[0].metrics).filter((m) => m !== primary)]
  const lower = lowerIsBetter(primary)
  const scale = foldScale(leaderboard)
  const nFolds = ranked[0].fold_scores.length || 5
  const best = ranked.find((r) => r.model_id === bestModelId)

  return (
    <section aria-label="Leaderboard" className="flex flex-col gap-2.5 rounded-[28px] bg-card px-6 pt-[22px] pb-3 shadow-card">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h3 className="m-0 font-display text-xl font-normal">Leaderboard</h3>
        <span className="text-xs text-muted">
          Same {nFolds} folds for every model · ranked by {metricLabel(primary)},{' '}
          {lower ? 'lower' : 'higher'} is better
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[820px] border-collapse text-[13px]">
          <thead>
            <tr>
              <th className={`${TH} w-9 text-left`}>#</th>
              <th className={`${TH} text-left`}>Model</th>
              {metrics.map((m) => (
                <th key={m} className={`${TH} text-right ${m === primary ? 'text-ink' : ''}`}>
                  {metricLabel(m)}
                  {m === primary && (lower ? ' ↓' : ' ↑')}
                </th>
              ))}
              <th className={`${TH} pl-5 text-left`}>Folds</th>
              <th className={`${TH} text-right`}>Train s</th>
              <th className={`${TH} text-right`}>Predict ms</th>
            </tr>
          </thead>
          <tbody>
            {leaderboard.map((r, i) => {
              if (r.status !== 'success') {
                return (
                  <tr key={r.model_id} className="text-bad-fg">
                    <td className={`${TD} font-mono`}>–</td>
                    <td className={TD} colSpan={metrics.length + 4}>
                      {r.model_name}: failed ({r.error})
                    </td>
                  </tr>
                )
              }
              const selected = r.model_id === bestModelId
              const baseline = r.model_id === BASELINE_ID
              return (
                <tr key={r.model_id} className={selected ? 'bg-ok-bg/55' : ''}>
                  <td className={`${TD} font-mono text-muted`}>{i + 1}</td>
                  <td className={`${TD} whitespace-nowrap`}>
                    <span className="inline-flex items-center gap-2">
                      <span className={selected ? 'font-bold' : 'font-medium'}>{r.model_name}</span>
                      {selected && (
                        <span className="rounded-full bg-ok-bg px-[9px] py-0.5 text-[11px] font-semibold text-ok-fg">
                          selected
                        </span>
                      )}
                      {baseline && (
                        <span className="rounded-full border border-divider px-[9px] py-0.5 text-[11px] font-semibold text-muted">
                          baseline
                        </span>
                      )}
                    </span>
                  </td>
                  {metrics.map((m) => (
                    <td key={m} className={`${TD} text-right font-mono text-[12.5px] whitespace-nowrap`}>
                      <span className={m === primary ? 'font-semibold' : 'text-muted'}>{formatMetric(r.metrics[m])}</span>{' '}
                      <span className="text-[11px] text-muted">±{formatMetric(r.metrics_std[m])}</span>
                    </td>
                  ))}
                  <td className={`${TD} pl-5`}>
                    {/* Baselines get no dots: they would only stretch the shared scale. */}
                    {scale && !baseline && <FoldDots scores={r.fold_scores} scale={scale} selected={selected} />}
                  </td>
                  <td className={`${TD} text-right font-mono text-[12.5px] text-muted`}>{r.training_time_s.toFixed(2)}</td>
                  <td className={`${TD} text-right font-mono text-[12.5px] text-muted`}>
                    {(r.predict_time_s * 1000).toFixed(1)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {scale && (
        <p className="m-0 pt-1 pb-1.5 text-[11.5px] text-muted">
          Folds: each dot is one cross-validation fold on a shared scale ({formatMetric(scale.lo)} –{' '}
          {formatMetric(scale.hi)} {metricLabel(primary)}); tighter clusters mean steadier models.
        </p>
      )}
      {explainTie && best && best !== ranked[0] && (
        <p className="m-0 pb-1.5 text-[11.5px] text-muted">
          Why isn’t the top row selected? A model with a better average only wins if it beats every
          simpler model on most of the {nFolds} folds, and the models above {best.model_name} did not.
        </p>
      )}
    </section>
  )
}
