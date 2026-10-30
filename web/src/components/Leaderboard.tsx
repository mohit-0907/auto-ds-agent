// The arena's leaderboard: every model, every metric as mean ± std, timings, the winner.

import { formatMetric, METRIC_LABELS } from '../stages'
import type { ModelResult } from '../types'

interface Props {
  leaderboard: ModelResult[] // already ranked by the server, failed models last
  bestModelId: string | null
}

export function Leaderboard({ leaderboard, bestModelId }: Props) {
  const ranked = leaderboard.filter((r) => r.status === 'success')
  if (ranked.length === 0) {
    return <p className="text-sm text-slate-500">No models were trained.</p>
  }

  // Columns come from the run itself: ROC-AUC... for Titanic, RMSE... for Ames.
  const primary = ranked[0].primary_metric
  const metrics = [primary, ...Object.keys(ranked[0].metrics).filter((m) => m !== primary)]
  const best = ranked.find((r) => r.model_id === bestModelId)
  const winnerIsNotTop = best !== undefined && best !== ranked[0]

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="py-2 pr-3 font-medium">#</th>
              <th className="py-2 pr-3 font-medium">Model</th>
              {metrics.map((m) => (
                <th key={m} className="py-2 pr-3 text-right font-medium">
                  {METRIC_LABELS[m] ?? m}
                </th>
              ))}
              <th className="py-2 pr-3 text-right font-medium">Train (s)</th>
              <th className="py-2 text-right font-medium">Predict (ms)</th>
            </tr>
          </thead>
          <tbody>
            {leaderboard.map((r, i) =>
              r.status === 'success' ? (
                <tr
                  key={r.model_id}
                  className={`border-b border-slate-100 ${r.model_id === bestModelId ? 'bg-emerald-50 font-medium' : ''}`}
                >
                  <td className="py-2 pr-3 text-slate-500">{i + 1}</td>
                  <td className="py-2 pr-3">
                    {r.model_name}
                    {r.model_id === bestModelId && (
                      <span className="ml-2 rounded bg-emerald-600 px-1.5 py-0.5 text-xs text-white">
                        chosen
                      </span>
                    )}
                  </td>
                  {metrics.map((m) => (
                    <td key={m} className="py-2 pr-3 text-right font-mono tabular-nums">
                      {formatMetric(r.metrics[m])}
                      <span className="text-slate-400"> ± {formatMetric(r.metrics_std[m])}</span>
                    </td>
                  ))}
                  <td className="py-2 pr-3 text-right font-mono tabular-nums">
                    {r.training_time_s.toFixed(2)}
                  </td>
                  <td className="py-2 text-right font-mono tabular-nums">
                    {(r.predict_time_s * 1000).toFixed(1)}
                  </td>
                </tr>
              ) : (
                <tr key={r.model_id} className="border-b border-slate-100 text-red-700">
                  <td className="py-2 pr-3">–</td>
                  <td className="py-2 pr-3" colSpan={metrics.length + 3}>
                    {r.model_name}: failed ({r.error})
                  </td>
                </tr>
              ),
            )}
          </tbody>
        </table>
      </div>
      {winnerIsNotTop && (
        <p className="mt-3 text-sm text-slate-600">
          Why isn’t the top row chosen? {best.model_name} was picked by the selection rule: a
          model with a better average only wins if it beats every simpler model on at least 4 of
          the 5 cross-validation folds, and the models above it did not.
        </p>
      )}
    </div>
  )
}
