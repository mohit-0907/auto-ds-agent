// The Results tab: which of its two shapes to show, and in what order.
//   a model was chosen -> warnings, recommendation + artifacts side by side, leaderboard
//   no model           -> one card saying why (out of scope, crashed, interrupted...)

import type { RunSummary } from '../types'
import { ArtifactsCard } from './ArtifactsCard'
import { EndedCard } from './EndedCard'
import { Leaderboard } from './Leaderboard'
import { RecommendationCard } from './RecommendationCard'
import { WarningsStrip } from './WarningsStrip'

interface Props {
  summary: RunSummary
  onOpenReport: () => void
}

export function ResultsTab({ summary, onOpenReport }: Props) {
  const best = summary.leaderboard.find((r) => r.model_id === summary.best_model_id)
  if (summary.status !== 'finished' || !best) {
    return <EndedCard summary={summary} onOpenReport={onOpenReport} />
  }
  return (
    <>
      <WarningsStrip warnings={summary.warnings} />
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        <RecommendationCard best={best} selection={summary.selection} leaderboard={summary.leaderboard} />
        <ArtifactsCard
          runId={summary.run_id}
          files={summary.downloads}
          costUsd={summary.token_cost_usd}
          fallback={summary.outcome === 'fallback'}
        />
      </div>
      <Leaderboard
        leaderboard={summary.leaderboard}
        bestModelId={summary.best_model_id}
        explainTie={summary.selection === null}
      />
    </>
  )
}
