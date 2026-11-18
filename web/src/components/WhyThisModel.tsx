// "Why this model": the selection rule's fold-by-fold evidence, one row per other model.
// The server recorded every number when it chose (Selection); this file only draws it.

import { comparisonView, whySentence, type Square } from '../results'
import type { ModelResult, Selection } from '../types'

const SQUARE: Record<Square, string> = {
  chosen: 'border-transparent bg-ok-dot', // a fold the chosen model won
  other: 'border-transparent bg-faint', // a fold the other model won
  empty: 'border-divider bg-transparent',
}

interface Props {
  selection: Selection
  leaderboard: ModelResult[]
  chosenName: string
  metric: string
}

export function WhyThisModel({ selection, leaderboard, chosenName, metric }: Props) {
  return (
    <div className="flex flex-col gap-2 border-t border-divider pt-4">
      <h3 className="m-0 text-[13px] font-semibold">Why this model</h3>
      <p className="m-0 text-[13.5px] text-muted text-pretty">{whySentence(selection, leaderboard, metric)}</p>
      {selection.comparisons.length > 0 && (
        <ul className="m-0 mt-1 flex list-none flex-col gap-1.5 p-0">
          {selection.comparisons.map((row) => {
            const view = comparisonView(row, selection, chosenName)
            return (
              <li key={row.model_id} className="grid grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-3.5 text-[13px]">
                <span className="truncate">
                  <span className="text-muted">vs</span> {row.model_name}
                  {row.better_mean && <span className="text-muted"> (better mean)</span>}
                </span>
                {/* role="img" + label: the squares are a picture; this is what it says. */}
                <span role="img" aria-label={view.squaresLabel} className="flex gap-1">
                  {view.squares.map((square, i) => (
                    <span key={i} className={`h-3 w-3 rounded-[4px] border-[1.5px] ${SQUARE[square]}`} />
                  ))}
                </span>
                <span
                  className={`min-w-[92px] text-right font-mono text-xs font-medium ${view.tone === 'ok' ? 'text-ok-fg' : 'text-muted'}`}
                >
                  {view.verdict}
                </span>
              </li>
            )
          })}
        </ul>
      )}
      <p className="m-0 mt-1 text-[11.5px] text-muted">
        Rule: a higher-scoring model must beat every simpler model on at least{' '}
        {selection.min_fold_wins} of {selection.n_folds} folds, otherwise the simpler one wins.
        Squares: sage = folds {chosenName} won, grey = folds the other model won.
      </p>
    </div>
  )
}
