// A run's status as a coloured pill with a dot: "Running", "Finished · fallback", "No model"...

import { statusTag } from '../stages'
import { TONE_DOT, TONE_TAG } from '../tones'
import type { RunOutcome, RunStatus } from '../types'

interface Props {
  status: RunStatus
  outcome: RunOutcome | null
}

export function StatusTag({ status, outcome }: Props) {
  const { label, tone } = statusTag(status, outcome)
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold whitespace-nowrap ${TONE_TAG[tone]}`}
    >
      <span className={`h-[7px] w-[7px] rounded-full ${TONE_DOT[tone]}`} aria-hidden="true" />
      {label}
    </span>
  )
}
