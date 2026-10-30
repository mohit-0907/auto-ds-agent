// A small coloured label for a run's status, shared by the run page and the history page.

import type { RunStatus } from '../types'

const STYLES: Record<RunStatus, string> = {
  running: 'bg-indigo-100 text-indigo-700',
  finished: 'bg-emerald-100 text-emerald-700',
  failed: 'bg-red-100 text-red-700',
  interrupted: 'bg-amber-100 text-amber-800',
}

export function StatusPill({ status }: { status: RunStatus }) {
  return (
    <span className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-medium ${STYLES[status]}`}>
      {status}
    </span>
  )
}
