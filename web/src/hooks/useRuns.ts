// The run list for the sidebar: fetched when the page changes, and every few seconds
// while any run is still running (so its status and "Running · <stage>" stay current).

import { useEffect, useState } from 'react'
import { useLocation } from 'react-router'
import { listRuns } from '../api'
import type { RunListItem } from '../types'

export const RUNS_POLL_MS = 3000

export function useRuns(): RunListItem[] {
  const { pathname } = useLocation()
  const [runs, setRuns] = useState<RunListItem[]>([])
  const [tick, setTick] = useState(0) // bumping this re-runs the fetch effect

  useEffect(() => {
    let cancelled = false
    listRuns()
      .then((result) => {
        if (!cancelled) setRuns(result)
      })
      .catch(() => {
        // keep showing the last list; the next route change or tick retries
      })
    return () => {
      cancelled = true
    }
  }, [pathname, tick])

  // Poll only while something is running: an idle app makes no background requests.
  const anyRunning = runs.some((run) => run.status === 'running')
  useEffect(() => {
    if (!anyRunning) return
    const timer = setInterval(() => setTick((t) => t + 1), RUNS_POLL_MS)
    return () => clearInterval(timer)
  }, [anyRunning])

  return runs
}
