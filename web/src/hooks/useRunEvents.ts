// A custom hook: subscribe to one run's live stage events (Server-Sent Events).
//
//   const { events, stream, endStatus } = useRunEvents(runId)
//
// Components get a plain, ordered, duplicate-free list; the connection details live here.

import { useEffect, useMemo, useState } from 'react'
import { eventsUrl } from '../api'
import type { RunStatus, StageEvent } from '../types'

/** connecting -> live -> ended, or error if the stream could not be opened at all. */
export type StreamState = 'connecting' | 'live' | 'ended' | 'error'

export function useRunEvents(runId: string) {
  // Events stored BY INDEX, not appended to a list: receiving event 3 twice (a reconnect,
  // a refresh, or React StrictMode connecting twice in development) just rewrites slot 3.
  const [byIndex, setByIndex] = useState<Record<number, StageEvent>>({})
  const [stream, setStream] = useState<StreamState>('connecting')
  const [endStatus, setEndStatus] = useState<RunStatus | null>(null)

  // useEffect: after React has drawn the page, open the connection. The function it
  // returns is the clean-up: React calls it when the page is left (or, in development,
  // between StrictMode's two mounts), so a connection never outlives its page.
  useEffect(() => {
    const source = new EventSource(eventsUrl(runId))

    source.onopen = () => setStream('live')

    source.addEventListener('stage', (message) => {
      const event = JSON.parse((message as MessageEvent<string>).data) as StageEvent
      // Returning `previous` unchanged when the slot is filled tells React nothing changed.
      setByIndex((previous) =>
        previous[event.index] ? previous : { ...previous, [event.index]: event },
      )
    })

    source.addEventListener('end', (message) => {
      setEndStatus(JSON.parse((message as MessageEvent<string>).data).status as RunStatus)
      setStream('ended')
      // Essential: when the server finishes a stream, EventSource treats it as a dropped
      // connection and reconnects every few seconds, replaying the run forever. Close it.
      source.close()
    })

    source.onerror = () => {
      // While the browser is retrying by itself, readyState is CONNECTING; only a closed
      // source (e.g. the server answered 404) is a real failure.
      if (source.readyState === EventSource.CLOSED) setStream('error')
    }

    return () => source.close()
  }, [runId])

  // Derived, not stored: the ordered list is recomputed only when byIndex changes.
  const events = useMemo(
    () => Object.values(byIndex).sort((a, b) => a.index - b.index),
    [byIndex],
  )

  return { events, stream, endStatus }
}
