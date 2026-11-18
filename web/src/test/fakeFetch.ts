// A fake fetch that answers by URL and method, not by call order: components can make
// their requests in any order (the form fetches /api/samples on mount, for example).

import { vi } from 'vitest'

type Reply = { status: number; body: unknown }
type Routes = Record<string, Reply | Reply[]> // "POST /api/runs" or "GET /api/samples"

export function fakeFetch(routes: Routes) {
  const queues = Object.fromEntries(
    Object.entries(routes).map(([key, reply]) => [key, Array.isArray(reply) ? [...reply] : [reply]]),
  )
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const key = `${init?.method ?? 'GET'} ${url}`
    const queue = queues[key]
    if (!queue) throw new Error(`fakeFetch: no reply for ${key}`)
    // A list of replies is used in order; the last one repeats.
    const reply = queue.length > 1 ? queue.shift()! : queue[0]
    return new Response(JSON.stringify(reply.body), {
      status: reply.status,
      headers: { 'Content-Type': 'application/json' },
    })
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}
