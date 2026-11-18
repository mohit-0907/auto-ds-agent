import { render, renderHook, screen } from '@testing-library/react'
import { StrictMode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { stageEvent } from '../test/events'
import { FakeEventSource } from '../test/fakeEventSource'
import { useRunEvents } from './useRunEvents'

function stage(index: number, node: string) {
  return stageEvent({ index, node })
}

beforeEach(() => {
  FakeEventSource.instances = []
  vi.stubGlobal('EventSource', FakeEventSource)
})
afterEach(() => vi.unstubAllGlobals())

describe('useRunEvents', () => {
  it('connects to the run and collects events in order', () => {
    const { result } = renderHook(() => useRunEvents('run-1'))
    const source = FakeEventSource.latest()
    expect(source.url).toBe('/api/runs/run-1/events')
    expect(result.current.stream).toBe('connecting')

    source.open()
    source.emit('stage', stage(1, 'router')) // out of order on purpose
    source.emit('stage', stage(0, 'profiler'))

    expect(result.current.stream).toBe('live')
    expect(result.current.events.map((e) => e.node)).toEqual(['profiler', 'router'])
  })

  it('ignores a replayed event instead of showing it twice', () => {
    const { result } = renderHook(() => useRunEvents('run-1'))
    const source = FakeEventSource.latest()
    for (const index of [0, 1, 0, 1, 2]) source.emit('stage', stage(index, `node${index}`))

    expect(result.current.events.map((e) => e.index)).toEqual([0, 1, 2])
  })

  it('survives StrictMode connecting twice in development', () => {
    // Rendered like main.tsx does: a real component inside <StrictMode>. (A renderHook
    // wrapper does not trigger StrictMode's double mount -- an earlier version of this
    // test used one and so never tested StrictMode at all.)
    function Probe() {
      const { events } = useRunEvents('run-1')
      return <p>{events.length} events</p>
    }
    render(
      <StrictMode>
        <Probe />
      </StrictMode>,
    )

    // StrictMode mounted, cleaned up, and mounted again: two connections, the first closed.
    expect(FakeEventSource.instances).toHaveLength(2)
    expect(FakeEventSource.instances[0].readyState).toBe(FakeEventSource.CLOSED)

    const live = FakeEventSource.latest()
    live.emit('stage', stage(0, 'profiler'))
    live.emit('stage', stage(0, 'profiler'))
    expect(screen.getByText('1 events')).toBeInTheDocument()
  })

  it('closes the connection on "end", so the browser does not reconnect and replay', () => {
    const { result } = renderHook(() => useRunEvents('run-1'))
    const source = FakeEventSource.latest()
    source.emit('end', { status: 'finished' })

    expect(result.current.stream).toBe('ended')
    expect(result.current.endStatus).toBe('finished')
    expect(source.readyState).toBe(FakeEventSource.CLOSED)
  })

  it('closes the connection when the page is left', () => {
    const { unmount } = renderHook(() => useRunEvents('run-1'))
    unmount()
    expect(FakeEventSource.latest().readyState).toBe(FakeEventSource.CLOSED)
  })

  it('reports an error when the stream cannot be opened', () => {
    const { result } = renderHook(() => useRunEvents('missing'))
    FakeEventSource.latest().fail()
    expect(result.current.stream).toBe('error')
  })
})
