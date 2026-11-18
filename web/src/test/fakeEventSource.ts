// A stand-in for the browser's EventSource (the test browser, jsdom, has none).
// Tests decide exactly which messages "arrive", and when.

import { act } from '@testing-library/react'

type Listener = (message: MessageEvent<string>) => void

export class FakeEventSource {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSED = 2
  /** Every connection opened during a test, in order. */
  static instances: FakeEventSource[] = []

  readonly url: string
  readyState = FakeEventSource.CONNECTING
  onopen: (() => void) | null = null
  onerror: (() => void) | null = null
  private listeners: Record<string, Listener[]> = {}

  constructor(url: string) {
    this.url = url
    FakeEventSource.instances.push(this)
  }

  addEventListener(type: string, listener: Listener) {
    ;(this.listeners[type] ??= []).push(listener)
  }

  close() {
    this.readyState = FakeEventSource.CLOSED
  }

  /** The server accepted the connection. */
  open() {
    act(() => {
      this.readyState = FakeEventSource.OPEN
      this.onopen?.()
    })
  }

  /** A message arrives, e.g. emit('stage', {...}). act() lets React finish redrawing. */
  emit(type: string, data: unknown) {
    act(() => {
      const message = new MessageEvent(type, { data: JSON.stringify(data) })
      for (const listener of this.listeners[type] ?? []) listener(message)
    })
  }

  /** The connection failed for good (e.g. the server answered 404). */
  fail() {
    act(() => {
      this.readyState = FakeEventSource.CLOSED
      this.onerror?.()
    })
  }

  static latest(): FakeEventSource {
    return FakeEventSource.instances[FakeEventSource.instances.length - 1]
  }
}
