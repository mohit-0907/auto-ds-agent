// The Feature code tab: line numbers, muted comments, and the Copy button's three states.

import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { CodeView } from './CodeView'

const CODE = '# Generated for mpg.csv\nimport pandas as pd\n\ndef engineer_features(df, t):\n    return df\n'

function stubClipboard(writeText: () => Promise<void>) {
  // userEvent replaces the clipboard with its own fake, so these tests click with fireEvent.
  Object.defineProperty(navigator, 'clipboard', { value: { writeText: vi.fn(writeText) }, configurable: true })
  return navigator.clipboard.writeText as ReturnType<typeof vi.fn>
}

afterEach(() => vi.useRealTimers())

describe('CodeView', () => {
  it('numbers every line, mutes comments, and does not count the final newline', () => {
    render(<CodeView code={CODE} tag={{ label: 'attempt 2 · passed validation', tone: 'ok' }} />)

    const view = screen.getByRole('region', { name: 'Feature code' })
    expect(view).toHaveTextContent('attempt 2 · passed validation')
    expect(view).toHaveTextContent('5 lines') // 5 lines of code, not 6
    expect(screen.getByText('# Generated for mpg.csv')).toHaveClass('text-muted')
    expect(screen.getByText('import pandas as pd')).not.toHaveClass('text-muted')
    // Line numbers are hidden from screen readers (and not selectable): they are not code.
    expect(screen.getByText('5')).toHaveAttribute('aria-hidden', 'true')
  })

  it('copies the code, says so, and goes back to "Copy" after a moment', async () => {
    vi.useFakeTimers()
    const writeText = stubClipboard(() => Promise.resolve())
    render(<CodeView code={CODE} tag={null} />)

    fireEvent.click(screen.getByRole('button', { name: 'Copy' }))
    await act(async () => {}) // let the clipboard's promise settle
    expect(writeText).toHaveBeenCalledWith(CODE)
    expect(screen.getByRole('button', { name: 'Copied' })).toBeInTheDocument()

    act(() => vi.advanceTimersByTime(2000))
    expect(screen.getByRole('button', { name: 'Copy' })).toBeInTheDocument()
  })

  it('says when the browser refuses the clipboard', async () => {
    stubClipboard(() => Promise.reject(new Error('NotAllowedError')))
    render(<CodeView code={CODE} tag={null} />)

    fireEvent.click(screen.getByRole('button', { name: 'Copy' }))
    expect(await screen.findByRole('button', { name: 'Copy failed' })).toBeInTheDocument()
  })
})
