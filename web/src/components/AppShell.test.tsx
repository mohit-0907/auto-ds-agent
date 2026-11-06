// The frame around every page: the sidebar's run list, the selected run, and the theme switch.

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { THEME_KEY } from '../hooks/useTheme'
import { fakeFetch } from '../test/fakeFetch'
import type { RunListItem } from '../types'
import { AppShell } from './AppShell'

function run(overrides: Partial<RunListItem>): RunListItem {
  return {
    run_id: 'r1',
    status: 'finished',
    outcome: 'model',
    dataset: 'mpg.csv',
    description: 'Predict mpg',
    target_column: null,
    created_at: null,
    task_type: 'regression',
    best_model_id: 'hgb',
    best_model_name: 'Hist Gradient Boosting',
    primary_metric: 'rmse',
    best_score: 2.8712,
    last_event: null,
    ...overrides,
  }
}

function renderAt(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <AppShell>
        <p>page</p>
      </AppShell>
    </MemoryRouter>,
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
  localStorage.clear()
})

describe('AppShell', () => {
  it('lists runs in the sidebar and marks the open one', async () => {
    fakeFetch({
      'GET /api/runs': {
        status: 200,
        body: [
          run({ run_id: 'r2', dataset: 'penguins.csv', outcome: 'fallback' }),
          run({ run_id: 'r1' }),
        ],
      },
    })
    renderAt('/runs/r1')

    const nav = screen.getByRole('navigation', { name: 'Runs' })
    const open = await within(nav).findByRole('link', { current: 'page' })
    expect(open).toHaveTextContent('mpg.csv')
    expect(open).toHaveTextContent('Hist Gradient Boosting · RMSE 2.871')
    expect(within(nav).getAllByRole('link')).toHaveLength(2)

    // The top bar names the open run and shows its status.
    const crumbs = screen.getByRole('navigation', { name: 'Breadcrumb' })
    expect(crumbs).toHaveTextContent('mpg.csv')
    expect(screen.getByText('Finished')).toBeInTheDocument()
  })

  it('on narrow screens, opens the sidebar as a drawer and closes it again', async () => {
    // jsdom applies no CSS, so these tests check the state and the class that hides the
    // drawer below 760 px, not what is visible on screen.
    fakeFetch({ 'GET /api/runs': { status: 200, body: [run({ run_id: 'r1' })] } })
    renderAt('/')
    const user = userEvent.setup()
    const menu = screen.getByRole('button', { name: 'Open menu' })
    const drawer = document.getElementById('sidebar')!
    expect(menu).toHaveAttribute('aria-controls', 'sidebar')
    expect(drawer).toHaveClass('max-[759px]:hidden')

    await user.click(menu)
    expect(screen.getByRole('button', { name: 'Close menu' })).toHaveAttribute('aria-expanded', 'true')
    expect(drawer).not.toHaveClass('max-[759px]:hidden')

    await user.keyboard('{Escape}')
    expect(screen.getByRole('button', { name: 'Open menu' })).toHaveAttribute('aria-expanded', 'false')

    // Going to another page closes it: here, tapping a run inside the drawer.
    await user.click(screen.getByRole('button', { name: 'Open menu' }))
    const nav = screen.getByRole('navigation', { name: 'Runs' })
    await user.click(await within(nav).findByRole('link', { name: /mpg\.csv/ }))
    expect(screen.getByRole('button', { name: 'Open menu' })).toHaveAttribute('aria-expanded', 'false')
    expect(drawer).toHaveClass('max-[759px]:hidden')
  })

  it('switches theme, applies it to <html>, and remembers it', async () => {
    fakeFetch({ 'GET /api/runs': { status: 200, body: [] } })
    renderAt('/')
    const user = userEvent.setup()

    await user.click(screen.getByRole('button', { name: 'Dark theme' }))
    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(localStorage.getItem(THEME_KEY)).toBe('dark')
    expect(screen.getByRole('button', { name: 'Dark theme' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'Light theme' })).toHaveAttribute('aria-pressed', 'false')
  })
})
