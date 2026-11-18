// The app frame: sidebar on the left, the current page on the right under a sticky top bar.
// Below 760 px wide the sidebar becomes a drawer, opened from a menu button in the top bar.

import { ChevronRight, Menu, X } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { Link, useLocation } from 'react-router'
import { useRuns } from '../hooks/useRuns'
import { useTheme } from '../hooks/useTheme'
import type { RunListItem } from '../types'
import { Sidebar } from './Sidebar'
import { StatusTag } from './StatusTag'

export function AppShell({ children }: { children: ReactNode }) {
  const runs = useRuns()
  const [theme, setTheme] = useTheme()
  const { pathname } = useLocation()
  const runId = pathname.startsWith('/runs/') ? decodeURIComponent(pathname.slice(6)) : null
  const current = runs.find((run) => run.run_id === runId) ?? null

  // The drawer remembers which page it was opened on, so going to another page (say, tapping
  // a run in it) closes it by itself -- no effect watching the URL needed.
  const [drawerAt, setDrawerAt] = useState<string | null>(null)
  const drawerOpen = drawerAt === pathname

  // Escape closes it. The key listener lives on the document (outside React), so an effect
  // adds it while the drawer is open and its clean-up removes it again.
  useEffect(() => {
    if (!drawerOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setDrawerAt(null)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [drawerOpen])

  return (
    // Wide: two columns, full viewport height; each column scrolls on its own.
    // Narrow: one column, and the sidebar is either hidden or floating over the page.
    <div className="grid h-screen grid-cols-1 bg-page text-ink min-[760px]:grid-cols-[268px_minmax(0,1fr)]">
      <div
        id="sidebar"
        // min-w-0: a grid item is never narrower than its widest content by default, so a
        // long run subline would push the sidebar past its 268 px column, over the page.
        className={`fixed inset-y-0 left-0 z-30 flex w-[268px] min-w-0 min-[760px]:static min-[760px]:w-auto ${
          drawerOpen ? 'shadow-card' : 'max-[759px]:hidden'
        }`}
      >
        <Sidebar runs={runs} selectedId={runId} theme={theme} onTheme={setTheme} />
      </div>
      {drawerOpen && (
        // The dimmed page behind the drawer; clicking it closes the drawer.
        <div aria-hidden="true" onClick={() => setDrawerAt(null)} className="fixed inset-0 z-20 bg-ink/30 min-[760px]:hidden" />
      )}
      <main className="min-h-0 min-w-0 overflow-auto">
        <TopBar
          pathname={pathname}
          run={current}
          menuOpen={drawerOpen}
          onMenu={() => setDrawerAt(drawerOpen ? null : pathname)}
        />
        {children}
      </main>
    </div>
  )
}

interface TopBarProps {
  pathname: string
  run: RunListItem | null
  menuOpen: boolean
  onMenu: () => void
}

function TopBar({ pathname, run, menuOpen, onMenu }: TopBarProps) {
  const onRun = pathname.startsWith('/runs/')
  return (
    <div className="sticky top-0 z-10 flex items-center gap-3 border-b border-divider bg-page px-4 py-4 min-[760px]:px-9">
      <button
        type="button"
        onClick={onMenu}
        aria-label={menuOpen ? 'Close menu' : 'Open menu'}
        aria-expanded={menuOpen}
        aria-controls="sidebar"
        className="flex h-8 w-8 flex-none cursor-pointer items-center justify-center rounded-full border border-divider bg-transparent text-ink hover:bg-sel min-[760px]:hidden"
      >
        {menuOpen ? <X size={16} strokeWidth={2.75} aria-hidden="true" /> : <Menu size={16} strokeWidth={2.75} aria-hidden="true" />}
      </button>
      {onRun ? (
        <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-3 text-[13px]">
          <Link to="/runs" className="text-muted no-underline hover:text-ink">
            Runs
          </Link>
          <ChevronRight size={14} strokeWidth={2.75} className="flex-none text-muted" aria-hidden="true" />
          <span className="truncate font-semibold">{run?.dataset ?? '…'}</span>
        </nav>
      ) : (
        <span className="text-[13px] text-muted">{pathname === '/runs' ? 'Runs' : 'New run'}</span>
      )}
      <span className="flex-1" />
      {onRun && run && <StatusTag status={run.status} outcome={run.outcome} />}
    </div>
  )
}
