// The app frame: sidebar on the left, the current page on the right under a sticky top bar.

import { ChevronRight } from 'lucide-react'
import type { ReactNode } from 'react'
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

  return (
    // Two columns, full viewport height; each column scrolls on its own.
    <div className="grid h-screen grid-cols-[268px_minmax(0,1fr)] bg-page text-ink">
      <Sidebar runs={runs} selectedId={runId} theme={theme} onTheme={setTheme} />
      <main className="min-h-0 min-w-0 overflow-auto">
        <TopBar pathname={pathname} run={current} />
        {children}
      </main>
    </div>
  )
}

function TopBar({ pathname, run }: { pathname: string; run: RunListItem | null }) {
  const onRun = pathname.startsWith('/runs/')
  return (
    <div className="sticky top-0 z-10 flex items-center gap-3 border-b border-divider bg-page px-9 py-4">
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
