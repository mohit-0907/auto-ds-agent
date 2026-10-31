// The left column: brand, "New run", the run list, and the theme switch.

import { Moon, Plus, Sun } from 'lucide-react'
import { Link } from 'react-router'
import type { Theme } from '../hooks/useTheme'
import { relativeTime, runSubline, statusTag } from '../stages'
import { TONE_DOT } from '../tones'
import type { RunListItem } from '../types'

interface Props {
  runs: RunListItem[]
  selectedId: string | null
  theme: Theme
  onTheme: (theme: Theme) => void
}

const KICKER = 'text-[11px] font-semibold tracking-[0.08em] uppercase text-muted'

export function Sidebar({ runs, selectedId, theme, onTheme }: Props) {
  return (
    <aside className="flex min-h-0 flex-col border-r border-divider bg-surface">
      <Link to="/" className="flex items-center gap-2.5 px-[18px] pt-5 pb-4 text-ink no-underline">
        {/* Brand mark: an accent circle with a surface-coloured dot. */}
        <span className="flex h-[26px] w-[26px] flex-none items-center justify-center rounded-full bg-accent">
          <span className="h-2.5 w-2.5 rounded-full bg-surface" />
        </span>
        <span className="font-display text-[19px] tracking-[-0.01em]">auto-ds-agent</span>
      </Link>

      <div className="px-3.5 pb-3.5">
        <Link
          to="/"
          className="flex w-full items-center justify-center gap-2 rounded-full bg-accent px-3.5 py-2.5 font-display text-sm text-on-accent no-underline hover:brightness-[0.93]"
        >
          <Plus size={16} strokeWidth={2.75} aria-hidden="true" />
          New run
        </Link>
      </div>

      <div className="flex items-center justify-between px-[18px] py-1.5">
        <span className={KICKER}>Runs</span>
        <Link
          to="/runs"
          className="rounded-full px-1.5 py-0.5 text-xs font-semibold text-accent-fg no-underline hover:bg-sel"
        >
          View all
        </Link>
      </div>

      {/* The only part that scrolls in the sidebar. */}
      <nav aria-label="Runs" className="flex min-h-0 flex-1 flex-col gap-0.5 overflow-auto px-2.5 pt-0.5 pb-2.5">
        {runs.map((run) => {
          const { tone } = statusTag(run.status, run.outcome)
          const selected = run.run_id === selectedId
          return (
            <Link
              key={run.run_id}
              to={`/runs/${run.run_id}`}
              aria-current={selected ? 'page' : undefined}
              className={`flex w-full items-start gap-2.5 rounded-[14px] px-2.5 py-[9px] text-left text-ink no-underline hover:bg-sel ${selected ? 'bg-sel' : ''}`}
            >
              <span className={`mt-1.5 h-2 w-2 flex-none rounded-full ${TONE_DOT[tone]}`} aria-hidden="true" />
              <span className="flex min-w-0 flex-1 flex-col">
                <span className="truncate text-[13.5px] font-semibold">{run.dataset}</span>
                <span className="truncate text-xs text-muted">{runSubline(run)}</span>
              </span>
              <span className="mt-0.5 flex-none text-[11px] text-muted">
                {relativeTime(run.created_at)}
              </span>
            </Link>
          )
        })}
        {runs.length === 0 && <p className="px-2.5 py-2 text-xs text-muted">No runs yet.</p>}
      </nav>

      <ThemeToggle theme={theme} onTheme={onTheme} />
    </aside>
  )
}

function ThemeToggle({ theme, onTheme }: { theme: Theme; onTheme: (theme: Theme) => void }) {
  const option = (value: Theme, Icon: typeof Sun, label: string) => (
    <button
      type="button"
      aria-pressed={theme === value}
      aria-label={`${label} theme`}
      onClick={() => onTheme(value)}
      className={`flex items-center gap-1.5 rounded-full px-2.5 py-[5px] text-xs font-semibold ${
        theme === value ? 'bg-accent text-on-accent' : 'text-muted hover:bg-sel'
      }`}
    >
      <Icon size={14} strokeWidth={2.75} aria-hidden="true" />
      {label}
    </button>
  )
  return (
    <div className="flex items-center justify-between gap-2 border-t border-divider p-3.5">
      <span className="text-xs text-muted">Theme</span>
      <div className="flex gap-0.5 rounded-full border border-divider p-[3px]">
        {option('light', Sun, 'Light')}
        {option('dark', Moon, 'Dark')}
      </div>
    </div>
  )
}
