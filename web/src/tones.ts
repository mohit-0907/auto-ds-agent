// Tailwind classes for each status tone. Status colours appear only on status (tags, dots,
// icons) -- never as decoration (design rule).

import type { Tone } from './stages'

export const TONE_TAG: Record<Tone, string> = {
  running: 'bg-accent-bg text-accent-fg',
  ok: 'bg-ok-bg text-ok-fg',
  warn: 'bg-warn-bg text-warn-fg',
  bad: 'bg-bad-bg text-bad-fg',
}

export const TONE_DOT: Record<Tone, string> = {
  running: 'bg-accent animate-pulse-dot',
  ok: 'bg-ok-dot',
  warn: 'bg-warn-dot',
  bad: 'bg-bad-dot',
}
