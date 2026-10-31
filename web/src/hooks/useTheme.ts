// Light/dark theme: remembered in localStorage, applied as <html data-theme="...">.

import { useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'

export const THEME_KEY = 'autods-theme'

/** The saved choice, else the operating system's preference, else light. */
export function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(THEME_KEY)
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    // storage blocked (private mode, sandboxed preview): fall through
  }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function useTheme() {
  // useState(initialTheme) -- passing the function, not calling it -- runs it only once.
  const [theme, setTheme] = useState<Theme>(initialTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme // the CSS variables switch on this
    try {
      localStorage.setItem(THEME_KEY, theme)
    } catch {
      // not saved, but the page still switches
    }
  }, [theme])

  return [theme, setTheme] as const
}
