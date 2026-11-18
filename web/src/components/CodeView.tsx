// The Feature code tab: fe_code.py with line numbers, its status, and a Copy button.

import { Check, Copy } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { Tone } from '../stages'
import { TONE_TAG } from '../tones'

const COPIED_MS = 2000 // how long "Copied" shows

interface Props {
  code: string
  tag: { label: string; tone: Tone } | null // "attempt 2 · passed validation" / "fallback recipe"
}

type CopyState = 'idle' | 'copied' | 'failed'

export function CodeView({ code, tag }: Props) {
  const [copy, setCopy] = useState<CopyState>('idle')
  // A file ends with a newline; that last empty "line" is not a line of code.
  const lines = code.replace(/\n$/, '').split('\n')

  // "Copied" goes back to "Copy" after a moment. The clean-up cancels the timer if the
  // person leaves the tab first, so it never sets state on a component that is gone.
  useEffect(() => {
    if (copy === 'idle') return
    const timer = setTimeout(() => setCopy('idle'), COPIED_MS)
    return () => clearTimeout(timer)
  }, [copy])

  const onCopy = () => {
    // The clipboard is outside the page, so the browser answers later (a Promise), and may
    // refuse (no permission, or a page not served over https/localhost).
    navigator.clipboard.writeText(code).then(
      () => setCopy('copied'),
      () => setCopy('failed'),
    )
  }

  return (
    <section aria-label="Feature code" className="flex flex-col overflow-hidden rounded-[28px] bg-card shadow-card">
      <div className="flex flex-wrap items-center gap-3 border-b border-divider px-6 py-4">
        <span className="font-mono text-[13px] font-medium">fe_code.py</span>
        {tag && (
          <span className={`rounded-full px-3 py-1 text-xs font-semibold whitespace-nowrap ${TONE_TAG[tag.tone]}`}>
            {tag.label}
          </span>
        )}
        <span className="font-mono text-xs text-muted">{lines.length} lines</span>
        <span className="flex-1" />
        <button
          type="button"
          onClick={onCopy}
          className="inline-flex cursor-pointer items-center gap-1.5 rounded-full border border-divider bg-transparent px-3.5 py-1.5 text-xs font-semibold text-ink hover:bg-sel"
        >
          {copy === 'copied' ? <Check size={14} strokeWidth={2.75} aria-hidden="true" /> : <Copy size={14} strokeWidth={2.75} aria-hidden="true" />}
          {copy === 'copied' ? 'Copied' : copy === 'failed' ? 'Copy failed' : 'Copy'}
        </button>
      </div>
      {/* One row per line: number (not selectable, so a manual copy gets only code) + code. */}
      <pre className="m-0 overflow-x-auto bg-code py-4 font-mono text-[12.5px] leading-[1.7]">
        {lines.map((line, i) => (
          <div key={i} className="flex">
            <span className="w-[52px] flex-none pr-4 text-right text-faint select-none" aria-hidden="true">
              {i + 1}
            </span>
            <code className={`pr-6 whitespace-pre ${line.trimStart().startsWith('#') ? 'text-muted' : ''}`}>
              {line || ' '}
            </code>
          </div>
        ))}
      </pre>
    </section>
  )
}
