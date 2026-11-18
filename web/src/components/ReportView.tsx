// The run's report.md, rendered as a formatted page in the design's type styles.
//
// Safety: the report contains LLM-written prose, which is untrusted. react-markdown builds
// React elements and does NOT render raw HTML inside the Markdown, so text like
// "<script>...</script>" in the report can never run in the browser. Never swap this for
// dangerouslySetInnerHTML or add the rehype-raw plugin.

import { ArrowRight } from 'lucide-react'
import { useMemo } from 'react'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface Props {
  markdown: string
  onOpenLeaderboard: () => void // the Results tab shows the leaderboard properly
}

/**
 * Which element to draw for each kind of Markdown. A paragraph that is nothing but italics
 * is the report's "_Generated ... by auto-ds-agent_" line, shown as a small meta line.
 */
function components(onOpenLeaderboard: () => void): Components {
  return {
    h1: () => null, // "# Run report: mpg.csv" repeats the page header
    h2: ({ children }) => <h2 className="mt-[26px] mb-2 font-display text-[21px] font-normal">{children}</h2>,
    p: ({ node, children }) => {
      const onlyItalics = node?.children.length === 1 && node.children[0].type === 'element' && node.children[0].tagName === 'em'
      return onlyItalics ? (
        <p className="m-0 text-[12.5px] text-muted italic">{children}</p>
      ) : (
        <p className="my-2.5 text-[14.5px] leading-[1.65]">{children}</p>
      )
    },
    ul: ({ children }) => <ul className="my-2.5 flex list-none flex-col gap-1.5 p-0">{children}</ul>,
    li: ({ children }) => (
      // The accent bullet is a small circle drawn by ::before.
      <li className="relative pl-5 text-[14.5px] leading-[1.65] before:absolute before:top-[0.62em] before:left-1 before:h-1.5 before:w-1.5 before:rounded-full before:bg-accent">
        {children}
      </li>
    ),
    code: ({ children }) => <code className="rounded-md bg-code px-1.5 py-0.5 font-mono text-[12.5px]">{children}</code>,
    a: ({ href, children }) => (
      // href has already been made safe by react-markdown (a "javascript:" link loses it).
      <a href={href} className="text-accent-fg underline">
        {children}
      </a>
    ),
    // The report's only table is the leaderboard, which the Results tab draws properly.
    table: () => (
      <button
        type="button"
        onClick={onOpenLeaderboard}
        className="my-1 inline-flex cursor-pointer items-center gap-2 rounded-full border border-divider bg-transparent px-4 py-[9px] text-[13px] font-semibold text-ink hover:bg-sel"
      >
        Leaderboard table — open in Results
        <ArrowRight size={14} strokeWidth={2.75} aria-hidden="true" />
      </button>
    ),
  }
}

export function ReportView({ markdown, onOpenLeaderboard }: Props) {
  // Built once per callback, not on every render: react-markdown treats a new function as a
  // new kind of element, and would throw away and rebuild the whole report each time.
  const elements = useMemo(() => components(onOpenLeaderboard), [onOpenLeaderboard])
  return (
    <article className="max-w-[720px] [overflow-wrap:anywhere]">
      {/* remark-gfm parses GitHub-style tables, so the leaderboard table is recognised. */}
      <Markdown remarkPlugins={[remarkGfm]} components={elements}>
        {markdown}
      </Markdown>
    </article>
  )
}
