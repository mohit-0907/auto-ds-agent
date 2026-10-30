// The run's report.md, rendered as a formatted page.
//
// Safety: the report contains LLM-written prose, which is untrusted. react-markdown builds
// React elements and does NOT render raw HTML inside the Markdown, so text like
// "<script>...</script>" in the report can never run in the browser. Never swap this for
// dangerouslySetInnerHTML or add the rehype-raw plugin.

import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export function ReportView({ markdown }: { markdown: string }) {
  return (
    // `prose` (Typography plugin) styles headings, lists and tables; overflow-x-auto lets
    // the wide leaderboard table scroll on narrow screens instead of breaking the layout.
    <article className="prose prose-slate max-w-none overflow-x-auto prose-table:text-sm">
      {/* remark-gfm adds GitHub-style tables, which the report's leaderboard uses. */}
      <Markdown remarkPlugins={[remarkGfm]}>{markdown}</Markdown>
    </article>
  )
}
