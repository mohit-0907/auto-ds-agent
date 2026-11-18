// Everything the run warned about, above the results: fallback used, target inferred, ties...

import { TriangleAlert } from 'lucide-react'

export function WarningsStrip({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null
  return (
    <section aria-label="Warnings" className="flex flex-col gap-1.5 rounded-[20px] bg-warn-bg px-[18px] py-3.5 text-warn-fg">
      {warnings.map((warning, i) => (
        // Index as key: warnings are plain strings and the same text can appear twice.
        <p key={i} className="m-0 flex items-start gap-2.5 text-[13px]">
          <TriangleAlert size={15} strokeWidth={2.75} className="mt-0.5 flex-none" aria-hidden="true" />
          {warning}
        </p>
      ))}
    </section>
  )
}
