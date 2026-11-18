// The run's files as download links with their sizes, and what the run cost in LLM calls.
// Plain <a download> links: the browser does the download, no JavaScript involved.

import { Download, File } from 'lucide-react'
import { fileUrl } from '../api'
import { formatBytes } from '../results'
import type { DownloadFile } from '../types'

const DESCRIPTIONS: Record<string, string> = {
  'best_pipeline.joblib': 'The chosen model, retrained on all rows',
  'fe_code.py': 'Generated feature code',
  'report.md': 'The written report',
  'leaderboard.csv': 'Every model’s scores',
  'run.json': 'Every decision, warning and cost',
}

interface Props {
  runId: string
  files: DownloadFile[] // summary.downloads: only what the server will actually serve
  costUsd: number
  fallback: boolean // the features came from the fixed recipe, not generated code
}

export function ArtifactsCard({ runId, files, costUsd, fallback }: Props) {
  return (
    <section aria-label="Artifacts" className="flex flex-col gap-1.5 rounded-[28px] bg-card p-[22px] shadow-card">
      <h3 className="m-0 px-1.5 pb-1.5 text-[11px] font-semibold tracking-[0.08em] text-muted uppercase">Artifacts</h3>
      {files.length === 0 && <p className="m-0 px-1.5 text-sm text-muted">No files to download.</p>}
      <ul className="m-0 flex list-none flex-col p-0">
        {files.map((file) => (
          <li key={file.name}>
            <a
              href={fileUrl(runId, file.name)}
              download // "save this", even if the browser could display it
              className="flex items-center gap-3 rounded-2xl px-2 py-2.5 text-ink no-underline hover:bg-sel"
            >
              <span className="flex h-9 w-9 flex-none items-center justify-center rounded-full bg-page text-muted">
                <File size={16} strokeWidth={2.75} aria-hidden="true" />
              </span>
              <span className="flex min-w-0 flex-1 flex-col">
                <span className="truncate font-mono text-[12.5px] font-medium">{file.name}</span>
                <span className="text-xs text-muted">
                  {file.name === 'fe_code.py' && fallback ? 'Fallback feature recipe' : (DESCRIPTIONS[file.name] ?? '')}
                </span>
              </span>
              <span className="font-mono text-[11.5px] text-muted">{formatBytes(file.size_bytes)}</span>
              <Download size={16} strokeWidth={2.75} className="flex-none text-accent-fg" aria-hidden="true" />
            </a>
          </li>
        ))}
      </ul>
      <p className="m-0 mt-auto flex justify-between border-t border-divider px-2 pt-3 text-xs text-muted">
        <span>LLM cost</span>
        <span className="font-mono">${costUsd.toFixed(4)}</span>
      </p>
    </section>
  )
}
