// Download links for a run's artifacts. Plain <a> links: the browser does the download.

import { fileUrl } from '../api'

const DESCRIPTIONS: Record<string, string> = {
  'report.md': 'The written report',
  'leaderboard.csv': 'Every model’s scores',
  'fe_code.py': 'The feature engineering code that was used',
  'best_pipeline.joblib': 'The chosen model, retrained on all rows',
  'run.json': 'Every decision, warning and cost',
}

interface Props {
  runId: string
  files: string[] // summary.downloads: only what the server will actually serve
}

export function Downloads({ runId, files }: Props) {
  if (files.length === 0) return <p className="text-sm text-slate-500">No files to download.</p>
  return (
    <ul className="grid gap-2 sm:grid-cols-2">
      {files.map((name) => (
        <li key={name}>
          <a
            href={fileUrl(runId, name)}
            download // "save this", even if the browser could display it
            className="block rounded-lg border border-slate-200 px-3 py-2 hover:border-indigo-400 hover:bg-indigo-50"
          >
            <span className="font-mono text-sm text-indigo-700">{name}</span>
            <span className="block text-xs text-slate-500">{DESCRIPTIONS[name] ?? ''}</span>
          </a>
        </li>
      ))}
    </ul>
  )
}
