// The top-level component: page frame plus whichever view is current.
// Task 7.4 replaces the "run started" card with the live run view.

import { useState } from 'react'
import { NewRunForm } from './components/NewRunForm'

export default function App() {
  const [runId, setRunId] = useState<string | null>(null)

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-3xl px-6 py-4">
          <h1 className="text-lg font-semibold">auto-ds-agent</h1>
          <p className="text-sm text-slate-500">
            Upload a dataset and describe your goal. The agent picks the task, writes and repairs
            feature engineering code, benchmarks models, and writes a report.
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-6 py-8">
        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          {runId ? (
            <div className="space-y-3">
              <p className="font-medium">Run started</p>
              <p className="font-mono text-sm text-slate-600">{runId}</p>
              <button
                onClick={() => setRunId(null)}
                className="text-sm text-indigo-600 hover:underline"
              >
                Start another run
              </button>
            </div>
          ) : (
            <NewRunForm onStarted={setRunId} />
          )}
        </section>
      </main>
    </div>
  )
}
