// The top-level component: the page frame, and which page each URL shows.

import { Link, Route, Routes } from 'react-router'
import { NewRunPage } from './pages/NewRunPage'
import { RunPage } from './pages/RunPage'

export default function App() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-6 py-4">
          <div>
            <Link to="/" className="text-lg font-semibold hover:text-indigo-700">
              auto-ds-agent
            </Link>
            <p className="text-sm text-slate-500">
              Describe your goal; the agent writes, repairs and benchmarks the rest.
            </p>
          </div>
          <Link to="/" className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50">
            New run
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-6 py-8">
        {/* Routes picks the first Route whose path matches the current URL. */}
        <Routes>
          <Route path="/" element={<NewRunPage />} />
          <Route path="/runs/:runId" element={<RunPage />} />
          <Route
            path="*"
            element={
              <p>
                Page not found.{' '}
                <Link to="/" className="text-indigo-600 hover:underline">Start a new run</Link>
              </p>
            }
          />
        </Routes>
      </main>
    </div>
  )
}
