// The top-level component: the app shell, and which page each URL shows inside it.

import { Link, Route, Routes } from 'react-router'
import { AppShell } from './components/AppShell'
import { HistoryPage } from './pages/HistoryPage'
import { NewRunPage } from './pages/NewRunPage'
import { RunPage } from './pages/RunPage'

export default function App() {
  return (
    <AppShell>
      {/* Routes picks the first Route whose path matches the current URL. */}
      <Routes>
        <Route path="/" element={<NewRunPage />} />
        <Route path="/runs" element={<HistoryPage />} />
        <Route path="/runs/:runId" element={<RunPage />} />
        <Route
          path="*"
          element={
            <p className="px-9 py-8">
              Page not found.{' '}
              <Link to="/" className="text-accent-fg">
                Start a new run
              </Link>
            </p>
          }
        />
      </Routes>
    </AppShell>
  )
}
