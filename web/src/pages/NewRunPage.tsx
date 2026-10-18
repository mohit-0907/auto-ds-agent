// URL "/": the upload form. On success, go to the new run's own page.

import { useNavigate } from 'react-router'
import { NewRunForm } from '../components/NewRunForm'

export function NewRunPage() {
  const navigate = useNavigate() // a function that changes the URL, like clicking a link

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="mb-5 text-base font-semibold">Start a new run</h2>
      <NewRunForm onStarted={(runId) => navigate(`/runs/${runId}`)} />
    </section>
  )
}
