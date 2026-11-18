// URL "/": the upload form beside "What the agent does". On success, open the new run.

import { useNavigate } from 'react-router'
import { NewRunForm } from '../components/NewRunForm'
import { PipelineAside } from '../components/PipelineAside'

export function NewRunPage() {
  const navigate = useNavigate() // a function that changes the URL, like clicking a link

  return (
    // Two columns that wrap: the aside drops below the form on narrow screens.
    <div className="flex max-w-[1080px] flex-wrap items-start gap-10 px-4 pt-10 pb-16 min-[760px]:px-9">
      <div className="min-w-0 flex-[1_1_420px]">
        <h1 className="mt-0 mb-2 font-display text-4xl font-normal tracking-[-0.015em]">
          Start a new run
        </h1>
        <p className="mt-0 mb-7 max-w-[54ch] text-[15px] text-pretty text-muted">
          Upload a table and say what you want to predict. The agent profiles it, writes and
          repairs feature code, benchmarks six models and explains its pick.
        </p>
        <NewRunForm onStarted={(runId) => navigate(`/runs/${runId}`)} />
      </div>
      <PipelineAside />
    </div>
  )
}
