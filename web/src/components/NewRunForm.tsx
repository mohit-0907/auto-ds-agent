// The "start a new run" form: pick or drop a dataset, describe the goal, press Start.

import { useState, type DragEvent, type FormEvent } from 'react'
import { ApiError, checkFile, startRun } from '../api'

interface Props {
  /** Called with the new run's ID once the server has accepted the upload. */
  onStarted: (runId: string) => void
}

export function NewRunForm({ onStarted }: Props) {
  // State: the component's memory. Changing any of these makes React redraw the form.
  const [file, setFile] = useState<File | null>(null)
  const [description, setDescription] = useState('')
  const [target, setTarget] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [dragging, setDragging] = useState(false)

  // Not state: computed fresh on every redraw from the state above.
  const canSubmit = file !== null && description.trim() !== '' && !submitting

  function chooseFile(candidate: File | undefined) {
    if (!candidate) return
    const problem = checkFile(candidate) // checked here for an instant answer...
    setError(problem)
    setFile(problem ? null : candidate)
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault() // stop the browser from opening the dropped file itself
    setDragging(false)
    chooseFile(event.dataTransfer.files[0])
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault() // stop the browser's default "reload the page" on submit
    if (!file) return
    setSubmitting(true)
    setError(null)
    try {
      const created = await startRun(file, description.trim(), target.trim() || undefined)
      onStarted(created.run_id)
    } catch (err) {
      // ...and again on the server, which has the final word (400, 409, 413).
      setError(err instanceof ApiError ? err.message : 'Could not reach the server.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={onSubmit} className="space-y-5">
      <label
        onDragOver={(event) => {
          event.preventDefault() // required, or the browser refuses the drop
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition ${
          dragging ? 'border-indigo-500 bg-indigo-50' : 'border-slate-300 hover:border-slate-400'
        }`}
      >
        <input
          type="file"
          accept=".csv,.parquet,.pq"
          className="sr-only"
          aria-label="Dataset file"
          onChange={(event) => chooseFile(event.target.files?.[0])}
        />
        {file ? (
          <>
            <span className="font-medium text-slate-900">{file.name}</span>
            <span className="mt-1 text-sm text-slate-500">
              {(file.size / 1024).toFixed(0)} KB · click or drop to replace
            </span>
          </>
        ) : (
          <>
            <span className="font-medium text-slate-900">Drop a CSV or Parquet file here</span>
            <span className="mt-1 text-sm text-slate-500">or click to choose one · up to 200 MB</span>
          </>
        )}
      </label>

      <div>
        <label htmlFor="description" className="block text-sm font-medium text-slate-700">
          What do you want to predict?
        </label>
        <textarea
          id="description"
          rows={2}
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          placeholder="e.g. Predict which passengers survived"
          className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 focus:border-indigo-500 focus:outline-none"
        />
      </div>

      <div>
        <label htmlFor="target" className="block text-sm font-medium text-slate-700">
          Target column <span className="font-normal text-slate-400">(optional)</span>
        </label>
        <input
          id="target"
          value={target}
          onChange={(event) => setTarget(event.target.value)}
          placeholder="Leave empty and the agent will work it out"
          className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 focus:border-indigo-500 focus:outline-none"
        />
      </div>

      {error && (
        <p role="alert" className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </p>
      )}

      <button
        type="submit"
        disabled={!canSubmit}
        className="w-full rounded-lg bg-indigo-600 px-4 py-2.5 font-medium text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300"
      >
        {submitting ? 'Uploading…' : 'Start run'}
      </button>
    </form>
  )
}
