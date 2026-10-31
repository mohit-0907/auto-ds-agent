// The "start a new run" form: drop or pick a dataset (or choose a sample), describe the
// goal, optionally name the target, press Start.

import { ArrowRight, File as FileIcon, TriangleAlert, Upload, X } from 'lucide-react'
import {
  useEffect,
  useRef,
  useState,
  type DragEvent,
  type FormEvent,
  type MouseEvent,
} from 'react'
import { ApiError, checkFile, listSamples, startRun } from '../api'
import type { SampleDataset } from '../types'

interface Props {
  /** Called with the new run's ID once the server has accepted it. */
  onStarted: (runId: string) => void
}

/** What the run will start from: a file the user picked, or a server sample. */
type Source = { file: File } | { sample: SampleDataset } | null

function formatSize(bytes: number): string {
  if (bytes > 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${Math.max(1, Math.round(bytes / 1024))} KB`
}

const KICKER = 'text-[11px] font-semibold tracking-[0.08em] uppercase text-muted'
const FIELD =
  'w-full border border-divider bg-page text-ink caret-accent placeholder:text-faint focus:border-accent focus:outline-none'

export function NewRunForm({ onStarted }: Props) {
  // State: the form's memory. Changing any of these makes React redraw the form.
  const [source, setSource] = useState<Source>(null)
  const [description, setDescription] = useState('')
  const [target, setTarget] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [samples, setSamples] = useState<SampleDataset[]>([])
  const fileInput = useRef<HTMLInputElement>(null) // a handle on the hidden <input type=file>

  // Fetch the sample list once. If it fails, the samples row simply doesn't appear.
  useEffect(() => {
    let cancelled = false
    listSamples()
      .then((list) => {
        if (!cancelled && Array.isArray(list)) setSamples(list)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  // Derived, not stored (see Task 7.3): recomputed on every redraw.
  const canSubmit = source !== null && description.trim() !== '' && !submitting

  function chooseFile(candidate: File | undefined) {
    if (!candidate) return
    const problem = checkFile(candidate) // checked here for an instant answer...
    setError(problem)
    setSource(problem ? null : { file: candidate })
  }

  function chooseSample(sample: SampleDataset) {
    setSource({ sample })
    setDescription(sample.description)
    setTarget(sample.target)
    setError(null)
  }

  function clear(event: MouseEvent) {
    // The X sits inside the drop zone's <label>: without this, clicking it would also
    // open the file picker.
    event.preventDefault()
    event.stopPropagation()
    setSource(null)
    if (fileInput.current) fileInput.current.value = '' // so the same file can be re-picked
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault() // stop the browser from opening the dropped file itself
    setDragging(false)
    chooseFile(event.dataTransfer.files[0])
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault() // stop the browser's default "reload the page" on submit
    if (!source) return
    setSubmitting(true)
    setError(null)
    try {
      const created = await startRun(
        'file' in source ? { file: source.file } : { sample: source.sample.name },
        description.trim(),
        target.trim() || undefined,
      )
      onStarted(created.run_id)
    } catch (err) {
      // ...and again on the server, which has the final word (400, 409, 413).
      setError(err instanceof ApiError ? err.message : 'Could not reach the server.')
    } finally {
      setSubmitting(false)
    }
  }

  let title = 'Drop a CSV or Parquet file here'
  let subline = 'or click to choose one · .csv, .parquet · up to 200 MB'
  if (source && 'file' in source) {
    title = source.file.name
    subline = `${formatSize(source.file.size)} · click or drop to replace`
  } else if (source) {
    title = source.sample.name
    subline = `${source.sample.rows.toLocaleString('en-US')} rows · sample dataset · click or drop to replace`
  }

  const zone = dragging
    ? 'border-accent bg-accent-bg'
    : source
      ? 'border-ok-dot bg-ok-bg/45'
      : 'border-divider bg-page'
  const icon = source ? 'bg-ok-bg text-ok-fg' : 'bg-accent-bg text-accent-fg'

  return (
    <div>
      <form
        onSubmit={onSubmit}
        className="flex flex-col gap-[22px] rounded-[32px] bg-card p-7 shadow-card"
      >
        <label
          onDragOver={(event) => {
            event.preventDefault() // required, or the browser refuses the drop
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`flex cursor-pointer items-center gap-[18px] rounded-3xl border-[1.5px] border-dashed px-6 py-[26px] ${zone}`}
        >
          <input
            ref={fileInput}
            type="file"
            accept=".csv,.parquet,.pq"
            className="sr-only"
            aria-label="Dataset file"
            onChange={(event) => chooseFile(event.target.files?.[0])}
          />
          <span className={`flex h-[52px] w-[52px] flex-none items-center justify-center rounded-full ${icon}`}>
            {source ? (
              <FileIcon size={22} strokeWidth={2.75} aria-hidden="true" />
            ) : (
              <Upload size={22} strokeWidth={2.75} aria-hidden="true" />
            )}
          </span>
          <span className="flex min-w-0 flex-1 flex-col">
            <span className="truncate text-[15px] font-semibold">{title}</span>
            <span className="text-[13px] text-muted">{subline}</span>
          </span>
          {source && (
            <button
              type="button"
              onClick={clear}
              aria-label="Remove file"
              className="flex h-8 w-8 flex-none items-center justify-center rounded-full text-muted hover:bg-sel"
            >
              <X size={16} strokeWidth={2.75} aria-hidden="true" />
            </button>
          )}
        </label>

        <div className="flex flex-col gap-1.5">
          <label htmlFor="description" className="text-[13px] font-semibold">
            What do you want to predict?
          </label>
          <textarea
            id="description"
            rows={3}
            value={description}
            onChange={(event) => setDescription(event.target.value)}
            placeholder="e.g. Predict which passengers survived"
            className={`${FIELD} resize-y rounded-[20px] px-4 py-3 text-sm`}
          />
          <span className="text-xs text-muted">
            Plain language. The agent uses it to choose the task and the target.
          </span>
        </div>

        <div className="flex flex-col gap-1.5">
          <label htmlFor="target" className="text-[13px] font-semibold">
            Target column <span className="font-normal text-muted">optional</span>
          </label>
          <input
            id="target"
            value={target}
            onChange={(event) => setTarget(event.target.value)}
            placeholder="Leave empty and the agent will work it out"
            className={`${FIELD} rounded-full px-4 py-2.5 font-mono text-[13px]`}
          />
        </div>

        {error && (
          <div role="alert" className="flex items-start gap-2.5 rounded-2xl bg-bad-bg px-3.5 py-2.5 text-[13px] text-bad-fg">
            <TriangleAlert size={16} strokeWidth={2.75} className="mt-0.5 flex-none" aria-hidden="true" />
            {error}
          </div>
        )}

        <div className="flex flex-wrap items-center gap-4">
          <button
            type="submit"
            disabled={!canSubmit}
            className="inline-flex items-center gap-2 rounded-full bg-accent px-6 py-3 font-display text-[15px] text-on-accent enabled:hover:brightness-[0.93]"
          >
            {submitting ? 'Starting…' : 'Start run'}
            <ArrowRight size={16} strokeWidth={2.75} aria-hidden="true" />
          </button>
          <span className="text-xs text-muted">One run at a time · usually about a minute</span>
        </div>
      </form>

      {samples.length > 0 && (
        <div className="mt-7 flex flex-col gap-2.5">
          <span className={KICKER}>Or try a sample dataset</span>
          <div className="flex flex-wrap gap-2">
            {samples.map((sample) => (
              <button
                key={sample.name}
                type="button"
                onClick={() => chooseSample(sample)}
                className="flex flex-col items-start gap-px rounded-[18px] border border-divider px-4 py-[9px] text-left hover:bg-sel"
              >
                <span className="text-[13px] font-semibold">{sample.name}</span>
                <span className="text-[11.5px] text-muted">
                  {sample.rows.toLocaleString('en-US')} rows · {sample.kind}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
