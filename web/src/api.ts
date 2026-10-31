// The typed API client: every request the frontend makes goes through this file.
// Components call these functions and never build URLs or parse responses themselves.

import type { RunCreated, RunListItem, RunSummary, SampleDataset } from './types'

// Mirrors SUPPORTED_EXTENSIONS and MAX_DATASET_MB in src/autods/config.py. The browser
// checks these for an instant answer; the server checks them again and has the final word.
export const SUPPORTED_EXTENSIONS = ['.csv', '.parquet', '.pq']
export const MAX_DATASET_MB = 200

/** A request the server answered with an error status (400, 404, 409, 413, ...). */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

/** Send a request; return the parsed JSON, or throw ApiError with the server's message. */
async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    let message = response.statusText || `Request failed (${response.status})`
    try {
      const body = await response.json()
      // FastAPI puts the reason in "detail", e.g. "A run is already in progress; ..."
      if (typeof body.detail === 'string') message = body.detail
    } catch {
      // not JSON: keep the status text
    }
    throw new ApiError(response.status, message)
  }
  return (await response.json()) as T
}

/** What a run starts from: an uploaded file, or one of the server's sample datasets. */
export type RunSource = { file: File } | { sample: string }

/** POST /api/runs: upload a dataset (or name a sample) and start the agent. */
export function startRun(
  source: RunSource,
  description: string,
  target?: string,
): Promise<RunCreated> {
  const form = new FormData() // the same "multipart form" format as curl -F
  if ('file' in source) form.append('file', source.file)
  else form.append('sample', source.sample)
  form.append('description', description)
  if (target) form.append('target', target)
  return request<RunCreated>('/api/runs', { method: 'POST', body: form })
}

/** GET /api/runs/{id}: everything about one run. */
export function getRun(runId: string): Promise<RunSummary> {
  return request<RunSummary>(`/api/runs/${encodeURIComponent(runId)}`)
}

/** GET /api/samples: datasets that can start a run without an upload. */
export function listSamples(): Promise<SampleDataset[]> {
  return request<SampleDataset[]>('/api/samples')
}

/** GET /api/runs: run history, newest first. */
export function listRuns(): Promise<RunListItem[]> {
  return request<RunListItem[]>('/api/runs')
}

/** The download URL for one of a run's artifacts (an <a href> target, not fetched by code). */
export function fileUrl(runId: string, name: string): string {
  return `/api/runs/${encodeURIComponent(runId)}/files/${encodeURIComponent(name)}`
}

/** The live-events URL for one run, for the browser's EventSource (Task 7.4). */
export function eventsUrl(runId: string): string {
  return `/api/runs/${encodeURIComponent(runId)}/events`
}

/** A file problem the browser can spot before uploading, or null if the file looks fine. */
export function checkFile(file: File): string | null {
  const dot = file.name.lastIndexOf('.')
  const extension = dot >= 0 ? file.name.slice(dot).toLowerCase() : ''
  if (!SUPPORTED_EXTENSIONS.includes(extension)) {
    return `Unsupported file type; expected one of: ${SUPPORTED_EXTENSIONS.join(', ')}.`
  }
  if (file.size === 0) return 'The file is empty.'
  if (file.size > MAX_DATASET_MB * 1024 * 1024) {
    return `File is larger than the ${MAX_DATASET_MB} MB limit.`
  }
  return null
}
