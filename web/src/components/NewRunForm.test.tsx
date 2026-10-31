// Tests use the form the way a person would: find things by their labels, type, upload, click.
// The network is replaced by a fake fetch, so no server is needed.

import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { fakeFetch } from '../test/fakeFetch'
import { NewRunForm } from './NewRunForm'

const csv = () => new File(['a,b\n1,2\n'], 'titanic.csv', { type: 'text/csv' })

const SAMPLES = [
  {
    name: 'mpg.csv',
    rows: 398,
    kind: 'regression',
    description: "Predict a car's fuel efficiency",
    target: 'mpg',
  },
]

afterEach(() => vi.unstubAllGlobals())

async function fillForm(file: File = csv()) {
  const user = userEvent.setup()
  await user.upload(screen.getByLabelText('Dataset file'), file)
  await user.type(screen.getByLabelText('What do you want to predict?'), 'Predict survival')
  return user
}

function postedForm(fetchMock: ReturnType<typeof fakeFetch>): FormData {
  const post = fetchMock.mock.calls.find(([, init]) => init?.method === 'POST')!
  return post[1]!.body as FormData
}

describe('NewRunForm', () => {
  it('keeps Start disabled until a file and a description are given', async () => {
    fakeFetch({ 'GET /api/samples': { status: 200, body: [] } })
    render(<NewRunForm onStarted={() => {}} />)
    const start = screen.getByRole('button', { name: 'Start run' })
    expect(start).toBeDisabled()

    await fillForm()
    expect(start).toBeEnabled()
  })

  it('refuses an unsupported file instantly, without uploading it', async () => {
    const fetchMock = fakeFetch({ 'GET /api/samples': { status: 200, body: [] } })
    render(<NewRunForm onStarted={() => {}} />)
    // applyAccept: false lets the test pick a file the file dialog would normally hide
    const user = userEvent.setup({ applyAccept: false })
    await user.upload(screen.getByLabelText('Dataset file'), new File(['x'], 'data.xlsx'))

    expect(screen.getByRole('alert')).toHaveTextContent('Unsupported file type')
    expect(screen.getByRole('button', { name: 'Start run' })).toBeDisabled()
    expect(fetchMock.mock.calls.some(([, init]) => init?.method === 'POST')).toBe(false)
  })

  it('refuses a file over the size limit', async () => {
    fakeFetch({ 'GET /api/samples': { status: 200, body: [] } })
    render(<NewRunForm onStarted={() => {}} />)
    const huge = csv()
    Object.defineProperty(huge, 'size', { value: 300 * 1024 * 1024 }) // pretend 300 MB
    await userEvent.setup().upload(screen.getByLabelText('Dataset file'), huge)

    expect(screen.getByRole('alert')).toHaveTextContent('200 MB limit')
  })

  it('uploads the form and reports the new run id', async () => {
    const fetchMock = fakeFetch({
      'GET /api/samples': { status: 200, body: [] },
      'POST /api/runs': { status: 202, body: { run_id: '20261018_120000_titanic', status: 'running' } },
    })
    const onStarted = vi.fn()
    render(<NewRunForm onStarted={onStarted} />)
    const user = await fillForm()
    await user.type(screen.getByLabelText(/Target column/), 'Survived')
    await user.click(screen.getByRole('button', { name: 'Start run' }))

    expect(onStarted).toHaveBeenCalledWith('20261018_120000_titanic')
    const sent = postedForm(fetchMock)
    expect((sent.get('file') as File).name).toBe('titanic.csv')
    expect(sent.get('description')).toBe('Predict survival')
    expect(sent.get('target')).toBe('Survived')
  })

  it("shows the server's reason when it refuses the run", async () => {
    fakeFetch({
      'GET /api/samples': { status: 200, body: [] },
      'POST /api/runs': {
        status: 409,
        body: { detail: 'A run is already in progress; try again when it finishes.' },
      },
    })
    const onStarted = vi.fn()
    render(<NewRunForm onStarted={onStarted} />)
    const user = await fillForm()
    await user.click(screen.getByRole('button', { name: 'Start run' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('A run is already in progress')
    expect(onStarted).not.toHaveBeenCalled()
  })

  it('accepts a dropped file', () => {
    fakeFetch({ 'GET /api/samples': { status: 200, body: [] } })
    render(<NewRunForm onStarted={() => {}} />)
    const dropZone = screen.getByText('Drop a CSV or Parquet file here')
    fireEvent.drop(dropZone, { dataTransfer: { files: [csv()] } })

    expect(screen.getByText('titanic.csv')).toBeInTheDocument()
  })

  it('a sample chip fills the form and starts a run without an upload', async () => {
    const fetchMock = fakeFetch({
      'GET /api/samples': { status: 200, body: SAMPLES },
      'POST /api/runs': { status: 202, body: { run_id: 'r-mpg', status: 'running' } },
    })
    const onStarted = vi.fn()
    render(<NewRunForm onStarted={onStarted} />)
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: /mpg\.csv/ }))

    expect(screen.getByLabelText('What do you want to predict?')).toHaveValue(
      "Predict a car's fuel efficiency",
    )
    expect(screen.getByLabelText(/Target column/)).toHaveValue('mpg')
    expect(screen.getByText(/398 rows · sample dataset/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Start run' }))
    expect(onStarted).toHaveBeenCalledWith('r-mpg')
    const sent = postedForm(fetchMock)
    expect(sent.get('sample')).toBe('mpg.csv')
    expect(sent.get('file')).toBeNull()
  })

  it('the X button clears the chosen file without opening the file picker', async () => {
    fakeFetch({ 'GET /api/samples': { status: 200, body: [] } })
    render(<NewRunForm onStarted={() => {}} />)
    await fillForm()
    await userEvent.setup().click(screen.getByRole('button', { name: 'Remove file' }))

    expect(screen.getByText('Drop a CSV or Parquet file here')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Start run' })).toBeDisabled()
  })

  it('hides the samples row when the server has none', async () => {
    fakeFetch({ 'GET /api/samples': { status: 500, body: {} } })
    render(<NewRunForm onStarted={() => {}} />)
    await Promise.resolve()
    expect(screen.queryByText('Or try a sample dataset')).not.toBeInTheDocument()
  })
})
