// Tests use the form the way a person would: find things by their labels, type, upload, click.
// The network is replaced by a fake fetch, so no server is needed.

import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { NewRunForm } from './NewRunForm'

const csv = () => new File(['a,b\n1,2\n'], 'titanic.csv', { type: 'text/csv' })

function fakeFetch(status: number, body: unknown) {
  const fetchMock = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }),
  )
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => vi.unstubAllGlobals())

async function fillForm(file: File = csv()) {
  const user = userEvent.setup()
  await user.upload(screen.getByLabelText('Dataset file'), file)
  await user.type(screen.getByLabelText('What do you want to predict?'), 'Predict survival')
  return user
}

describe('NewRunForm', () => {
  it('keeps Start disabled until a file and a description are given', async () => {
    render(<NewRunForm onStarted={() => {}} />)
    const start = screen.getByRole('button', { name: 'Start run' })
    expect(start).toBeDisabled()

    await fillForm()
    expect(start).toBeEnabled()
  })

  it('refuses an unsupported file instantly, without contacting the server', async () => {
    const fetchMock = fakeFetch(202, {})
    render(<NewRunForm onStarted={() => {}} />)
    // applyAccept: false lets the test pick a file the file dialog would normally hide
    const user = userEvent.setup({ applyAccept: false })
    await user.upload(screen.getByLabelText('Dataset file'), new File(['x'], 'data.xlsx'))

    expect(screen.getByRole('alert')).toHaveTextContent('Unsupported file type')
    expect(screen.getByRole('button', { name: 'Start run' })).toBeDisabled()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('refuses a file over the size limit', async () => {
    render(<NewRunForm onStarted={() => {}} />)
    const huge = csv()
    Object.defineProperty(huge, 'size', { value: 300 * 1024 * 1024 }) // pretend 300 MB
    await userEvent.setup().upload(screen.getByLabelText('Dataset file'), huge)

    expect(screen.getByRole('alert')).toHaveTextContent('200 MB limit')
  })

  it('uploads the form and reports the new run id', async () => {
    const fetchMock = fakeFetch(202, { run_id: '20261018_120000_titanic', status: 'running' })
    const onStarted = vi.fn()
    render(<NewRunForm onStarted={onStarted} />)
    const user = await fillForm()
    await user.type(screen.getByLabelText(/Target column/), 'Survived')
    await user.click(screen.getByRole('button', { name: 'Start run' }))

    expect(onStarted).toHaveBeenCalledWith('20261018_120000_titanic')
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/runs')
    expect(init.method).toBe('POST')
    const sent = init.body as FormData
    expect((sent.get('file') as File).name).toBe('titanic.csv')
    expect(sent.get('description')).toBe('Predict survival')
    expect(sent.get('target')).toBe('Survived')
  })

  it("shows the server's reason when it refuses the run", async () => {
    fakeFetch(409, { detail: 'A run is already in progress; try again when it finishes.' })
    const onStarted = vi.fn()
    render(<NewRunForm onStarted={onStarted} />)
    const user = await fillForm()
    await user.click(screen.getByRole('button', { name: 'Start run' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('A run is already in progress')
    expect(onStarted).not.toHaveBeenCalled()
  })

  it('accepts a dropped file', () => {
    render(<NewRunForm onStarted={() => {}} />)
    const dropZone = screen.getByText('Drop a CSV or Parquet file here')
    fireEvent.drop(dropZone, { dataTransfer: { files: [csv()] } })

    expect(screen.getByText('titanic.csv')).toBeInTheDocument()
  })
})
