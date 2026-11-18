// "What the agent does": the seven stages, with the three LLM steps marked.

const STEPS: { label: string; llm: boolean; text: string }[] = [
  { label: 'Profile data', llm: false, text: 'Column types, missing values, cardinality.' },
  { label: 'Choose task', llm: true, text: 'Classification or regression, and the target.' },
  { label: 'Check decision', llm: false, text: 'Stops early on out-of-scope tasks.' },
  {
    label: 'Write features',
    llm: true,
    text: 'Pandas code, run in a sandbox. Errors go back to the LLM, up to 3 attempts.',
  },
  { label: 'Fallback features', llm: false, text: 'A fixed recipe if all attempts fail.' },
  { label: 'Benchmark models', llm: false, text: 'Six models on the same 5 folds.' },
  { label: 'Write report', llm: true, text: 'Every number checked against the results.' },
]

export function PipelineAside() {
  return (
    <aside className="flex min-w-[260px] flex-[0_1_300px] flex-col gap-3.5 pt-1.5">
      <span className="text-[11px] font-semibold tracking-[0.08em] text-muted uppercase">
        What the agent does
      </span>
      <ol className="m-0 flex list-none flex-col p-0">
        {STEPS.map((step, i) => (
          <li key={step.label} className="grid grid-cols-[28px_1fr] gap-3 pb-4">
            <span
              className={`flex h-7 w-7 items-center justify-center rounded-full font-mono text-xs font-medium ${
                step.llm ? 'bg-accent-bg text-accent-fg' : 'bg-sel text-muted'
              }`}
            >
              {i + 1}
            </span>
            <span className="flex flex-col pt-[3px]">
              <span className="text-[13.5px] font-semibold">
                {step.label}
                {step.llm && (
                  <span className="ml-1 text-[11px] font-medium text-accent-fg">LLM</span>
                )}
              </span>
              <span className="text-[12.5px] text-pretty text-muted">{step.text}</span>
            </span>
          </li>
        ))}
      </ol>
      <p className="m-0 text-xs text-pretty text-muted">
        The LLM makes three judgments. Every metric, statistic and training run is plain Python.
      </p>
    </aside>
  )
}
