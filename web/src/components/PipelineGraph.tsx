// The pipeline as a row of pills: what is done, failed, running now, still to come or skipped.
// Every state comes from pipelineState() in stages.ts; this file only draws it.

import { Check, CornerDownRight, Repeat, X } from 'lucide-react'
import { MAX_FE_ATTEMPTS, stageLabel, type GraphNode, type NodeState, type PipelineState } from '../stages'
import { Spinner } from './Spinner'

const PILL = 'inline-flex items-center gap-[7px] rounded-full border-[1.5px] px-[13px] py-[7px] text-[12.5px] font-semibold whitespace-nowrap'

const PILL_STATE: Record<NodeState, string> = {
  done: 'border-transparent bg-ok-bg text-ok-fg',
  fail: 'border-transparent bg-bad-bg text-bad-fg',
  active: 'border-accent text-ink',
  pending: 'border-divider text-muted',
  skipped: 'border-dashed border-divider text-muted opacity-70',
}

const FALLBACK_STATE: Record<PipelineState['fallback'], string> = {
  used: 'border-transparent bg-warn-bg text-warn-fg',
  failed: 'border-transparent bg-bad-bg text-bad-fg',
  active: 'border-accent text-ink',
  idle: 'border-dashed border-divider text-muted opacity-70',
}

function Pill({ node, state }: { node: GraphNode; state: NodeState }) {
  return (
    <span className={`${PILL} ${PILL_STATE[state]}`}>
      {state === 'done' && <Check size={13} strokeWidth={3} aria-hidden="true" />}
      {state === 'fail' && <X size={13} strokeWidth={3} aria-hidden="true" />}
      {state === 'active' && <Spinner />}
      {stageLabel(node)}
      {/* Colour and icons are not enough on their own: screen readers hear the state. */}
      <span className="sr-only"> ({state})</span>
    </span>
  )
}

/** The 22 px line before a node: green once that node is done. */
function Connector({ done }: { done: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={`-ml-0.5 h-0.5 w-[22px] flex-none rounded-sm ${done ? 'bg-ok-dot' : 'bg-divider'}`}
    />
  )
}

/** One step: the connector and the node after it share a wrapper, so a wrapped row never
 *  ends with a dangling line. */
function Step({ node, state, first = false }: { node: GraphNode; state: NodeState; first?: boolean }) {
  return (
    <li className="inline-flex items-center">
      {!first && <Connector done={state === 'done'} />}
      <Pill node={node} state={state} />
    </li>
  )
}

interface Props {
  pipeline: PipelineState
  caption: string
}

export function PipelineGraph({ pipeline, caption }: Props) {
  const { nodes, attempt, fallback } = pipeline
  return (
    <section aria-label="Pipeline" className="flex flex-col gap-3.5 rounded-[28px] bg-card px-6 pt-5 pb-[22px] shadow-card">
      <div className="flex items-center justify-between gap-3">
        <span className="text-[11px] font-semibold tracking-[0.08em] text-muted uppercase">
          Pipeline
        </span>
        <span className="text-xs text-muted">{caption}</span>
      </div>

      <ol className="m-0 flex list-none flex-wrap items-center gap-y-3.5 p-0">
        <Step node="profiler" state={nodes.profiler} first />
        <Step node="router" state={nodes.router} />
        <Step node="sanity_check" state={nodes.sanity_check} />

        <li className="inline-flex items-center">
          <Connector done={nodes.fe_codegen === 'done'} />
          <div className="inline-flex flex-col gap-2 rounded-[24px] border-[1.5px] border-dashed border-divider px-3 py-2.5">
            <div className="flex items-center justify-between gap-2.5 text-[11px] text-muted">
              <span className="font-semibold">Corrective loop</span>
              <span className="font-mono">
                {attempt ? `attempt ${attempt} / ${MAX_FE_ATTEMPTS}` : `≤ ${MAX_FE_ATTEMPTS} attempts`}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <Pill node="fe_codegen" state={nodes.fe_codegen} />
              <Repeat size={16} strokeWidth={2.75} className="text-muted" aria-label="repeats with" />
              <Pill node="sandbox" state={nodes.sandbox} />
            </div>
            <div className="flex items-center gap-1.5 text-[11px] text-muted">
              <CornerDownRight size={12} strokeWidth={2.75} aria-hidden="true" />
              after {MAX_FE_ATTEMPTS} failures
              <span
                className={`inline-flex items-center rounded-full border-[1.5px] px-2.5 py-[3px] text-[11.5px] font-semibold ${FALLBACK_STATE[fallback]}`}
              >
                Fallback features
                <span className="sr-only"> ({fallback})</span>
              </span>
            </div>
          </div>
        </li>

        <Step node="arena" state={nodes.arena} />
        <Step node="artifacts" state={nodes.artifacts} />
        <Step node="reporter" state={nodes.reporter} />
      </ol>
    </section>
  )
}
