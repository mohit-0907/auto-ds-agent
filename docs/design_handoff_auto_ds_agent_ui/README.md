# Handoff: auto-ds-agent web UI redesign

## Overview
This is a redesign of the `web/` React frontend for auto-ds-agent. It replaces the single-column layout with a tool-style app shell: a left sidebar listing past runs and a main pane. It covers these screens:
- New run
- Live run, with a pipeline graph and a step list that streams in
- Finished run: results and leaderboard
- Report
- Feature code
- All runs history
- Failure states: the fallback-recipe run and the no-model run

It also adds a light/dark theme toggle.

## About the design files
`Auto DS Agent.dc.html` is a **design reference built in HTML**. It is a working prototype of the intended look and behavior, not production code. Rebuild it inside the existing `web/` app using its current stack:
- Vite
- React 19
- TypeScript
- react-router 8
- Tailwind v4
- Vitest

Keep the existing architecture:
- `api.ts` stays the only place that makes requests.
- `types.ts` mirrors the Pydantic models.
- `stages.ts` holds the pure event→view functions.
- `useRunEvents` handles SSE.

Don't add a UI library. Open the HTML in a browser, click through every screen, and read its logic class: all the view-model derivations are in `renderVals()`.

`Current UI.dc.html` is a recreation of today's UI, for before/after comparison.

## Fidelity
**High fidelity.** Final colors, type, spacing, radii and interactions. Match it closely.

---

## Design tokens

Define these as CSS variables in `index.css`, under `:root` and `[data-theme="dark"]`. Map them into Tailwind v4 with `@theme inline`, so you can write classes like `bg-card` and `text-muted`.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#f5ead8` | `#1f1c18` | page / main pane |
| `--surface` | `#ebddc5` | `#171512` | sidebar |
| `--card` | `#faf4ea` | `#292520` | cards, table containers |
| `--text` | `#201e1d` | `#f2e8d8` | primary text |
| `--muted` | `#6b6255` | `#b0a491` | secondary text, labels |
| `--faint` | `#a39885` | `#7c7263` | line numbers, inactive fold squares |
| `--divider` | `#201e1d` @ 13% | `#f2e8d8` @ 12% | borders, rules (use `color-mix`) |
| `--sel` | `#201e1d` @ 7% | `#f2e8d8` @ 8% | hover / selected row bg |
| `--code` | `#f2e8d8` | `#191714` | code / detail blocks |
| `--accent` | `#c67139` | `#df8a54` | primary buttons, active tab underline, running state, focus ring |
| `--accent-fg` | `#8c491a` | `#f6a06b` | accent-colored text (links, "Recommended model" kicker) |
| `--accent-bg` | `#ffe1d0` | `#3d2618` | running tag bg, LLM step badges, drag-over |
| `--on-accent` | `#fbf4ea` | `#1f1c18` | text on accent fill |
| `--ok-bg / -fg / -dot` | `#e1eecc / #46522f / #7a8a5e` | `#2f3826 / #c3d4a5 / #93a675` | success (sage) |
| `--bad-bg / -fg / -dot` | `#f7dbd1 / #8f2f1c / #bf4b30` | `#46241c / #f4ab96 / #d9664a` | failure |
| `--warn-bg / -fg / -dot` | `#f6e6c0 / #6b4b0c / #c9952e` | `#3f321a / #ecc77a / #d6a640` | warnings, fallback |
| `--shadow-sm` | `0 1px 2px rgb(46 43 37 / .12)` | `0 0 0 1px rgb(242 232 216 / .07)` | card elevation |

Color rules:
- Use one accent only.
- Status colors appear only on status (tags, icons, dots); never decorate with them.
- Paragraph-size accent text uses `--accent-fg`, never `--accent`.

**Type** (load from Google Fonts):
- **Caprasimo** (400) for display: h1 at 32–36px with `-0.015em` letter-spacing, the card titles on the result screen (model name 30px, Leaderboard 20px, no-model card 24px), report h3s at 21px, and buttons in the heading font.
- **Figtree** for body. Base size 14px, line-height 1.5. Weights 400, 500, 600 and 700.
- **JetBrains Mono** (400/500) for every number, metric, timing, column name, filename and all code.
- **Kicker labels**: 11px, 600 weight, uppercase, `0.08em` tracking, `--muted`.

**Radii:**
- Pills (`999px`): buttons, inputs, tags, chips, graph nodes, segmented controls.
- 28–32px: big cards.
- 20–24px: drop zone, attempt groups, alerts.
- 14–16px: sidebar rows, step rows, code blocks.

Never use sharp corners.

**Spacing:**
- Main pane padding: `32px 36px 64px`. The New run screen uses `40px` on top.
- Content `max-width`: 1180px for run and history, 1080px for New run.
- Gaps between sections: 20–24px.
- Card padding: 22–28px.

**Icons:** Lucide (`lucide-react`) at `strokeWidth={2.75}`, 13–22px. Icons used:
- Plus, ChevronRight, Sun, Moon, Upload, File, X
- TriangleAlert, Check, ArrowRight, RotateCcw, Repeat
- CornerDownRight (fallback branch), Search, Download, Copy

**Focus:** `:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px }` everywhere.
**Disabled:** 45% opacity.
**Hover:**
- Ghost and secondary elements: `--sel` background.
- Primary button: `filter: brightness(.93)`.

---

## App shell (`App.tsx`)
The layout is a CSS grid, `268px | minmax(0,1fr)`, at full viewport height. Each column scrolls on its own.

**Sidebar** (`--surface` background, 1px right border in `--divider`), top to bottom:
1. **Brand.** A 26px accent circle with a 10px `--surface` dot in the middle, followed by "auto-ds-agent" in Caprasimo 19px.
2. **"+ New run"** primary pill button, full width.
3. **Run list header.** "RUNS" kicker on the left, a "View all" ghost link (goes to `/runs`) on the right.
4. **Run list.** Built from `listRuns()`, newest first, scrolls independently. Each row is a button with 14px radius and `9px 10px` padding, containing:
   - An 8px status dot. Running uses the accent and pulses (opacity 1 → .35 → 1, 1.2s). Finished is ok-dot, fallback is warn-dot, no model is bad-dot.
   - The dataset name (13.5px, 600 weight, ellipsized).
   - A subline in `--muted` (12px): "Hist Gradient Boosting · ROC-AUC 0.879", "Running · Write features", or "No model · Time series".
   - The relative time on the right (11px).

   The selected row gets the `--sel` background.
5. **Footer.** "Theme" label next to a Light/Dark segmented pill. The active option is filled with the accent. Save the choice in `localStorage` and apply it as `document.documentElement.dataset.theme`. Default to `prefers-color-scheme`.

**Main pane top bar** (sticky, `--bg` background, bottom border):
- A breadcrumb: "Runs › Titanic-Dataset.csv", or "New run".
- On run pages, the run's status tag on the right.

Add a route `/runs` for the history page. Keep `/` and `/runs/:runId`.

## Screen: New run (`/`)
The layout is two flex columns that wrap. The form takes `flex: 1 1 420px`. The aside takes `flex: 0 1 300px` with `min-width: 260px` and moves below the form on narrow screens.

**Header:**
- h1: "Start a new run".
- Below it, in `--muted` at 15px with `max-width: 54ch`: "Upload a table and say what you want to predict. The agent profiles it, writes and repairs feature code, benchmarks six models and explains its pick."

**Form card** (`--card`, 32px radius, 28px padding, 22px gap):
- **Drop zone.** A `<label>` wrapping a hidden file input, `accept=".csv,.parquet,.pq"`. Horizontal layout:
  - A 52px circle icon: Upload when empty, File once chosen.
  - A title: "Drop a CSV or Parquet file here", or the file name once chosen.
  - A subline: "or click to choose one · .csv, .parquet · up to 200 MB", or "{size} · click or drop to replace" once chosen.
  - An X button to clear the file.

  States:

  | State | Border | Background | Icon circle |
  |---|---|---|---|
  | Empty | 1.5px dashed `--divider` | `--bg` | `--accent-bg` |
  | Drag-over | accent | `--accent-bg` | `--accent-bg` |
  | File chosen | ok-dot | ok-bg at 45% | ok |

  Validate with the existing `checkFile()`.
- **Description.** Label "What do you want to predict?", a textarea with 3 rows and 20px radius. Placeholder: "e.g. Predict which passengers survived". Helper text: "Plain language. The agent uses it to choose the task and the target."
- **Target column.** Label "Target column" plus a muted "optional". A pill input in mono 13px. Placeholder: "Leave empty and the agent will work it out".
- **Error alert.** `role="alert"`, `--bad-bg` / `--bad-fg`, 16px radius, TriangleAlert icon. Shows `checkFile` errors and `ApiError.message`.
- **Submit.** A "Start run →" primary pill (Caprasimo 15px), disabled until a file is chosen and the description isn't blank. Next to it in muted text: "One run at a time · usually about a minute".

**Sample datasets** (below the card):
- A kicker, "Or try a sample dataset", above a row of pill-ish chips: dataset name in 600 weight plus a meta line like "891 rows · binary".
- In the prototype a chip fills in the form. Real version: either add `GET /api/samples` plus a `sample` form field, or hide the section. Ask the owner.

**Aside: "What the agent does"** is a numbered list of 7 steps:
- Each number sits in a 28px mono circle. LLM steps have an accent-bg circle plus a small "LLM" badge; non-LLM steps have a `--sel` circle with muted text.
- Steps (LLM ones marked):
  1. Profile data — "Column types, missing values, cardinality."
  2. Choose task (LLM) — "Classification or regression, and the target."
  3. Check decision — "Stops early on out-of-scope tasks."
  4. Write features (LLM) — "Pandas code, run in a sandbox. Errors go back to the LLM, up to 3 attempts."
  5. Fallback features — "A fixed recipe if all attempts fail."
  6. Benchmark models — "Six models on the same 5 folds."
  7. Write report (LLM) — "Every number checked against the results."
- Footnote: "The LLM makes three judgments. Every metric, statistic and training run is plain Python."

Profile data shows an "LLM-free" label in the copy, but it's only visible in the prototype data, not the UI. Skip it.

## Screen: Run (`/runs/:runId`)
**Header:**
- h1 with the dataset name.
- The description in quotes, `--muted`.
- Outlined pill chips, each with a key in `--muted` and a mono value: task (e.g. "Binary classification"), target (`target_column`, or "inferred" when it was null), shape (`rows × cols`) and duration.
- While running: "elapsed 00:38" in mono on the right.

**Pipeline graph card** (`--card`, 28px radius):
- A "PIPELINE" kicker, with a caption on the right:
  - Running: "Live · 5 of ~9 stages".
  - Finished: "Completed in 1:00".
  - Fallback: "Recovered with the fallback recipe".
  - No model: "Stopped at the sanity check".
- A wrapping flex row of nodes: Profile data → Choose task → Check decision → [Corrective loop] → Benchmark models → Save model → Write report.
- Between nodes, a 22×2px connector. It's sage when the next node is done, otherwise `--divider`. Keep each connector in the same wrapper as the node after it, so a wrapped row never ends with a hanging line.
- Node pill (12.5px, 600 weight, 1.5px border). States:

  | State | Background | Border | Content |
  |---|---|---|---|
  | done | ok-bg | none | Check icon, ok-fg text |
  | fail | bad-bg | none | X icon, bad-fg text |
  | active | transparent | accent | 12px spinner (accent top border, 0.8s linear rotation) |
  | pending | transparent | `--divider` | muted text |
  | skipped | transparent | dashed `--divider` | 70% opacity |
- **Corrective loop:** a dashed rounded box (24px radius) containing:
  - A header, "Corrective loop", with "attempt N / 3" (mono) on the right.
  - The Write features ⇄ Run in sandbox nodes, joined by a Repeat icon.
  - A footer: "↳ after 3 failures", then a small "Fallback features" pill. The pill uses the warn colors when it was used, the accent border while it's active, and dashed otherwise.
- Derive the whole graph state from `events` in `stages.ts`, as a pure function with tests. The next/active node is: the successor of the last event; after a failed sandbox, `fe_codegen` again (or `fallback_fe` once attempt 3 has failed); after a passing sandbox, `arena`.

**Tabs** (underline style, 2px accent underline on the active tab, muted text otherwise):
- **Results.** Disabled while running.
- **Steps.** Shows a mono count. This is the default while running.
- **Report.** Disabled while running.
- **Feature code.** Disabled if the run has no FE code.

When the run finishes, switch automatically to Results.

### Tab: Steps
A card holding a list of step rows. Each row is a button laid out as a grid: `24px | 150px | 1fr | auto | 18px`. Columns:
- A 24px status circle: Check on ok-bg, X on bad-bg, or a spinner on accent-bg.
- The stage label, using the existing `stages.ts` labels (600 weight).
- The event text, in `--muted`; in bad-fg when the step failed.
- The time since start, mono `mm:ss`.
- A chevron that rotates 90° when expanded. Hide it when the event has no detail.

Expanding a row shows a `<pre>` (mono 12px, line-height 1.6, `--code` background, 16px radius, indented 46px). Failures use bad-fg text.

Retries: consecutive `fe_codegen` / `sandbox` events with the same `attempt` go into an **attempt group**: 22px radius, 1.5px border, `--bg` background. Its header reads "ATTEMPT N" next to a status tag: passed (ok), failed (bad) or in progress (accent, with an accent border on the group).

While running, append a ghost row (the next stage's label, "working…", and a spinner) followed by `aria-live="polite"` text: "Streaming live from the agent. Results appear here when the run finishes."

**Backend gap:** `StageEvent` has no detail payload. Add an optional `detail: string | null` (plus `detail_kind: 'code' | 'error' | null`) in `progress.py` and `types.ts`. Contents:
- router: the decision JSON
- fe_codegen: the head of the code
- sandbox: the traceback, or the validation checklist
- arena: a one-line summary

### Tab: Results (finished with a model)
1. **Warnings strip** (only if `warnings.length`): warn-bg, 20px radius, one TriangleAlert row per warning.
2. **Two-column grid** (`1.35fr | 1fr`, 20px gap):
   - **Recommendation card:**
     - "RECOMMENDED MODEL" kicker in `--accent-fg`, then the model name (Caprasimo 30px).
     - The score in mono 44px at line-height 1, with "{metric} ± {std} · 5-fold CV" beneath it.
     - Next to the score, the lift in mono 15px ok-fg. For metrics where higher is better: "+0.379", with "over 0.500 baseline (Dummy (class priors))". For error metrics: "−64.0% error", with "vs 7.805 baseline (Dummy (mean))".
     - A divider, then **"Why this model"**: a one-sentence explanation, then one row per simpler (or higher-scoring) model: "vs LightGBM", five 12px rounded squares (filled sage for each fold won, outlined for each fold lost) and a verdict ("won 4/5" in ok-fg, or "only 3/5 · kept HGB" in muted).
     - Footnote: "Rule: a higher-scoring model must beat every simpler model on at least 4 of 5 folds, otherwise the simpler one wins."
     - **Backend gap:** the per-pair fold wins are computed in the arena's selection step; expose them on `RunSummary`, e.g. `selection: {vs_model_id, folds_won, kept: boolean}[]` plus `selection_reason`.
   - **Artifacts card:** a list of download buttons, one per file in `RunSummary.files`:
     - A 36px circle with a File icon.
     - The filename in mono and a short description.
     - The size, then a Download icon in accent-fg.
     - Files: `best_pipeline.joblib`, `fe_code.py`, `report.md`, `leaderboard.csv`.
     - Footer row: "LLM cost" with `$0.0412` in mono.
     - Needs `GET /api/runs/{id}/files/{name}`, if it doesn't exist yet.
3. **Leaderboard card:**
   - "Leaderboard" (Caprasimo 20px), with a caption on the right: "Same 5 folds for every model · ranked by ROC-AUC, higher is better".
   - The table scrolls horizontally, `min-width: 820px`. Columns:
     - `#`
     - Model, with tags: "selected" (ok) and "baseline" (outlined)
     - Every metric as mean in mono 12.5px, followed by `±std` in muted 11px. The primary metric is first, gets an ↑/↓ in its header and 600 weight.
     - **Folds**
     - Train s
     - Predict ms
   - **Folds column:** a 120px track (a 2px `--divider` line) with one 8px dot per `fold_scores` value. The scale is shared across all non-baseline models, with 8% padding. Dots are sage for the selected model and muted otherwise, at 85% opacity. Baselines show no dots.
   - The selected model's row is tinted with ok-bg at 55%.
   - Caption: "Folds: each dot is one cross-validation fold on a shared scale (0.835 – 0.905 ROC-AUC); tighter clusters mean steadier models."

### Results: no-model state (`best_model_id == null`, e.g. out-of-scope task)
- A single card with a 48px bad-bg circle and a TriangleAlert icon.
- A "ENDED WITHOUT A MODEL" kicker in bad-fg.
- A Caprasimo 24px title, e.g. "Time-series forecasting is out of scope".
- An explanation (from `error` or the report summary).
- Two buttons: "Read the report" (secondary) and "Try another dataset" (primary, goes to `/`).

The pipeline graph shows Check decision as fail and every later node as skipped. Status tag: "No model" (bad).

### Fallback state
- The sandbox node and attempts 1–3 show as failed, and "Fallback features" uses the warn colors.
- Status tag: "Finished · fallback" (warn).
- The warnings strip shows the fallback warning.
- The code tab tag reads "fallback recipe" (warn).

### Tab: Report
- A card with `36px 44px` padding, holding an `<article>` with `max-width: 720px`.
- Render `report_markdown`:
  - Skip the h1 (it duplicates the page header).
  - The italic `_Generated…_` line becomes a 12.5px muted italic meta line.
  - `##` headings become Caprasimo 21px with `margin: 26px 0 8px`.
  - Paragraphs: 14.5px, line-height 1.65.
  - List items get an accent bullet.
  - Replace the markdown leaderboard table with a secondary pill: "Leaderboard table — open in Results →".
- Use `react-markdown` or a tiny renderer. Keep it light.

### Tab: Feature code
- A header row: `fe_code.py` (mono), a status tag ("attempt 1 · passed validation" in ok, or "fallback recipe" in warn), "{n} lines", and a Copy button that uses the clipboard.
- The body has the `--code` background, mono 12.5px, line-height 1.7, and horizontal scrolling. Line numbers sit in a 52px right-aligned column in `--faint` and are not selectable. Comment lines are muted.
- Needs the FE code in `RunSummary`, either as a new `fe_code` field or fetched from the files endpoint.

## Screen: All runs (`/runs`)
- h1: "All runs".
- Below it, muted: "Every run keeps its report, leaderboard, feature code and saved model."
- On the right, a search pill ("Search datasets") and a status segmented filter: All / Running / Finished / No model.
- A table card (`min-width: 720px`, scrolls horizontally). Columns:
  - Dataset, with the description as a muted ellipsized subline
  - Task
  - Target (mono)
  - Best model (nowrap)
  - Score, as a muted metric name followed by a mono value
  - Status tag, with a dot
  - Started
- The whole row is clickable and opens the run.
- Empty state: "No runs match."
- **Backend gap:** `RunListItem` needs `description`, `target_column`, `best_model_name`, `primary_metric` and `best_score`.

## Status tag mapping
`RunStatus` + outcome:

| Status | Tag | Colors |
|---|---|---|
| `running` | "Running" | accent-bg / accent-fg, with a pulsing accent dot |
| `finished` + model | "Finished" | ok |
| `finished` + fallback warning | "Finished · fallback" | warn |
| `finished` with no model, or `failed` | "No model" (or "Failed") | bad |
| `interrupted` | "Interrupted" | warn |

## Interactions & state
- **Theme:** `useTheme()` hook (localStorage plus `data-theme` on `<html>`).
- **New run:** local form state (`file`, `description`, `target`, `error`, `dragging`, `busy`). On submit, call `startRun()` and navigate to `/runs/{run_id}`. On error, show the alert. A 409 response means "a run is already in progress".
- **Run:**
  - `getRun()` on mount.
  - While `running`, `useRunEvents(eventsUrl)` streams events into the steps list and the graph.
  - On the terminal event, refetch the summary and switch to Results.
  - Keep the expanded-row set (by event index) and the active tab in local state. Optionally keep the tab in `?tab=`.
- **Sidebar list:** refetch on route change, and on an interval while any run is running.
- **Animations:** only three:
  - the spinner (0.8s linear)
  - the status-dot pulse (1.2s ease-in-out)
  - the chevron rotation (150ms)

  Respect `prefers-reduced-motion`.
- **Responsive:** the form and aside wrap, tables scroll horizontally, and the graph wraps. Below about 760px, collapse the sidebar into a drawer.

## Suggested component breakdown
- `AppShell`
- `Sidebar`
- `RunListItem`
- `ThemeToggle`
- `NewRunForm` (restyle the existing one)
- `PipelineAside`
- `RunHeader`
- `PipelineGraph`
- `RunTabs`
- `Timeline`: extend it with expandable rows and attempt groups (the existing grouping logic stays in `stages.ts`)
- `ResultsTab`, containing:
  - `WarningsStrip`
  - `RecommendationCard`
  - `WhyThisModel`
  - `ArtifactsCard`
  - `Leaderboard`
  - `FoldDots`
- `NoModelCard`
- `ReportTab`
- `CodeTab`
- `HistoryPage`
- `StatusTag`

Add a Vitest test for each new pure function: graph state, the fold-dot scale, the lift text and the status mapping.

## Files
- `Auto DS Agent.dc.html`: the full interactive prototype. Its props let you switch theme and the starting screen (`run`, `live`, `new`, `history`, `failed`). All derivation logic is in the class's `renderVals()`.
- `Current UI.dc.html`: today's UI, for reference.
- `styles.css`: the Organic design-system token sheet the prototype was themed from.
