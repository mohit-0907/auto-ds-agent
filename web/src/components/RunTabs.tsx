// Underline tabs. Each tab is a button; the page decides which one is active and what it shows.

export interface TabSpec<Id extends string> {
  id: Id
  label: string
  count?: number
  disabled?: boolean
}

interface Props<Id extends string> {
  tabs: TabSpec<Id>[]
  active: Id
  onSelect: (id: Id) => void
}

// <Id extends string> is a type parameter: the component works for any set of tab ids,
// and TypeScript then checks that `active` and `onSelect` use ids from that same set.
export function RunTabs<Id extends string>({ tabs, active, onSelect }: Props<Id>) {
  return (
    <div role="tablist" aria-label="Run views" className="flex gap-1 border-b border-divider">
      {tabs.map((tab) => {
        const selected = tab.id === active
        return (
          <button
            key={tab.id}
            type="button"
            role="tab"
            id={`tab-${tab.id}`}
            aria-selected={selected}
            aria-controls={`panel-${tab.id}`}
            disabled={tab.disabled}
            onClick={() => onSelect(tab.id)}
            className={`-mb-px inline-flex items-center gap-2 border-b-2 px-3.5 py-2.5 text-[13.5px] font-semibold hover:text-ink ${
              selected ? 'border-accent text-ink' : 'border-transparent text-muted'
            }`}
          >
            {tab.label}
            {tab.count !== undefined && (
              <span className="font-mono text-[11px] font-medium text-muted">{tab.count}</span>
            )}
          </button>
        )
      })}
    </div>
  )
}
