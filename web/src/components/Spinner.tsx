// The design's one spinner: a ring with an accent top edge, turning every 0.8 s.

export function Spinner({ size = 12 }: { size?: number }) {
  return (
    <span
      aria-hidden="true"
      style={{ width: size, height: size }}
      className="inline-block flex-none animate-spin-fast rounded-full border-2 border-divider border-t-accent"
    />
  )
}
