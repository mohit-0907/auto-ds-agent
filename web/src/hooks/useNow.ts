// The current time, re-read every second while `ticking` -- for the live "elapsed" clock.

import { useEffect, useState } from 'react'

export function useNow(ticking: boolean): Date {
  const [now, setNow] = useState(() => new Date())

  useEffect(() => {
    if (!ticking) return
    const timer = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(timer) // stop ticking when the run ends or the page closes
  }, [ticking])

  return now
}
