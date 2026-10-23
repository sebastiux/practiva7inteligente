import { useEffect, useRef, useState, type ReactNode } from 'react'

export interface NavItem {
  id: string
  n: number
  title: string
}

export function Section({ id, n, title, kicker, children }: { id: string; n: number; title: string; kicker?: string; children: ReactNode }) {
  return (
    <section aria-labelledby={`${id}-h`} className="border-b border-[var(--line)] py-14 md:py-20">
      <div className="mb-6 flex items-end gap-4">
        <span className="font-display text-6xl leading-none text-[var(--accent)] opacity-90 md:text-7xl" aria-hidden>
          {String(n).padStart(2, '0')}
        </span>
        <div>
          {kicker && <p className="text-xs font-semibold uppercase tracking-[0.2em] opacity-80">{kicker}</p>}
          <h2 id={`${id}-h`} className="font-display text-4xl leading-tight tracking-wide md:text-5xl">
            {title}
          </h2>
        </div>
      </div>
      <div className="prose-rl max-w-none">{children}</div>
    </section>
  )
}

/** Monta la sección sólo cuando se acerca al viewport (o si se navega a ella).
 *  El id vive en este contenedor estable para que el índice lateral funcione. */
export function Deferred({ id, minHeight = 900, children }: { id: string; minHeight?: number; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null)
  const [show, setShow] = useState(() => window.location.hash === `#${id}`)
  useEffect(() => {
    if (show || !ref.current) return
    const io = new IntersectionObserver((e) => e.some((x) => x.isIntersecting) && setShow(true), { rootMargin: '800px 0px' })
    io.observe(ref.current)
    const onHash = () => window.location.hash === `#${id}` && setShow(true)
    window.addEventListener('hashchange', onHash)
    return () => (io.disconnect(), window.removeEventListener('hashchange', onHash))
  }, [show, id])
  return (
    <div id={id} ref={ref} className="scroll-mt-16" style={show ? undefined : { minHeight }}>
      {show ? children : null}
    </div>
  )
}

export function Callout({ children, tone = 'gold' }: { children: ReactNode; tone?: 'gold' | 'red' | 'sky' }) {
  const border = tone === 'gold' ? 'border-l-gold-500' : tone === 'red' ? 'border-l-red-400' : 'border-l-sky-400'
  return <div className={`card my-5 border-l-4 ${border} px-5 py-3`}>{children}</div>
}

export function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card px-4 py-3">
      <p className="text-xs uppercase tracking-wider opacity-80">{label}</p>
      <p className="font-display text-3xl text-[var(--accent)]">{value}</p>
      {sub && <p className="text-xs opacity-80">{sub}</p>}
    </div>
  )
}

export function useActiveSection(ids: string[], ready = true) {
  const [active, setActive] = useState(ids[0])
  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) => {
        const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
        if (vis[0]) setActive(vis[0].target.id)
      },
      { rootMargin: '-20% 0px -70% 0px' },
    )
    ids.forEach((id) => {
      const el = document.getElementById(id)
      if (el) io.observe(el)
    })
    return () => io.disconnect()
  }, [ids, ready])
  return active
}

export function useScrollProgress() {
  const [p, setP] = useState(0)
  useEffect(() => {
    let raf = 0
    const on = () => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(() => {
        const h = document.documentElement
        const max = h.scrollHeight - h.clientHeight
        setP(max > 0 ? h.scrollTop / max : 0)
      })
    }
    on()
    window.addEventListener('scroll', on, { passive: true })
    return () => window.removeEventListener('scroll', on)
  }, [])
  return p
}

export function ThemeToggle() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    try {
      localStorage.setItem('theme', dark ? 'dark' : 'light')
    } catch {
      /* almacenamiento no disponible */
    }
  }, [dark])
  return (
    <button
      type="button"
      onClick={() => setDark((d) => !d)}
      className="rounded-full border border-[var(--line)] px-3 py-1 text-sm"
      aria-label={dark ? 'Cambiar a modo claro' : 'Cambiar a modo oscuro'}
    >
      {dark ? '☀ Claro' : '☾ Oscuro'}
    </button>
  )
}
