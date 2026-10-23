import { useEffect, useRef, useState } from 'react'

/* Plotly (bundle "cartesian": líneas + heatmaps) se carga perezosamente la
   primera vez que una gráfica entra al viewport, para no penalizar la carga. */
type PlotlyLib = {
  react: (el: HTMLElement, data: unknown[], layout: unknown, config: unknown) => Promise<unknown>
  purge: (el: HTMLElement) => void
}
let plotlyPromise: Promise<PlotlyLib> | null = null
const loadPlotly = () =>
  (plotlyPromise ??= import('plotly.js-cartesian-dist-min').then((m) => (m.default ?? m) as PlotlyLib))

export const NAVY = '#0B1F3A'
export const GOLD = '#FFD200'
export const SKY = '#7cc4ff'
/** Escala azul -> amarillo coherente con la paleta. */
export const BLUE_GOLD: [number, string][] = [
  [0, '#0b1f3a'],
  [0.2, '#16365f'],
  [0.4, '#1f5a96'],
  [0.6, '#3f8fc9'],
  [0.8, '#a9c96a'],
  [1, '#ffd200'],
]

function useDark() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))
  useEffect(() => {
    const obs = new MutationObserver(() => setDark(document.documentElement.classList.contains('dark')))
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    return () => obs.disconnect()
  }, [])
  return dark
}

export function baseLayout(dark: boolean) {
  const fg = dark ? '#fff6d5' : '#0b1f3a'
  const grid = dark ? 'rgba(255,246,213,0.12)' : 'rgba(11,31,58,0.12)'
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter Variable, system-ui, sans-serif', color: fg, size: 13 },
    margin: { l: 56, r: 16, t: 40, b: 92 },
    xaxis: { gridcolor: grid, zerolinecolor: grid },
    yaxis: { gridcolor: grid, zerolinecolor: grid },
    legend: { orientation: 'h', y: -0.22, x: 0, yanchor: 'top' },
    hoverlabel: { font: { family: 'Inter Variable, system-ui' } },
  }
}

export function Plot({
  data,
  layout,
  height = 380,
  label,
}: {
  data: (dark: boolean) => unknown[]
  layout: (dark: boolean) => Record<string, unknown>
  height?: number
  label: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  const [ready, setReady] = useState(false)
  const dark = useDark()

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(
      (entries) => entries.some((e) => e.isIntersecting) && (setVisible(true), io.disconnect()),
      { rootMargin: '400px' },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])

  useEffect(() => {
    if (!visible || !ref.current) return
    let alive = true
    const el = ref.current
    loadPlotly().then((P) => {
      if (!alive) return
      const base = baseLayout(dark)
      const l = layout(dark)
      P.react(
        el,
        data(dark),
        {
          ...base,
          ...l,
          xaxis: { ...base.xaxis, ...(l.xaxis as object) },
          yaxis: { ...base.yaxis, ...(l.yaxis as object) },
          height,
          autosize: true,
        },
        { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] },
      ).then(() => alive && setReady(true))
    })
    return () => {
      alive = false
    }
  }, [visible, dark, data, layout, height])

  useEffect(() => {
    const el = ref.current
    return () => {
      if (el && plotlyPromise) plotlyPromise.then((P) => P.purge(el))
    }
  }, [])

  return (
    <figure className="relative w-full" style={{ minHeight: height }} aria-label={label} role="img">
      {!ready && (
        <div className="absolute inset-0 grid place-items-center text-sm opacity-70" aria-hidden>
          Calentando en la banca…
        </div>
      )}
      <div ref={ref} className="w-full" />
    </figure>
  )
}
