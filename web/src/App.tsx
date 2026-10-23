import { lazy, Suspense, useCallback, useState } from 'react'
import { Eagle } from './components/Eagle'
import { ThemeToggle, useActiveSection, useScrollProgress, type NavItem } from './components/Layout'
import { useJson, type DqnData, type TabularData } from './data'

// Las secciones (KaTeX, Plotly, modo interactivo) van en un chunk aparte para
// que la primera pintura sólo dependa del encabezado.
const Body = lazy(() => import('./Body'))

const NAV: NavItem[] = [
  { id: 'intro', n: 1, title: 'Intro y objetivos' },
  { id: 'mdp', n: 2, title: 'Modelado del MDP' },
  { id: 'ecuaciones', n: 3, title: 'Ecuaciones' },
  { id: 'pseudocodigo', n: 4, title: 'Pseudocódigo + modo interactivo' },
  { id: 'convergencia', n: 5, title: 'Entregable 1 · Convergencia' },
  { id: 'heatmaps', n: 6, title: 'Entregable 2 · Mapas de calor' },
  { id: 'robotica', n: 7, title: 'Entregable 3 · Robótica real' },
  { id: 'dqn-teoria', n: 8, title: 'DQN · Teoría' },
  { id: 'dqn-curva', n: 9, title: 'DQN 1 · Curva de aprendizaje' },
  { id: 'dqn-combustible', n: 10, title: 'DQN 2 · Combustible' },
  { id: 'escalabilidad', n: 11, title: 'DQN 3 · Escalabilidad' },
  { id: 'conclusiones', n: 12, title: 'Conclusiones y datos' },
]
const IDS = NAV.map((n) => n.id)

function Hero({ tab, dqn }: { tab: TabularData | null; dqn: DqnData | null }) {
  const s = tab?.algos.sarsa.final_ma100.mean
  const q = tab?.algos.qlearning.final_ma100.mean
  return (
    <header className="jersey-stripes relative overflow-hidden border-b border-[var(--line)]">
      <div className="mx-auto grid max-w-6xl items-center gap-8 px-5 py-14 md:grid-cols-[1fr_auto] md:py-20">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.25em] text-[var(--accent)]">
            Práctica 7 · Aprendizaje por refuerzo
          </p>
          <h1 className="mt-3 font-display text-5xl leading-[0.95] tracking-wide md:text-7xl">
            SARSA, Q-learning <br className="hidden sm:block" />y Deep Q-Learning
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-relaxed text-[var(--fg-muted)]">
            Resolución paso a paso: un robot que <strong className="text-[var(--fg)]">busca el gol</strong> en{' '}
            <em>El Paso del Contrabandista</em>, y un módulo lunar que aprende a aterrizar con una red neuronal.
          </p>
          <p className="mt-2 text-sm opacity-80">Basado en la práctica de los Drs. Alexandro López González y Julio Caballero Mora (22 de octubre de 2026).</p>
          <dl className="mt-7 grid max-w-xl grid-cols-3 gap-3">
            {[
              ['SARSA', s !== undefined ? s.toFixed(1) : '…', 'pts/partido (entren.)'],
              ['Q-learning', q !== undefined ? q.toFixed(1) : '…', 'pts/partido (entren.)'],
              ['DQN greedy', dqn ? dqn.eval.mean.toFixed(1) : '…', `ep. ${dqn?.solved_at ?? '…'} → MA ≥ 200`],
            ].map(([k, v, sub]) => (
              <div key={k} className="card px-3 py-2">
                <dt className="text-xs uppercase tracking-wider opacity-80">{k}</dt>
                <dd className="font-display text-3xl text-[var(--accent)]">{v}</dd>
                <dd className="text-[11px] opacity-80">{sub}</dd>
              </div>
            ))}
          </dl>
        </div>
        <Eagle className="mx-auto w-48 drop-shadow-[0_10px_30px_rgba(255,210,0,0.25)] md:w-72" />
      </div>
    </header>
  )
}

export default function App() {
  const tab = useJson<TabularData>('tabular.json')
  const dqn = useJson<DqnData>('dqn.json')
  const [bodyReady, setBodyReady] = useState(false)
  const onReady = useCallback(() => setBodyReady(true), [])
  const active = useActiveSection(IDS, bodyReady)
  const progress = useScrollProgress()

  return (
    <div className="pitch-texture min-h-screen">
      <a href="#contenido" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded focus:bg-gold-500 focus:px-3 focus:py-2 focus:text-navy-900">
        Saltar al contenido
      </a>
      <div className="fixed inset-x-0 top-0 z-40 h-1 bg-transparent" aria-hidden>
        <div className="h-full origin-left bg-gold-500" style={{ transform: `scaleX(${progress})` }} />
      </div>
      <div className="sticky top-0 z-30 flex items-center justify-between gap-3 border-b border-[var(--line)] bg-[var(--bg)]/90 px-4 py-2 backdrop-blur lg:hidden">
        <details className="relative">
          <summary className="cursor-pointer list-none rounded-lg border border-[var(--line)] px-3 py-1.5 text-sm font-semibold">
            ☰ {NAV.find((n) => n.id === active)?.n}. {NAV.find((n) => n.id === active)?.title}
          </summary>
          <nav aria-label="Secciones" className="card absolute left-0 mt-2 w-72 p-2 shadow-xl">
            {NAV.map((n) => (
              <a key={n.id} href={`#${n.id}`} className="block rounded px-2 py-1.5 text-sm hover:bg-[var(--bg)]">
                {n.n}. {n.title}
              </a>
            ))}
          </nav>
        </details>
        <ThemeToggle />
      </div>
      <Hero tab={tab} dqn={dqn} />
      <div className="mx-auto flex max-w-7xl gap-10 px-5">
        <aside className="sticky top-0 hidden h-screen w-64 shrink-0 overflow-y-auto py-10 lg:block">
          <div className="mb-6 flex items-center justify-between">
            <span className="font-display text-2xl tracking-wide text-[var(--accent)]">Alineación</span>
            <ThemeToggle />
          </div>
          <nav aria-label="Secciones">
            <ol className="space-y-1 border-l border-[var(--line)]">
              {NAV.map((n) => {
                const on = n.id === active
                return (
                  <li key={n.id}>
                    <a
                      href={`#${n.id}`}
                      aria-current={on ? 'true' : undefined}
                      className={`-ml-px block border-l-2 py-1.5 pl-3 text-sm transition-colors ${
                        on ? 'border-gold-500 font-semibold text-[var(--accent)]' : 'border-transparent text-[var(--fg-muted)] hover:text-[var(--fg)]'
                      }`}
                    >
                      <span className="mr-1.5 font-mono text-xs opacity-80">{String(n.n).padStart(2, '0')}</span>
                      {n.title}
                    </a>
                  </li>
                )
              })}
            </ol>
          </nav>
          <div className="mt-6 h-1.5 w-full overflow-hidden rounded bg-[var(--bg-elev)]" aria-hidden>
            <div className="h-full bg-gold-500" style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
          <p className="mt-1 text-xs opacity-80">Minuto {Math.round(progress * 90)}′ del partido</p>
        </aside>
        <main id="contenido" className="min-w-0 flex-1">
          <Suspense fallback={<div className="min-h-screen py-20 text-center opacity-80">Saliendo del vestidor…</div>}>
            <Body tab={tab} dqn={dqn} onReady={onReady} />
          </Suspense>
        </main>
      </div>
    </div>
  )
}
