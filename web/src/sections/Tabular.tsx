import { useCallback, useMemo, useState } from 'react'
import { GridView } from '../components/GridView'
import { Callout, Section, Stat } from '../components/Layout'
import { BLUE_GOLD, GOLD, Plot, SKY } from '../components/Plot'
import { Tex } from '../components/Tex'
import { fmt, pct, type AlgoResult, type TabularData } from '../data'
import { ACTION_DELTAS, ACTION_NAMES } from '../rl/gridworld'

const LABEL = { qlearning: 'Q-learning', sarsa: 'SARSA' } as const
const COLOR = { qlearning: GOLD, sarsa: SKY } as const

/** Sigue la política greedy (sin resbalón) desde A para dibujar la ruta típica. */
function followPolicy(map: string[], policy: number[][], start: [number, number]): [number, number][] {
  const path: [number, number][] = [start]
  let [r, c] = start
  for (let k = 0; k < 80; k++) {
    const [dr, dc] = ACTION_DELTAS[policy[r][c]]
    r += dr
    c += dc
    if (r < 0 || c < 0 || r >= map.length || c >= map[0].length || map[r][c] === '#') break
    path.push([r, c])
    if (map[r][c] === 'B') break
  }
  return path
}

export function IntroSection() {
  return (
    <Section id="intro" n={1} title="Intro y objetivos" kicker="El silbatazo inicial">
      <p>
        Esta práctica compara dos formas de aprender a jugar el mismo partido. Un robot móvil con 8 direcciones de movimiento
        debe llevar el balón del punto <strong>A</strong> a la portería <strong>B</strong> en un almacén con obstáculos, y el
        piso resbala. <strong>El agente busca el gol</strong>, y cada choque es una tarjeta roja que termina el partido.
      </p>
      <p>
        Formalmente el problema es un <strong>Proceso de Decisión de Markov (MDP)</strong>{' '}
        <Tex>{String.raw`(\mathcal S,\mathcal A,P,R,\gamma)`}</Tex>: el siguiente estado depende sólo del estado y la acción
        actuales. El agente aprende una función de valor <Tex>{'Q(s,a)'}</Tex>, la recompensa futura descontada esperada si en
        el estado <Tex>s</Tex> ejecuta <Tex>a</Tex>.
      </p>
      <div className="grid gap-4 md:grid-cols-2">
        <Callout>
          <p className="!my-1 font-semibold text-[var(--fg)]">On-policy (SARSA)</p>
          <p className="!my-1">
            Evalúa y mejora <em>la misma</em> política con la que juega, exploración incluida. Si a veces tira a lo loco
            (ε-greedy), lo toma en cuenta al valorar cada jugada.
          </p>
        </Callout>
        <Callout tone="sky">
          <p className="!my-1 font-semibold text-[var(--fg)]">Off-policy (Q-learning)</p>
          <p className="!my-1">
            Juega con una política exploratoria pero aprende el valor de la política <em>óptima greedy</em>, como si en el
            futuro siempre fuera a ejecutar la mejor jugada.
          </p>
        </Callout>
      </div>
      <p className="!mb-1 font-semibold text-[var(--fg)]">Objetivos de la práctica</p>
      <ul>
        <li>Modelar la navegación como un MDP discreto con ruido de resbalamiento.</li>
        <li>Implementar y diferenciar SARSA y Q-learning, y medir su <strong>comportamiento de riesgo</strong>.</li>
        <li>Visualizar convergencia (curvas de recompensa) y trayectorias (mapas de calor).</li>
        <li>Escalar a estados continuos con Deep Q-Learning en LunarLander.</li>
      </ul>
    </Section>
  )
}

export function MdpSection({ data }: { data: TabularData | null }) {
  const map = data?.config.map
  const paths = useMemo(() => {
    if (!data) return []
    const { start } = data.config
    return [
      { cells: followPolicy(data.config.map, data.algos.qlearning.policy_seed0, start), color: '#ff5a5a', label: 'corta' },
      { cells: followPolicy(data.config.map, data.algos.sarsa.policy_seed0, start), color: '#FFF6D5', label: 'larga', dashed: true },
    ]
  }, [data])
  return (
    <Section id="mdp" n={2} title="Modelado del MDP" kicker="La cancha: El Paso del Contrabandista">
      <div className="grid items-start gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div>
          <p>
            <strong>Estados</strong> <Tex>{String.raw`\mathcal S`}</Tex>: las 400 celdas de una malla de 20×20, con{' '}
            <Tex>{'s=(\\text{fila},\\text{col})'}</Tex>. Una <strong>barrera vertical</strong> (columna 10, filas 0 a 11) separa A
            de B. Tiene una <strong>brecha de una sola celda</strong> en (6, 10).
          </p>
          <p>
            <strong>Acciones</strong> <Tex>{String.raw`\mathcal A=\{N,S,E,O,NE,NO,SE,SO\}`}</Tex>: 4 cardinales y 4 diagonales.
          </p>
          <p>
            <strong>Transición</strong> <Tex>P</Tex> con resbalón <Tex>{'p_{slip}=0.1'}</Tex>. Con probabilidad 0.9 el robot va a
            donde quiso. Con 0.1 patina y cae, con la misma probabilidad (1/30 cada una), en la dirección girada −45°, en la
            girada +45° (ambas celdas adyacentes a la deseada) o se queda en su lugar:
          </p>
          <Tex block>{String.raw`P(s'\mid s,a)=\begin{cases}0.9 & s'=s+\Delta(a)\\[2pt] 0.1/3 & s'=s+\Delta(a_{-45^\circ})\\[2pt] 0.1/3 & s'=s+\Delta(a_{+45^\circ})\\[2pt] 0.1/3 & s'=s\end{cases}`}</Tex>
          <p>
            <strong>Recompensa</strong> <Tex>R</Tex>: gol en B <Tex>{'+100'}</Tex>; choque con obstáculo o pared{' '}
            <Tex>{'-100'}</Tex> (termina el partido y el robot no avanza); paso cardinal <Tex>{'-1'}</Tex>; paso diagonal{' '}
            <Tex>{'-1.41\\approx-\\sqrt2'}</Tex>. El costo se cobra según la acción <em>comandada</em> (energía gastada), aunque
            el robot patine. Para evitar partidos infinitos se corta en 400 pasos (truncamiento, no estado terminal).
          </p>
          <Callout tone="red">
            <p className="!my-1 font-semibold text-[var(--fg)]">Regla de colisión diagonal (decisión de diseño)</p>
            <p className="!my-1">
              Un movimiento choca <strong>sólo si la celda destino</strong> es un obstáculo o queda fuera de la malla. Una
              diagonal puede pasar rozando las esquinas de dos obstáculos sin revisar las celdas ortogonales intermedias. Por
              eso la brecha se cruza en diagonal: (5, 9) → SE → (6, 10) → SE → (7, 11). Es justo lo que la vuelve peligrosa:
              en la brecha, un resbalón de ±45° o una acción exploratoria hacia N/S manda al robot contra la barrera.
            </p>
          </Callout>
        </div>
        <div className="card p-3">
          {map ? (
            <GridView
              map={map}
              paths={paths}
              highlight={data.config.gap_cells}
              annotations={[
                { r: 5.6, c: 11.3, text: '← brecha (1 celda)', color: '#ff8a8a' },
                { r: 2.2, c: 11.2, text: 'ruta corta y peligrosa', color: '#ff8a8a' },
                { r: 15.6, c: 7, text: 'ruta larga y segura', color: '#FFF6D5' },
              ]}
              title="Mapa El Paso del Contrabandista: inicio A, meta B, barrera vertical con brecha de una celda; en rojo la ruta por la brecha y en crema punteado la ruta que rodea"
            />
          ) : (
            <div className="aspect-square animate-pulse rounded-lg bg-[var(--bg)]" />
          )}
          <p className="!mt-2 px-1 text-xs">
            Las rutas dibujadas siguen la política greedy aprendida (semilla 0) sin resbalones: en rojo la de Q-learning y en
            crema punteado la de SARSA.
          </p>
        </div>
      </div>
    </Section>
  )
}

export function EquationsSection({ data }: { data: TabularData | null }) {
  const ex = useMemo(() => {
    if (!data) return null
    const Q = data.algos.qlearning.q_table_seed0
    const s = 5 * 20 + 9
    const s2 = 6 * 20 + 10
    const a = 6 // SE
    const a2 = 0 // N: acción exploratoria
    const { alpha, gamma } = data.config
    const r = -1.41
    const qsa = Q[s][a]
    const maxq = Math.max(...Q[s2])
    const argmax = Q[s2].indexOf(maxq)
    const qa2 = Q[s2][a2]
    const tQ = r + gamma * maxq
    const tS = r + gamma * qa2
    return { qsa, maxq, argmax, qa2, tQ, tS, alpha, gamma, r, newQ: qsa + alpha * (tQ - qsa), newS: qsa + alpha * (tS - qsa) }
  }, [data])
  const f = (x: number) => x.toFixed(2)
  return (
    <Section id="ecuaciones" n={3} title="Ecuaciones: SARSA vs Q-learning" kicker="El reglamento">
      <p>Ambos algoritmos son de <strong>diferencia temporal (TD)</strong>: corrigen <Tex>{'Q(s,a)'}</Tex> hacia un blanco que mezcla la recompensa inmediata y el valor estimado del siguiente estado.</p>
      <div className="grid gap-4">
        <div className="card px-4 py-2">
          <p className="!mb-0 font-semibold text-[var(--fg)]">Q-learning (off-policy), ec. (1)</p>
          <Tex block>{String.raw`Q(s,a)\leftarrow Q(s,a)+\alpha\Big[r+\gamma\,\textcolor{#FFD200}{\max_{a'}Q(s',a')}-Q(s,a)\Big]`}</Tex>
        </div>
        <div className="card px-4 py-2">
          <p className="!mb-0 font-semibold text-[var(--fg)]">SARSA (on-policy), ec. (2)</p>
          <Tex block>{String.raw`Q(s,a)\leftarrow Q(s,a)+\alpha\Big[r+\gamma\,\textcolor{#7cc4ff}{Q(s',a')}-Q(s,a)\Big]`}</Tex>
        </div>
      </div>
      <p>
        La <strong>única</strong> diferencia es el término resaltado. Q-learning usa la mejor acción posible en{' '}
        <Tex>{"s'"}</Tex>, la vaya a jugar o no. SARSA usa la acción <Tex>{"a'"}</Tex> que su política ε-greedy{' '}
        <em>realmente eligió</em>, y de ahí su nombre: <Tex>{"(s,a,r,s',a')"}</Tex>.
      </p>
      <h3 className="mt-8 font-display text-3xl tracking-wide">Ejemplo numérico: una jugada en la brecha</h3>
      {ex ? (
        <>
          <p>
            Tomamos la tabla Q final de Q-learning (semilla 0). El robot está en <Tex>{'s=(5,9)'}</Tex>, justo antes de la
            brecha, y ejecuta <Tex>{'a=SE'}</Tex>. No resbala, llega a <Tex>{"s'=(6,10)"}</Tex> (dentro de la brecha) y paga{' '}
            <Tex>{'r=-1.41'}</Tex>. Hiperparámetros: <Tex>{`\\alpha=${ex.alpha},\\ \\gamma=${ex.gamma}`}</Tex>. Valor actual{' '}
            <Tex>{`Q(s,a)=${f(ex.qsa)}`}</Tex>.
          </p>
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="card px-4 py-2">
              <p className="!mb-0 font-semibold text-[var(--accent)]">Q-learning</p>
              <Tex block>{String.raw`\max_{a'}Q(s',a') = Q(s',\text{${ACTION_NAMES[ex.argmax]}}) = ${f(ex.maxq)}`}</Tex>
              <Tex block>{String.raw`\text{blanco}= -1.41 + 0.9\times ${f(ex.maxq)} = ${f(ex.tQ)}`}</Tex>
              <Tex block>{String.raw`\delta = ${f(ex.tQ)} - ${f(ex.qsa)} = ${f(ex.tQ - ex.qsa)}`}</Tex>
              <Tex block>{String.raw`Q(s,a)\leftarrow ${f(ex.qsa)} + 0.1\times(${f(ex.tQ - ex.qsa)}) = \mathbf{${f(ex.newQ)}}`}</Tex>
            </div>
            <div className="card px-4 py-2">
              <p className="!mb-0 font-semibold text-sky-300 [html:not(.dark)_&]:text-sky-800">SARSA (con la misma tabla)</p>
              <p className="!my-1 text-sm">
                La ε-greedy sorteó una acción exploratoria: <Tex>{"a'=N"}</Tex>, que desde la brecha choca con la barrera.
              </p>
              <Tex block>{String.raw`Q(s',a'{=}N) = ${f(ex.qa2)}`}</Tex>
              <Tex block>{String.raw`\text{blanco}= -1.41 + 0.9\times(${f(ex.qa2)}) = ${f(ex.tS)}`}</Tex>
              <Tex block>{String.raw`Q(s,a)\leftarrow ${f(ex.qsa)} + 0.1\times(${f(ex.tS - ex.qsa)}) = \mathbf{${f(ex.newS)}}`}</Tex>
            </div>
          </div>
          <p>
            Con la misma experiencia, Q-learning casi no cambia su valoración de entrar a la brecha. SARSA la castiga porque
            con ε=0.1 de vez en cuando <em>de verdad</em> jugará esa acción mala. Repetida miles de veces, esta diferencia
            hace que SARSA aprenda a no pasar por ahí.
          </p>
        </>
      ) : (
        <p>Cargando datos…</p>
      )}
    </Section>
  )
}

const PSEUDO = (algo: 'sarsa' | 'qlearning') => [
  { t: `Inicializar Q(s,a) = 0 para todo s, a`, k: 'i' },
  { t: `para cada partido (episodio):`, k: 'l' },
  { t: `  s ← A`, k: 'n' },
  ...(algo === 'sarsa' ? [{ t: `  a ← ε-greedy(Q, s)`, k: 'n' }] : []),
  { t: `  repetir hasta gol, choque o 400 pasos:`, k: 'l' },
  ...(algo === 'qlearning' ? [{ t: `    a ← ε-greedy(Q, s)`, k: 'n' }] : []),
  { t: `    ejecutar a; observar r, s′ (con resbalón)`, k: 'n' },
  ...(algo === 'sarsa'
    ? [
        { t: `    a′ ← ε-greedy(Q, s′)   # ¡sí la jugará!`, k: 'diff' },
        { t: `    Q(s,a) += α[r + γ·Q(s′,a′) − Q(s,a)]`, k: 'diff' },
        { t: `    s ← s′ ;  a ← a′`, k: 'n' },
      ]
    : [
        { t: `    Q(s,a) += α[r + γ·máx Q(s′,·) − Q(s,a)]`, k: 'diff' },
        { t: `    s ← s′   # a′ se elige en la sig. vuelta`, k: 'n' },
      ]),
]

export function PseudocodeSection() {
  return (
    <Section id="pseudocodigo" n={4} title="Pseudocódigo" kicker="La pizarra del técnico">
      <p>
        El esqueleto es idéntico. Las líneas resaltadas marcan dónde difieren: SARSA elige <Tex>{"a'"}</Tex>{' '}
        <em>antes</em> de actualizar y la usa en el blanco. Q-learning usa el máximo y deja la elección para después.
      </p>
      <div className="grid gap-4 lg:grid-cols-2">
        {(['sarsa', 'qlearning'] as const).map((algo) => (
          <figure key={algo} className="card overflow-x-auto p-4">
            <figcaption className="mb-2 font-display text-2xl tracking-wide">{algo === 'sarsa' ? 'SARSA (on-policy)' : 'Q-learning (off-policy)'}</figcaption>
            <pre className="font-mono text-[13px] leading-6">
              {PSEUDO(algo).map((l, i) => (
                <div
                  key={i}
                  className={l.k === 'diff' ? '-mx-2 rounded bg-gold-500 px-2 font-semibold text-navy-900' : ''}
                >
                  <span className="mr-3 inline-block w-5 select-none text-right opacity-60">{i + 1}</span>
                  {l.t}
                </div>
              ))}
            </pre>
          </figure>
        ))}
      </div>
      <p>
        La implementación real está en <code>python/agents.py</code> (clase <code>AgenteRL</code>, configurable con{' '}
        <code>algo="sarsa" | "qlearning"</code>). Su port a TypeScript (<code>web/src/rl/</code>) mueve el juego 3D de la
        siguiente sección: <a href="#juego-contrabandista">ve a jugar</a>.
      </p>
    </Section>
  )
}

function downsample(arr: number[], step: number) {
  const x: number[] = []
  const y: number[] = []
  for (let i = step - 1; i < arr.length; i += step) {
    x.push(i + 1)
    y.push(arr[i])
  }
  return { x, y }
}

function bandTraces(name: string, color: string, r: AlgoResult, step = 10) {
  const m = downsample(r.ma100_mean, step)
  const s = downsample(r.ma100_std, step)
  const upper = m.y.map((v, i) => v + s.y[i])
  const lower = m.y.map((v, i) => v - s.y[i])
  return [
    { x: m.x, y: upper, mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false, type: 'scatter' },
    {
      x: m.x,
      y: lower,
      mode: 'lines',
      line: { width: 0 },
      fill: 'tonexty',
      fillcolor: color === GOLD ? 'rgba(255,210,0,0.22)' : 'rgba(124,196,255,0.22)',
      hoverinfo: 'skip',
      showlegend: false,
      type: 'scatter',
    },
    { x: m.x, y: m.y, mode: 'lines', name: `${name} (media ± σ, 5 semillas)`, line: { color, width: 2.4 }, type: 'scatter' },
  ]
}

export function ConvergenceSection({ data }: { data: TabularData | null }) {
  const traces = useCallback(
    () =>
      data
        ? [
            ...bandTraces('Q-learning', GOLD, data.algos.qlearning),
            ...bandTraces('SARSA', SKY, data.algos.sarsa),
          ]
        : [],
    [data],
  )
  const layout = useCallback(
    () => ({
      title: { text: 'Recompensa promedio (media móvil 100) vs partidos', font: { size: 15 } },
      xaxis: { title: { text: 'Episodio (partido)' } },
      yaxis: { title: { text: 'Recompensa por episodio' }, range: [-250, 100] },
    }),
    [],
  )
  const crashTraces = useCallback(
    () =>
      data
        ? (['qlearning', 'sarsa'] as const).map((a) => {
            const d = downsample(data.algos[a].crash_ma100, 10)
            return { x: d.x, y: d.y.map((v) => 100 * v), mode: 'lines', name: LABEL[a], line: { color: COLOR[a], width: 2 }, type: 'scatter' }
          })
        : [],
    [data],
  )
  const crashLayout = useCallback(
    () => ({
      title: { text: 'Choques (tarjetas rojas) por cada 100 partidos', font: { size: 15 } },
      xaxis: { title: { text: 'Episodio' } },
      yaxis: { title: { text: '% de episodios con choque' }, range: [0, 40] },
    }),
    [],
  )
  if (!data) return <Section id="convergencia" n={5} title="Entregable 1 · Convergencia">Cargando…</Section>
  const { qlearning: q, sarsa: s } = data.algos
  const cfg = data.config
  return (
    <Section id="convergencia" n={5} title="Entregable 1 · Convergencia" kicker="La tabla de posiciones">
      <p>
        Entrenamos ambos agentes en el mismo mapa con <Tex>{`\\alpha=${cfg.alpha},\\ \\gamma=${cfg.gamma},\\ \\varepsilon=${cfg.epsilon}`}</Tex>{' '}
        constante y <strong>{cfg.episodes.toLocaleString('es-MX')} episodios</strong>. Repetimos con {cfg.seeds.length} semillas
        ({cfg.seeds.join(', ')}). Cada semilla es la misma para SARSA y Q-learning: mismo generador del entorno y del agente.
        La curva es la media de las 5 semillas y la banda es ±1 desviación estándar.
      </p>
      <div className="card p-2">
        <Plot data={traces} layout={layout} label="Gráfica de recompensa promedio contra episodios para SARSA y Q-learning" />
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="SARSA · últimos 100" value={fmt(s.final_ma100.mean)} sub={`± ${fmt(s.final_ma100.std)} entre semillas`} />
        <Stat label="Q-learning · últimos 100" value={fmt(q.final_ma100.mean)} sub={`± ${fmt(q.final_ma100.std)} entre semillas`} />
        <Stat label="Choques SARSA (ε=0.1)" value={pct(s.eval_eps.crash_rate, 1)} sub="evaluación, 5×1000 partidos" />
        <Stat label="Choques Q-learning (ε=0.1)" value={pct(q.eval_eps.crash_rate, 1)} sub="evaluación, 5×1000 partidos" />
      </div>
      <Callout>
        <p className="!my-1 font-semibold text-[var(--fg)]">¿Cuál converge a un valor más alto durante el entrenamiento? SARSA.</p>
        <p className="!my-1">
          Con exploración activa, SARSA termina en <strong>{fmt(s.final_ma100.mean)}</strong> puntos por partido y Q-learning en{' '}
          <strong>{fmt(q.final_ma100.mean)}</strong>. La razón es la diferencia entre las ecuaciones (1) y (2). SARSA valora cada
          jugada <em>sabiendo que el 10 % de las veces tirará una acción al azar</em>, y aprende una ruta que tolera esos errores.
          Q-learning aprende el valor de la política greedy óptima e ignora que él mismo sigue explorando. Elige la ruta corta
          por la brecha, y ahí cada acción aleatoria o resbalón termina en un choque de −100 (y sin el +100 del gol). Por eso
          su curva online queda más abajo, es más ruidosa (σ = {fmt(q.final_ma100.std)} contra {fmt(s.final_ma100.std)}) y
          acumula ~{Math.round(q.eval_eps.crash_rate / Math.max(s.eval_eps.crash_rate, 1e-3))}× más choques.
        </p>
        <p className="!my-1">
          Si se apaga la exploración (ε = 0) la tabla se voltea: la política greedy de Q-learning promedia{' '}
          <strong>{fmt(q.eval_greedy.mean_return)}</strong> contra <strong>{fmt(s.eval_greedy.mean_return)}</strong> de SARSA. Q-learning
          sí aprendió el camino más corto, sólo que no lo puede jugar bien mientras explora.
        </p>
      </Callout>
      <div className="card mt-4 p-2">
        <Plot data={crashTraces} layout={crashLayout} height={300} label="Porcentaje de episodios con choque durante el entrenamiento" />
      </div>
      <p className="text-sm">
        Nota: los primeros ~1,500 partidos tienen recompensas muy negativas. Con <Tex>{'Q=0'}</Tex> inicial el agente deambula
        hasta el límite de 400 pasos (−1 por paso) porque todavía no conoce la meta. Una semilla de Q-learning (la{' '}
        {q.eval_per_seed.find((p) => p.greedy.gap_rate < 0.5)?.seed ?? '—'}) nunca descubrió la brecha con ε constante y se quedó
        con la ruta larga. Eso explica parte de su banda ancha.
      </p>
    </Section>
  )
}

type VisitKey = 'eval_eps' | 'train_last' | 'eval_greedy'
const VISIT_LABEL: Record<VisitKey, string> = {
  eval_eps: 'Evaluación con ε = 0.1 (recomendado)',
  train_last: 'Últimos 1,000 partidos de entrenamiento',
  eval_greedy: 'Evaluación greedy (ε = 0)',
}

export function HeatmapSection({ data }: { data: TabularData | null }) {
  const [src, setSrc] = useState<VisitKey>('eval_eps')
  const mk = useCallback(
    (algo: 'qlearning' | 'sarsa') => () => {
      if (!data) return []
      const V = data.algos[algo].visits[src]
      const map = data.config.map
      const z = V.map((row, i) => row.map((v, j) => (map[i][j] === '#' ? null : Math.log10(1 + v))))
      const text = V.map((row, i) => row.map((v, j) => (map[i][j] === '#' ? 'obstáculo' : `${Math.round(v).toLocaleString('es-MX')} visitas`)))
      const obs = map.map((row) => [...row].map((c) => (c === '#' ? 1 : null)))
      return [
        {
          z,
          text,
          type: 'heatmap',
          colorscale: BLUE_GOLD,
          hovertemplate: 'fila %{y}, col %{x}<br>%{text}<extra></extra>',
          colorbar: { title: { text: 'log₁₀(1+visitas)', side: 'right' }, thickness: 12 },
          xgap: 1,
          ygap: 1,
        },
        {
          z: obs,
          type: 'heatmap',
          colorscale: [
            [0, '#ff5a5a'],
            [1, '#ff5a5a'],
          ],
          showscale: false,
          hoverinfo: 'skip',
          xgap: 1,
          ygap: 1,
        },
      ]
    },
    [data, src],
  )
  const layoutFor = useCallback(
    (algo: 'qlearning' | 'sarsa') => () => ({
      title: { text: `${LABEL[algo]} · mapa de calor del jugador`, font: { size: 15 } },
      xaxis: { range: [-0.5, 19.5], dtick: 5, title: { text: 'columna' }, constrain: 'domain' },
      yaxis: { autorange: 'reversed', dtick: 5, title: { text: 'fila' }, scaleanchor: 'x' },
      margin: { l: 44, r: 8, t: 36, b: 40 },
      annotations: data
        ? [
            { x: data.config.start[1], y: data.config.start[0], text: '<b>A</b>', showarrow: false, font: { color: '#0B1F3A', size: 14 }, bgcolor: '#FFF6D5' },
            { x: data.config.goal[1], y: data.config.goal[0], text: '<b>B</b>', showarrow: false, font: { color: '#0B1F3A', size: 14 }, bgcolor: '#FFD200' },
          ]
        : [],
    }),
    [data],
  )
  const qData = useMemo(() => mk('qlearning'), [mk])
  const sData = useMemo(() => mk('sarsa'), [mk])
  const qLayout = useMemo(() => layoutFor('qlearning'), [layoutFor])
  const sLayout = useMemo(() => layoutFor('sarsa'), [layoutFor])
  if (!data) return <Section id="heatmaps" n={6} title="Entregable 2 · Mapas de calor">Cargando…</Section>
  const { qlearning: q, sarsa: s } = data.algos
  return (
    <Section id="heatmaps" n={6} title="Entregable 2 · Mapas de calor" kicker="¿Por dónde juega cada uno?">
      <p>
        La matriz de visitas cuenta cuántas veces el robot pisó cada celda una vez entrenado (suma de las 5 semillas, escala
        logarítmica). La obtenemos en una <strong>evaluación de 1,000 partidos por semilla con ε = 0.1 y resbalón activo</strong>,
        es decir, con la misma política conductual del entrenamiento pero sin actualizar Q.
      </p>
      <Callout tone="sky">
        <p className="!my-1 font-semibold text-[var(--fg)]">¿Por qué evaluar con ε = 0.1 y no con la política greedy pura?</p>
        <p className="!my-1">
          La pregunta es cómo se comporta el robot <em>cuando hay ruido</em>. Con ε = 0 cada agente repite casi siempre la misma
          trayectoria y el riesgo de la ruta elegida casi no se ve en el mapa: no aparecen los choques ni la nube de
          desviaciones alrededor de la brecha. Con ε = 0.1 se observa la distribución real de trayectorias, incluidos los
          errores, que es justo lo que cada algoritmo valoró distinto. Puedes cambiar la fuente abajo para comparar.
        </p>
      </Callout>
      <div className="mb-3 flex flex-wrap gap-2" role="group" aria-label="Fuente de la matriz de visitas">
        {(Object.keys(VISIT_LABEL) as VisitKey[]).map((k) => (
          <button
            key={k}
            type="button"
            aria-pressed={src === k}
            onClick={() => setSrc(k)}
            className={`rounded-lg px-3 py-1.5 text-sm font-semibold ${src === k ? 'bg-gold-500 text-navy-900' : 'border border-[var(--line)]'}`}
          >
            {VISIT_LABEL[k]}
          </button>
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="card p-2">
          <Plot data={qData} layout={qLayout} height={460} label="Mapa de calor de visitas de Q-learning" />
        </div>
        <div className="card p-2">
          <Plot data={sData} layout={sLayout} height={460} label="Mapa de calor de visitas de SARSA" />
        </div>
      </div>
      <h3 className="mt-8 font-display text-3xl tracking-wide">Política aprendida (flechas) sobre el valor V(s) = máx Q(s, a)</h3>
      <div className="grid gap-4 lg:grid-cols-2">
        {(['qlearning', 'sarsa'] as const).map((a) => (
          <figure key={a} className="card p-3">
            <figcaption className="mb-2 font-semibold">{LABEL[a]} · semilla 0</figcaption>
            <GridView
              map={data.config.map}
              policy={data.algos[a].policy_seed0}
              values={data.algos[a].v_seed0}
              highlight={data.config.gap_cells}
              title={`Política greedy de ${LABEL[a]} con flechas sobre el valor de cada celda (azul bajo, amarillo alto)`}
            />
          </figure>
        ))}
      </div>
      <Callout>
        <p className="!my-1 font-semibold text-[var(--fg)]">Hipótesis verificada ✔</p>
        <p className="!my-1">
          <strong>Q-learning corta esquinas:</strong> en la evaluación con ε = 0.1, el {pct(q.eval_eps.gap_rate)} de sus partidos
          pasan por la brecha de una celda, rozando las esquinas de la barrera. Las flechas apuntan en diagonal directo al hueco.
          <br />
          <strong>SARSA deja colchón de seguridad:</strong> {pct(s.eval_eps.gap_rate)} de sus partidos usan la brecha. Rodea la
          barrera por abajo y no se pega a la punta: pasa una o dos filas por debajo de la fila 11 y se mantiene lejos de las
          paredes exteriores. En la brecha su <Tex>{'V(s)'}</Tex> es mucho más bajo, porque ahí la exploración se paga con
          tarjeta roja.
        </p>
      </Callout>
      <div className="overflow-x-auto">
        <table className="mt-2 w-full min-w-[560px] text-left text-sm">
          <caption className="mb-2 text-left text-xs opacity-80">Evaluación sin aprendizaje: 1,000 partidos por semilla, promedio de 5 semillas</caption>
          <thead>
            <tr className="border-b border-[var(--line)]">
              <th scope="col" className="py-2">Métrica</th>
              <th scope="col">Q-learning ε=0.1</th>
              <th scope="col">SARSA ε=0.1</th>
              <th scope="col">Q-learning ε=0</th>
              <th scope="col">SARSA ε=0</th>
            </tr>
          </thead>
          <tbody>
            {(
              [
                ['Recompensa media', (e) => fmt(e.mean_return)],
                ['Goles', (e) => pct(e.success_rate, 1)],
                ['Choques', (e) => pct(e.crash_rate, 1)],
                ['Usa la brecha', (e) => pct(e.gap_rate, 0)],
                ['Pasos por partido', (e) => fmt(e.mean_steps)],
              ] as [string, (e: AlgoResult['eval_eps']) => string][]
            ).map(([k, fn]) => (
              <tr key={k} className="border-b border-[var(--line)]/50">
                <th scope="row" className="py-1.5 font-medium">{k}</th>
                <td>{fn(q.eval_eps)}</td>
                <td>{fn(s.eval_eps)}</td>
                <td>{fn(q.eval_greedy)}</td>
                <td>{fn(s.eval_greedy)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  )
}

export function RoboticsSection() {
  return (
    <Section id="robotica" n={7} title="Entregable 3 · Robótica real" kicker="Del simulador a la cancha de verdad">
      <Callout>
        <p className="!my-1">
          En un robot físico costoso que comparte espacio con personas elegiría <strong>SARSA</strong> (o, en general, un
          método on-policy y conservador). En el mundo real el robot nunca deja de tener ruido: patina, los sensores fallan y
          mientras aprende o se adapta sigue explorando. Q-learning optimiza el valor de una política ideal que supone que
          nunca se equivoca. El resultado es una ruta pegada a las esquinas donde un solo error cuesta un choque, que aquí es
          −100 y allá es un robot roto o una persona lastimada. SARSA incorpora en <Tex>{"Q(s,a')"}</Tex> la probabilidad real de
          equivocarse y por eso es <strong>aversa al riesgo (risk averse)</strong>. Sacrifica un poco de recompensa esperada
          (ruta más larga) a cambio de reducir mucho la varianza y la probabilidad de un desenlace catastrófico. En nuestros
          datos, ~9× menos choques con ε = 0.1. En robótica de servicio el costo de un fallo es asimétrico y no se puede
          deshacer, así que ese colchón de seguridad vale más que unos pasos ahorrados. Si se usa Q-learning, conviene apagar
          la exploración al desplegar y añadir restricciones de seguridad explícitas (zonas prohibidas, márgenes,
          penalizaciones de proximidad).
        </p>
      </Callout>
    </Section>
  )
}
