import { useCallback, useMemo } from 'react'
import { Callout, Section, Stat } from '../components/Layout'
import { GOLD, Plot, SKY } from '../components/Plot'
import { Tex } from '../components/Tex'
import { BASE, fmt, pct, type DqnData } from '../data'

export interface SeedRow {
  seed: number
  solved_at: number | null
  eval_mean: number
  eval_std: number
  landing_rate: number
  mean_steps: number
  timeouts: number
  train_time_s: number
}

function NetworkDiagram() {
  const layers = [
    { n: 8, show: 8, label: 'Entrada (8)', sub: 'x, y, vx, vy, θ, ω, pata izq., pata der.' },
    { n: 128, show: 9, label: 'Oculta 1 (128, ReLU)', sub: '' },
    { n: 128, show: 9, label: 'Oculta 2 (128, ReLU)', sub: '' },
    { n: 4, show: 4, label: 'Salida (4, lineal)', sub: 'Q(s, nada) · Q(s, izq.) · Q(s, principal) · Q(s, der.)' },
  ]
  const W = 640
  const H = 300
  const xs = [70, 250, 430, 590]
  const pos = layers.map((l, i) =>
    Array.from({ length: l.show }, (_, k) => ({ x: xs[i], y: 40 + ((H - 90) * (k + 0.5)) / l.show })),
  )
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Diagrama de la red neuronal: 8 entradas, dos capas ocultas de 128 neuronas ReLU y 4 salidas lineales, una por acción">
      <title>Arquitectura 8 → 128 → 128 → 4</title>
      {pos.slice(0, -1).map((A, i) =>
        A.map((a, ka) =>
          pos[i + 1].map((b, kb) => (
            <line key={`${i}-${ka}-${kb}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#FFD200" strokeOpacity={0.13} strokeWidth={0.8} />
          )),
        ),
      )}
      {pos.map((A, i) =>
        A.map((a, k) => {
          const dots = layers[i].n > layers[i].show && k === Math.floor(layers[i].show / 2)
          return dots ? (
            <text key={`${i}-${k}`} x={a.x} y={a.y + 5} textAnchor="middle" fill="currentColor" fontSize={18}>
              ⋮
            </text>
          ) : (
            <circle key={`${i}-${k}`} cx={a.x} cy={a.y} r={9} fill={i === 0 ? '#FFF6D5' : i === 3 ? '#FFD200' : '#2a4f86'} stroke="#FFD200" strokeWidth={1.2} />
          )
        }),
      )}
      {layers.map((l, i) => (
        <text key={l.label} x={xs[i]} y={H - 30} textAnchor="middle" fontSize={12} fontWeight={700} fill="currentColor">
          {l.label}
        </text>
      ))}
      <text x={xs[0]} y={H - 12} textAnchor="middle" fontSize={9.5} fill="currentColor" opacity={0.85}>
        estado s ∈ ℝ⁸
      </text>
      <text x={xs[3]} y={H - 12} textAnchor="middle" fontSize={9.5} fill="currentColor" opacity={0.85}>
        Q(s, a; w)
      </text>
    </svg>
  )
}

export function DqnTheorySection({ dqn }: { dqn: DqnData | null }) {
  const nParams = dqn?.config.n_params ?? 18180
  return (
    <Section id="dqn-teoria" n={8} title="DQN · Teoría" kicker="Segundo tiempo: de la tabla a la red">
      <p>
        En LunarLander el estado <Tex>{String.raw`s\in\mathbb R^8`}</Tex> es continuo: posición <Tex>{'(x,y)'}</Tex>, velocidades{' '}
        <Tex>{'(v_x,v_y)'}</Tex>, ángulo y velocidad angular <Tex>{String.raw`(\theta,\dot\theta)`}</Tex> y el contacto binario de
        cada pata. Ya no cabe en una tabla, así que reemplazamos la tabla Q por una red neuronal con pesos <Tex>w</Tex>:
      </p>
      <Tex block>{String.raw`Q(s,a)\;\approx\;Q(s,a;w)\qquad\text{(ec. 3)}`}</Tex>
      <p>La red se entrena minimizando el error cuadrático contra el blanco de Bellman, la ec. (4) del PDF:</p>
      <Tex block>{String.raw`\mathcal L(w)=\mathbb E_{(s,a,r,s')\sim\mathcal D}\Big[\big(\underbrace{r+\gamma\max_{a'}Q(s',a';\color{#FFD200}{w^-}\color{inherit}{})}_{\text{blanco}}-Q(s,a;w)\big)^2\Big]`}</Tex>
      <div className="grid gap-4 md:grid-cols-2">
        <Callout>
          <p className="!my-1 font-semibold text-[var(--fg)]">Experience Replay</p>
          <p className="!my-1">
            Cada transición <Tex>{"(s,a,r,s',\\text{terminal})"}</Tex> se guarda en un buffer circular de 100,000 jugadas. Cada 4
            pasos se entrena con un mini-lote aleatorio de 64. Así se rompe la correlación temporal entre muestras
            consecutivas y cada experiencia se reutiliza muchas veces.
          </p>
        </Callout>
        <Callout tone="sky">
          <p className="!my-1 font-semibold text-[var(--fg)]">Target network (mejora estándar sobre la ec. 4)</p>
          <p className="!my-1">
            La ec. (4) del PDF usa los mismos pesos <Tex>w</Tex> en el blanco y en la predicción, así que el blanco se mueve con
            cada paso de gradiente y el entrenamiento puede oscilar o divergir. Usamos una copia lenta <Tex>{'w^-'}</Tex> (Mnih
            et al., 2015) con actualización suave{' '}
            <Tex>{String.raw`w^-\leftarrow\tau w+(1-\tau)w^-,\ \tau=10^{-3}`}</Tex>. Es una mejora estándar, no parte del
            enunciado.
          </p>
        </Callout>
      </div>
      <figure className="card mt-4 p-4 text-[var(--fg)]">
        <NetworkDiagram />
        <figcaption className="mt-2 text-sm opacity-90">
          Red feedforward 8 → 128 ReLU → 128 ReLU → 4 lineal: <strong>{nParams.toLocaleString('es-MX')} parámetros</strong>{' '}
          (8·128+128 + 128·128+128 + 128·4+4). Optimizador Adam con <Tex>{String.raw`\alpha=0.0005`}</Tex>,{' '}
          <Tex>{String.raw`\gamma=0.99`}</Tex> y ε-greedy con decaimiento <Tex>{String.raw`\varepsilon_{k+1}=\max(0.01,\,0.995\,\varepsilon_k)`}</Tex> por
          episodio. Pérdida MSE (ec. 4) con recorte de gradiente a norma 10.
        </figcaption>
      </figure>
      {dqn && (
        <p className="text-sm">
          Entorno usado: <code>{dqn.env_id}</code> (Gymnasium {dqn.gymnasium_version}, PyTorch {dqn.torch_version.split('+')[0]}).{' '}
          <strong>LunarLander-v2 ya no existe</strong> en Gymnasium 1.x. La v3 conserva el mismo espacio de estados, las mismas 4
          acciones y la misma recompensa, con correcciones de bugs. Entrenado localmente en CPU en {Math.round(dqn.train_time_s)} s.
        </p>
      )}
    </Section>
  )
}

function rolling(xs: number[], w: number) {
  const out: number[] = []
  let s = 0
  for (let i = 0; i < xs.length; i++) {
    s += xs[i]
    if (i >= w) s -= xs[i - w]
    out.push(s / Math.min(i + 1, w))
  }
  return out
}

function usePhases(dqn: DqnData | null) {
  return useMemo(() => {
    if (!dqn) return null
    const h = dqn.history
    const n = h.reward.length
    const timeouts = rolling(h.steps.map((s) => (s >= 1000 ? 1 : 0)), 50)
    const landings = rolling(h.landed.map((l) => (l ? 1 : 0)), 50)
    const b1 = timeouts.findIndex((v) => v >= 0.1) + 1 || Math.round(n * 0.3)
    const b2 = landings.findIndex((v, i) => i > b1 && v >= 0.2) + 1 || Math.round(n * 0.6)
    const b3 = h.ma100.findIndex((v, i) => i > b2 && v >= 150) + 1 || Math.round(n * 0.85)
    return {
      n,
      timeouts,
      landings,
      phases: [
        { from: 1, to: b1, name: '1 · Exploración', color: 'rgba(255,90,90,0.13)', desc: 'ε alto: acciones casi aleatorias, choca en ~100 pasos.' },
        { from: b1, to: b2, name: '2 · Aprende el hover', color: 'rgba(124,196,255,0.13)', desc: 'Descubre que flotar evita el −100: muchos episodios llegan al límite de 1000 pasos.' },
        { from: b2, to: b3, name: '3 · Aterrizajes inestables', color: 'rgba(255,210,0,0.12)', desc: 'Empieza a posarse; a veces rebota, se inclina o se queda flotando.' },
        { from: b3, to: n, name: '4 · Convergencia', color: 'rgba(120,220,140,0.14)', desc: 'Aterrizajes consistentes entre las banderas: media móvil > +200.' },
      ],
    }
  }, [dqn])
}

export function DqnCurveSection({ dqn, seeds }: { dqn: DqnData | null; seeds: SeedRow[] | null }) {
  const ph = usePhases(dqn)
  const data = useCallback(() => {
    if (!dqn) return []
    const x = dqn.history.reward.map((_, i) => i + 1)
    return [
      { x, y: dqn.history.reward, type: 'scatter', mode: 'markers', name: 'Recompensa por episodio', marker: { color: 'rgba(124,196,255,0.55)', size: 4 } },
      { x, y: dqn.history.ma100, type: 'scatter', mode: 'lines', name: 'Media móvil (100)', line: { color: GOLD, width: 3 } },
      { x, y: dqn.history.epsilon, type: 'scatter', mode: 'lines', name: 'ε', yaxis: 'y2', line: { color: '#ff8a8a', width: 1.6, dash: 'dot' } },
    ]
  }, [dqn])
  const layout = useCallback(
    (dark: boolean) => ({
      title: { text: 'Curva de aprendizaje DQN · LunarLander', font: { size: 15 } },
      xaxis: { title: { text: 'Episodio' } },
      yaxis: { title: { text: 'Recompensa acumulada' }, range: [-450, 330] },
      yaxis2: { overlaying: 'y', side: 'right', range: [0, 1.05], title: { text: 'ε' }, showgrid: false, color: '#ff8a8a' },
      margin: { l: 56, r: 48, t: 60, b: 92 },
      shapes: [
        ...(ph?.phases.map((p) => ({ type: 'rect', xref: 'x', yref: 'paper', x0: p.from, x1: p.to, y0: 0, y1: 1, fillcolor: p.color, line: { width: 0 }, layer: 'below' })) ?? []),
        { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 200, y1: 200, line: { color: dark ? '#9ef0a8' : '#1b7a2c', dash: 'dash', width: 1.5 } },
      ],
      annotations: [
        ...(ph?.phases.map((p) => ({ x: (p.from + p.to) / 2, y: 1.0, yref: 'paper', text: p.name.split(' · ')[0], showarrow: false, font: { size: 14, color: dark ? '#fff6d5' : '#0b1f3a' }, yanchor: 'bottom' })) ?? []),
        { x: 2, xanchor: 'left', y: 210, text: 'umbral +200', showarrow: false, font: { size: 11, color: dark ? '#9ef0a8' : '#1b7a2c' } },
      ],
    }),
    [ph],
  )
  if (!dqn || !ph) return <Section id="dqn-curva" n={9} title="Entregable DQN 1 · Curva de aprendizaje">Cargando…</Section>
  const e = dqn.eval
  return (
    <Section id="dqn-curva" n={9} title="Entregable DQN 1 · Curva de aprendizaje" kicker="El partido completo">
      <p>
        Entrenamos hasta 1,000 episodios con el criterio de paro del enunciado. El entrenamiento se detuvo en el{' '}
        <strong>episodio {dqn.solved_at}</strong>, cuando la media móvil de 100 episodios superó <strong>+200</strong>
        {dqn.solved_at ? ` (ε ya había bajado a ${fmt(dqn.history.epsilon[dqn.solved_at - 1], 3)})` : ''}.
      </p>
      <div className="card p-2">
        <Plot data={data} layout={layout} height={440} label="Curva de aprendizaje de DQN con recompensa por episodio, media móvil de 100 y épsilon, con las cuatro fases sombreadas" />
      </div>
      <ol className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {ph.phases.map((p) => (
          <li key={p.name} className="card px-4 py-3" style={{ boxShadow: `inset 4px 0 0 ${p.color.replace(/0\.1\d\)/, '0.9)')}` }}>
            <p className="font-display text-xl tracking-wide">{p.name}</p>
            <p className="text-xs opacity-80">episodios {p.from}–{p.to}</p>
            <p className="!my-1 text-sm">{p.desc}</p>
          </li>
        ))}
      </ol>
      <h3 className="mt-8 font-display text-3xl tracking-wide">Evaluación greedy (ε = 0, 20 episodios)</h3>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Recompensa media ± σ" value={`${fmt(e.mean)} ± ${fmt(e.std)}`} />
        <Stat label="Aterrizajes" value={pct(e.landing_rate)} sub="terminan posados (+100)" />
        <Stat label="Pasos promedio" value={fmt(e.mean_steps, 0)} sub="límite: 1000" />
        <Stat label="Parámetros de la red" value={dqn.config.n_params.toLocaleString('es-MX')} />
      </div>
      <p>
        La media greedy ({fmt(e.mean)}) queda por debajo de 200 aunque la media móvil de entrenamiento lo superó. Al parar{' '}
        <em>justo</em> en el umbral, el agente todavía conserva algo del hábito de la fase 2:{' '}
        {e.episodes.filter((x) => x.timeout).length} de 20 episodios greedy se quedan flotando hasta los 1000 pasos. Durante el
        entrenamiento el pequeño ε restante lo sacaba del hover, y con ε = 0 no hay quien lo empuje. Para comprobar que no es
        mala suerte de una semilla repetimos el entrenamiento completo con 4 semillas:
      </p>
      {seeds && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[520px] text-left text-sm">
            <caption className="mb-2 text-left text-xs opacity-80">Mismo código e hiperparámetros, distinta semilla. La semilla 0 es la corrida principal de esta página.</caption>
            <thead>
              <tr className="border-b border-[var(--line)]">
                <th scope="col" className="py-2">Semilla</th>
                <th scope="col">Paro (MA100 ≥ 200)</th>
                <th scope="col">Greedy media ± σ</th>
                <th scope="col">Aterrizajes</th>
                <th scope="col">Se queda flotando</th>
              </tr>
            </thead>
            <tbody>
              {seeds.map((r) => (
                <tr key={r.seed} className="border-b border-[var(--line)]/50">
                  <th scope="row" className="py-1.5 font-medium">{r.seed}</th>
                  <td>episodio {r.solved_at}</td>
                  <td>
                    {fmt(r.eval_mean)} ± {fmt(r.eval_std)}
                  </td>
                  <td>{pct(r.landing_rate)}</td>
                  <td>{r.timeouts}/20</td>
                </tr>
              ))}
              <tr className="font-semibold">
                <th scope="row" className="py-1.5">Promedio</th>
                <td>episodio {Math.round(seeds.reduce((a, r) => a + (r.solved_at ?? 0), 0) / seeds.length)}</td>
                <td>{fmt(seeds.reduce((a, r) => a + r.eval_mean, 0) / seeds.length)}</td>
                <td>{pct(seeds.reduce((a, r) => a + r.landing_rate, 0) / seeds.length)}</td>
                <td />
              </tr>
            </tbody>
          </table>
        </div>
      )}
      <figure className="card mt-6 p-3">
        <video
          className="mx-auto w-full max-w-[600px] rounded-lg"
          controls
          muted
          loop
          playsInline
          preload="none"
          poster={`${BASE}media/dqn_poster.jpg`}
          width={600}
          height={400}
        >
          <source src={`${BASE}media/dqn_landing.mp4`} type="video/mp4" />
          <a href={`${BASE}media/dqn_landing.gif`}>Ver GIF del aterrizaje</a>
        </video>
        <figcaption className="mt-2 text-center text-sm">
          Aterrizaje con la política greedy final{dqn.video ? ` (recompensa ${fmt(dqn.video.reward)})` : ''}. ¡Gol!{' '}
          <a href={`${BASE}media/dqn_landing.gif`} className="underline">Versión GIF</a>
        </figcaption>
      </figure>
    </Section>
  )
}

export function FuelSection({ dqn, nofuel }: { dqn: DqnData | null; nofuel: DqnData | null }) {
  const data = useCallback(
    () =>
      dqn && nofuel
        ? [
            { y: dqn.history.ma100, type: 'scatter', mode: 'lines', name: 'Con costo de combustible (original)', line: { color: GOLD, width: 2.6 } },
            { y: nofuel.history.ma100, type: 'scatter', mode: 'lines', name: 'Costo de combustible = 0', line: { color: SKY, width: 2.6 } },
          ]
        : [],
    [dqn, nofuel],
  )
  const layout = useCallback(
    () => ({
      title: { text: 'Media móvil (100) de la recompensa que optimiza cada agente', font: { size: 15 } },
      xaxis: { title: { text: 'Episodio' } },
      yaxis: { title: { text: 'Recompensa' } },
    }),
    [],
  )
  return (
    <Section id="dqn-combustible" n={10} title="Entregable DQN 2 · Penalización por combustible" kicker="Administrar el físico">
      <p>
        La recompensa de LunarLander resta <Tex>{'0.3'}</Tex> por cada cuadro con el motor principal encendido y{' '}
        <Tex>{'0.03'}</Tex> por cada cuadro con un propulsor lateral. Un aterrizaje típico enciende el motor principal cientos
        de veces, así que este término suma decenas de puntos negativos y <strong>define el estilo de juego</strong>.
      </p>
      <ul>
        <li>
          <strong>Con penalización</strong> la política óptima es <em>eficiente</em>: deja caer el módulo casi en caída libre y
          enciende el motor principal tarde y en ráfagas cortas para frenar. Corrige el ángulo con pulsos laterales (más
          baratos) y apaga todo en cuanto las patas tocan el suelo. Flotar cuesta puntos cada cuadro.
        </li>
        <li>
          <strong>Con costo cero</strong> encender motores es gratis. Lo que emerge es <em>hover prolongado</em>: el agente
          descubre que mantenerse suspendido es una forma segura de evitar el −100 del choque y no le cuesta nada. Usa el motor
          principal de más para corregir cualquier desviación, desciende muy lento y tarda en asentarse. Con <Tex>{'\\gamma=0.99'}</Tex>{' '}
          hay poco apuro por cobrar el +100 pronto. En un sistema real eso es combustible desperdiciado y menos autonomía.
        </li>
      </ul>
      {dqn && nofuel ? (
        <>
          <p>
            Lo comprobamos entrenando una variante con un <em>wrapper</em> que devuelve la penalización de combustible (mismo
            código, hiperparámetros y semilla). Ambas versiones se evaluaron con su política greedy en los mismos 20 episodios:
          </p>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[520px] text-left text-sm">
              <thead>
                <tr className="border-b border-[var(--line)]">
                  <th scope="col" className="py-2">Greedy, 20 episodios</th>
                  <th scope="col">Con costo (original)</th>
                  <th scope="col">Costo = 0</th>
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ['Encendidos del motor principal por episodio', (d: DqnData) => fmt(d.eval.mean_main_engine, 0)],
                    ['Encendidos de propulsores laterales', (d: DqnData) => fmt(d.eval.mean_side_engine, 0)],
                    ['Duración media (pasos)', (d: DqnData) => fmt(d.eval.mean_steps, 0)],
                    ['Episodios que agotan los 1000 pasos', (d: DqnData) => `${d.eval.episodes.filter((x) => x.timeout).length}/20`],
                    ['Aterrizajes', (d: DqnData) => pct(d.eval.landing_rate)],
                    ['Recompensa que optimiza', (d: DqnData) => `${fmt(d.eval.mean)} ± ${fmt(d.eval.std)}`],
                    ['Recompensa original (con combustible)', (d: DqnData) => `${fmt(d.eval.mean_with_fuel)} ± ${fmt(d.eval.std_with_fuel)}`],
                    ['Episodio en que MA100 ≥ 200', (d: DqnData) => String(d.solved_at ?? '—')],
                  ] as [string, (d: DqnData) => string][]
                ).map(([k, f]) => (
                  <tr key={k} className="border-b border-[var(--line)]/50">
                    <th scope="row" className="py-1.5 font-medium">{k}</th>
                    <td>{f(dqn)}</td>
                    <td>{f(nofuel)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Callout>
            <p className="!my-1">
              Sin costo, el agente enciende el motor principal{' '}
              <strong>{fmt(nofuel.eval.mean_main_engine / Math.max(1, dqn.eval.mean_main_engine), 1)}×</strong> más veces por
              episodio. Su puntaje "inflado" ({fmt(nofuel.eval.mean)}) se desploma a {fmt(nofuel.eval.mean_with_fuel)} cuando se
              le cobra la gasolina. Además alcanza el umbral antes (episodio {nofuel.solved_at} contra {dqn.solved_at}): el
              problema es más fácil porque explorar con los motores no se castiga, pero la política resultante es derrochadora:
              episodios {fmt((nofuel.eval.mean_steps / dqn.eval.mean_steps - 1) * 100, 0)} % más largos y el doble de empuje
              principal. Matiz honesto: en esta corrida el agente <em>con</em> costo se queda flotando más veces hasta el límite ({dqn.eval.episodes.filter((x) => x.timeout).length}/20 contra {nofuel.eval.episodes.filter((x) => x.timeout).length}/20)
              porque paró justo en el umbral (ver sección 9). La diferencia de estilo se ve en <em>cuánto</em> empuja, no en si
              termina el episodio.
            </p>
          </Callout>
          <div className="card p-2">
            <Plot data={data} layout={layout} height={340} label="Comparación de la media móvil de recompensa con y sin costo de combustible" />
          </div>
          <figure className="card mt-4 p-3">
            <video className="mx-auto w-full max-w-[600px] rounded-lg" controls muted loop playsInline preload="none" poster={`${BASE}media/dqn_nofuel_poster.jpg`} width={600} height={400}>
              <source src={`${BASE}media/dqn_nofuel_landing.mp4`} type="video/mp4" />
            </video>
            <figcaption className="mt-2 text-center text-sm">
              Variante sin costo de combustible: compara el uso del motor principal con el video de la sección anterior.
            </figcaption>
          </figure>
        </>
      ) : (
        <p>Cargando…</p>
      )}
    </Section>
  )
}

export function ScalabilitySection({ dqn }: { dqn: DqnData | null }) {
  const params = dqn?.config.n_params ?? 18180
  const netBytes = params * 4
  // bytes = estados x 4 acciones x 4 B (float32)
  const full = 20 ** 8 * 4 * 4
  const legs = 20 ** 6 * 2 ** 2 * 4 * 4
  const data = useCallback(
    () => [
      {
        type: 'bar',
        orientation: 'h',
        y: ['Tabla Q (8 variables × 20)', 'Tabla Q (6 × 20, patas binarias)', `Red DQN (${params.toLocaleString('es-MX')} parámetros)`],
        x: [full, legs, netBytes],
        text: ['≈ 410 GB', '≈ 4.1 GB', `≈ ${fmt(netBytes / 1000, 1)} KB`],
        textposition: 'outside',
        marker: { color: ['#ff8a8a', GOLD, SKY] },
        hovertemplate: '%{y}: %{x:.3s} bytes<extra></extra>',
      },
    ],
    [full, legs, netBytes, params],
  )
  const layout = useCallback(
    () => ({
      title: { text: 'Memoria en float32 (escala logarítmica)', font: { size: 15 } },
      xaxis: { type: 'log', title: { text: 'bytes' }, range: [4, 12.6] },
      yaxis: { automargin: true },
      margin: { l: 220, r: 30, t: 40, b: 48 },
      showlegend: false,
    }),
    [],
  )
  return (
    <Section id="escalabilidad" n={11} title="Entregable DQN 3 · Escalabilidad" kicker="¿Y si lo hacemos con tabla?">
      <p>Discretizamos cada variable continua del módulo en 20 segmentos.</p>
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="card px-4 py-2">
          <p className="!mb-0 font-semibold text-[var(--fg)]">Discretizando las 8 variables</p>
          <Tex block>{String.raw`|\mathcal S| = 20^8 = 2.56\times10^{10}\ \text{estados}`}</Tex>
          <Tex block>{String.raw`|\mathcal S|\cdot|\mathcal A| = 2.56\times10^{10}\times4 = 1.024\times10^{11}\ \text{entradas}`}</Tex>
          <Tex block>{String.raw`1.024\times10^{11}\times 4\,\text{B} = 4.10\times10^{11}\,\text{B}\approx 410\ \text{GB}`}</Tex>
        </div>
        <div className="card px-4 py-2">
          <p className="!mb-0 font-semibold text-[var(--fg)]">Patas como variables binarias</p>
          <Tex block>{String.raw`|\mathcal S| = 20^6\times2^2 = 2.56\times10^{8}\ \text{estados}`}</Tex>
          <Tex block>{String.raw`2.56\times10^{8}\times4 = 1.024\times10^{9}\ \text{entradas}`}</Tex>
          <Tex block>{String.raw`1.024\times10^{9}\times 4\,\text{B} = 4.10\times10^{9}\,\text{B}\approx 4.1\ \text{GB}`}</Tex>
        </div>
      </div>
      <div className="card mt-4 px-4 py-2">
        <p className="!mb-0 font-semibold text-[var(--fg)]">Red neuronal</p>
        <Tex block>{String.raw`(8\cdot128+128)+(128\cdot128+128)+(128\cdot4+4) = ${params.toLocaleString('en-US').replace(',', '{,}')}\ \text{parámetros}\;\Rightarrow\; ${params}\times4\,\text{B}\approx ${fmt(netBytes / 1000, 1)}\ \text{KB}`}</Tex>
      </div>
      <div className="card mt-4 p-2">
        <Plot data={data} layout={layout} height={280} label="Comparación de memoria entre tablas Q discretizadas y la red neuronal" />
      </div>
      <Callout>
        <p className="!my-1">
          Aun en el caso optimista (patas binarias), la tabla ocupa ~4.1 GB, unas{' '}
          <strong>{Math.round(legs / netBytes).toLocaleString('es-MX')} veces</strong> más que la red. Discretizando todo son ~410 GB,{' '}
          <strong>{(full / netBytes / 1e6).toFixed(1)} millones de veces</strong> más. La memoria es sólo la mitad del problema:
          una tabla necesita <em>visitar</em> cada celda muchas veces para estimarla, y con 10⁸–10¹⁰ estados casi todas quedarían
          sin datos. Con 20 segmentos la física además se ve "pixelada" (el ángulo cambia en saltos de ~18°). La red{' '}
          <strong>generaliza</strong>: estados cercanos comparten pesos, así que lo aprendido en uno se transfiere a sus
          vecinos. Por eso basta con unos cientos de episodios en vez de miles de millones de visitas.
        </p>
      </Callout>
    </Section>
  )
}

export function ConclusionsSection() {
  const files = [
    ['tabular.json', 'SARSA y Q-learning: curvas, visitas, tabla Q y política'],
    ['dqn.json', 'DQN: recompensa, media móvil, ε por episodio y evaluación greedy'],
    ['dqn_nofuel.json', 'Variante DQN sin costo de combustible'],
    ['dqn_seeds.json', 'Resumen DQN con 4 semillas'],
    ['lunar_validation.json', 'Port de LunarLander: Python vs navegador'],
  ]
  return (
    <Section id="conclusiones" n={12} title="Conclusiones" kicker="El silbatazo final">
      <ul>
        <li>
          <strong>On-policy vs off-policy no es un detalle técnico, es una actitud ante el riesgo.</strong> Con exploración
          activa, SARSA aprende una ruta larga con colchón de seguridad y gana más puntos durante el entrenamiento. Q-learning
          aprende la ruta corta por la brecha, la óptima si no hubiera errores, y paga con muchas más tarjetas rojas.
        </li>
        <li>
          El ruido (resbalón y ε) es lo que separa a los dos algoritmos. Por eso el mapa de calor se mide con ε = 0.1: con la
          política greedy pura la diferencia de riesgo casi no se nota.
        </li>
        <li>Para un robot caro entre personas, la aversión al riesgo de SARSA es la elección sensata.</li>
        <li>
          En espacios continuos la tabla Q es inviable (~4 a 410 GB). Una red de ~18 mil parámetros resuelve LunarLander en
          ~500 episodios de CPU gracias a la generalización, el experience replay y la target network.
        </li>
        <li>
          La función de recompensa moldea el estilo de juego: sin costo de combustible el agente se vuelve derrochador y
          tiende al hover.
        </li>
      </ul>
      <h3 className="mt-6 font-display text-3xl tracking-wide">Descarga los datos</h3>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        {files.map(([f, d]) => (
          <a key={f} href={`${BASE}data/${f}`} download className="card flex items-center gap-3 px-4 py-3 no-underline hover:border-gold-500">
            <span aria-hidden className="grid h-10 w-10 place-items-center rounded-lg bg-gold-500 font-bold text-navy-900">
              ⬇
            </span>
            <span>
              <span className="block font-mono text-sm text-[var(--fg)]">{f}</span>
              <span className="block text-xs opacity-80">{d}</span>
            </span>
          </a>
        ))}
      </div>
    </Section>
  )
}
