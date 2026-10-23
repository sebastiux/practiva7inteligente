import { useCallback, useEffect, useRef, useState } from 'react'
import { AgenteRL, type Algo } from '../rl/agent'
import { ACTION_DELTAS, GridWorld, SMUGGLER_MAP, parseMap } from '../rl/gridworld'

type Tool = 'wall' | 'start' | 'goal'

const SPEEDS = [1, 10, 50, 200]

function heatColor(t: number) {
  // azul marino -> azul -> amarillo
  const stops = [
    [11, 31, 58],
    [31, 90, 150],
    [255, 210, 0],
  ]
  const x = Math.max(0, Math.min(1, t)) * 2
  const i = Math.min(1, Math.floor(x))
  const f = x - i
  const c = stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f))
  return `rgb(${c.join(',')})`
}

function Slider(props: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; id: string }) {
  return (
    <label htmlFor={props.id} className="flex flex-col gap-1 text-sm">
      <span className="flex justify-between">
        <span>{props.label}</span>
        <span className="font-mono text-[var(--accent)]">{props.value.toFixed(2)}</span>
      </span>
      <input
        id={props.id}
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </label>
  )
}

export function Playground() {
  const [rows, setRows] = useState<string[]>(() => [...SMUGGLER_MAP])
  const [algo, setAlgo] = useState<Algo>('qlearning')
  const [alpha, setAlpha] = useState(0.1)
  const [gamma, setGamma] = useState(0.9)
  const [epsilon, setEpsilon] = useState(0.1)
  const [pSlip, setPSlip] = useState(0.1)
  const [speed, setSpeed] = useState(50)
  const [running, setRunning] = useState(false)
  const [tool, setTool] = useState<Tool>('wall')
  const [showPolicy, setShowPolicy] = useState(true)
  const [stats, setStats] = useState({ episodes: 0, ma: 0, goals: 0, crashes: 0 })
  const [animating, setAnimating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sparkRef = useRef<HTMLCanvasElement>(null)
  const envRef = useRef<GridWorld | null>(null)
  const agentRef = useRef<AgenteRL | null>(null)
  const visitsRef = useRef<Float64Array>(new Float64Array(400))
  const historyRef = useRef<number[]>([])
  const outcomesRef = useRef<string[]>([])
  const robotRef = useRef<{ cell: number; trail: number[] } | null>(null)
  const rafRef = useRef<number>(0)

  const reset = useCallback(() => {
    try {
      const map = parseMap(rows)
      envRef.current = new GridWorld(map, { pSlip, seed: 1 })
      agentRef.current = new AgenteRL(envRef.current.nStates, algo, { alpha, gamma, epsilon }, 1)
      visitsRef.current = new Float64Array(envRef.current.nStates)
      historyRef.current = []
      outcomesRef.current = []
      robotRef.current = null
      setStats({ episodes: 0, ma: 0, goals: 0, crashes: 0 })
      setError(null)
    } catch (e) {
      setError((e as Error).message)
      setRunning(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, algo])

  useEffect(reset, [reset])

  // Hiperparámetros en vivo (sin reiniciar la tabla Q)
  useEffect(() => {
    if (agentRef.current) agentRef.current.params = { alpha, gamma, epsilon }
    if (envRef.current) envRef.current.pSlip = pSlip
  }, [alpha, gamma, epsilon, pSlip])

  const draw = useCallback(() => {
    const cv = canvasRef.current
    const env = envRef.current
    if (!cv || !env) return
    const parent = cv.parentElement!
    const size = Math.min(parent.clientWidth, 560)
    const dpr = window.devicePixelRatio || 1
    if (cv.width !== Math.round(size * dpr)) {
      cv.width = Math.round(size * dpr)
      cv.height = Math.round(size * dpr)
      cv.style.width = `${size}px`
      cv.style.height = `${size}px`
    }
    const ctx = cv.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const { rows: R, cols: W } = env.map
    const c = size / W
    const visits = visitsRef.current
    let vmax = 0
    for (let i = 0; i < visits.length; i++) vmax = Math.max(vmax, visits[i])
    const lmax = Math.log1p(vmax) || 1
    for (let i = 0; i < R; i++)
      for (let j = 0; j < W; j++) {
        const s = i * W + j
        ctx.fillStyle = env.map.obstacles[s] ? '#061226' : heatColor(Math.log1p(visits[s]) / lmax)
        ctx.fillRect(j * c, i * c, c, c)
        if (env.map.obstacles[s]) {
          ctx.strokeStyle = 'rgba(255,210,0,0.6)'
          ctx.lineWidth = 1.2
          ctx.beginPath()
          ctx.moveTo(j * c + 3, i * c + 3)
          ctx.lineTo(j * c + c - 3, i * c + c - 3)
          ctx.moveTo(j * c + c - 3, i * c + 3)
          ctx.lineTo(j * c + 3, i * c + c - 3)
          ctx.stroke()
        }
      }
    ctx.strokeStyle = 'rgba(255,255,255,0.08)'
    ctx.lineWidth = 1
    for (let k = 0; k <= W; k++) {
      ctx.beginPath()
      ctx.moveTo(k * c, 0)
      ctx.lineTo(k * c, size)
      ctx.moveTo(0, k * c)
      ctx.lineTo(size, k * c)
      ctx.stroke()
    }
    const agent = agentRef.current
    if (showPolicy && agent && stats.episodes > 0) {
      ctx.strokeStyle = 'rgba(255,246,213,0.85)'
      ctx.lineWidth = 1.4
      for (let s = 0; s < env.nStates; s++) {
        if (env.map.obstacles[s]) continue
        let best = 0
        for (let a = 1; a < 8; a++) if (agent.q(s, a) > agent.q(s, best)) best = a
        if (agent.q(s, best) === 0) continue
        const [i, j] = env.toPos(s)
        const [dr, dc] = ACTION_DELTAS[best]
        const n = Math.hypot(dr, dc)
        const cx = j * c + c / 2
        const cy = i * c + c / 2
        const L = c * 0.32
        const x2 = cx + (dc / n) * L
        const y2 = cy + (dr / n) * L
        const ang = Math.atan2(dr, dc)
        ctx.beginPath()
        ctx.moveTo(cx - (dc / n) * L * 0.6, cy - (dr / n) * L * 0.6)
        ctx.lineTo(x2, y2)
        ctx.lineTo(x2 - 4 * Math.cos(ang - 0.5), y2 - 4 * Math.sin(ang - 0.5))
        ctx.moveTo(x2, y2)
        ctx.lineTo(x2 - 4 * Math.cos(ang + 0.5), y2 - 4 * Math.sin(ang + 0.5))
        ctx.stroke()
      }
    }
    for (const [p, label, fill] of [
      [env.map.start, 'A', '#FFF6D5'],
      [env.map.goal, 'B', '#FFD200'],
    ] as const) {
      ctx.fillStyle = fill
      ctx.beginPath()
      ctx.arc(p[1] * c + c / 2, p[0] * c + c / 2, c * 0.45, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = '#0B1F3A'
      ctx.font = `800 ${Math.round(c * 0.6)}px Inter Variable, system-ui`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(label, p[1] * c + c / 2, p[0] * c + c / 2 + 1)
    }
    const robot = robotRef.current
    if (robot) {
      ctx.strokeStyle = '#ff5a5a'
      ctx.lineWidth = 3
      ctx.beginPath()
      robot.trail.forEach((s, k) => {
        const [i, j] = env.toPos(s)
        if (k === 0) ctx.moveTo(j * c + c / 2, i * c + c / 2)
        else ctx.lineTo(j * c + c / 2, i * c + c / 2)
      })
      ctx.stroke()
      const [i, j] = env.toPos(robot.cell)
      ctx.fillStyle = '#ffffff'
      ctx.strokeStyle = '#0B1F3A'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(j * c + c / 2, i * c + c / 2, c * 0.38, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
      // pentágonos del balón
      ctx.fillStyle = '#0B1F3A'
      ctx.beginPath()
      ctx.arc(j * c + c / 2, i * c + c / 2, c * 0.12, 0, Math.PI * 2)
      ctx.fill()
    }

    const sp = sparkRef.current
    if (sp) {
      const h = historyRef.current
      const w = sp.clientWidth
      const hh = 72
      sp.width = Math.round(w * dpr)
      sp.height = Math.round(hh * dpr)
      const sc = sp.getContext('2d')!
      sc.setTransform(dpr, 0, 0, dpr, 0, 0)
      sc.clearRect(0, 0, w, hh)
      if (h.length > 1) {
        const lo = -150
        const hi = 100
        const y = (v: number) => hh - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * hh
        sc.strokeStyle = 'rgba(255,246,213,0.25)'
        sc.beginPath()
        sc.moveTo(0, y(0))
        sc.lineTo(w, y(0))
        sc.stroke()
        sc.strokeStyle = '#FFD200'
        sc.lineWidth = 2
        sc.beginPath()
        const step = Math.max(1, Math.floor(h.length / w))
        let k = 0
        for (let i = 0; i < h.length; i += step, k++) {
          const lo2 = Math.max(0, i - 99)
          let sum = 0
          for (let t = lo2; t <= i; t++) sum += h[t]
          const v = sum / (i - lo2 + 1)
          const x = (i / (h.length - 1)) * w
          if (k === 0) sc.moveTo(x, y(v))
          else sc.lineTo(x, y(v))
        }
        sc.stroke()
      }
    }
  }, [showPolicy, stats.episodes])

  useEffect(() => {
    draw()
    const onResize = () => draw()
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [draw, rows])

  // Bucle de entrenamiento
  useEffect(() => {
    if (!running) return
    const loop = () => {
      const env = envRef.current
      const agent = agentRef.current
      if (!env || !agent) return
      const visits = visitsRef.current
      // olvido lento para que el mapa de calor refleje la política reciente
      for (let i = 0; i < visits.length; i++) visits[i] *= 0.995 ** speed
      for (let k = 0; k < speed; k++) {
        const res = agent.runEpisode(env, { visits })
        historyRef.current.push(res.ret)
        outcomesRef.current.push(res.outcome)
      }
      const h = historyRef.current
      const last = h.slice(-100)
      const lastO = outcomesRef.current.slice(-100)
      setStats({
        episodes: h.length,
        ma: last.reduce((a, b) => a + b, 0) / last.length,
        goals: lastO.filter((o) => o === 'goal').length / lastO.length,
        crashes: lastO.filter((o) => o === 'crash').length / lastO.length,
      })
      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(rafRef.current)
  }, [running, speed])

  useEffect(() => {
    draw()
  })

  const animate = () => {
    const env = envRef.current
    const agent = agentRef.current
    if (!env || !agent) return
    setRunning(false)
    setAnimating(true)
    const res = agent.runEpisode(env, { learn: false })
    let k = 0
    robotRef.current = { cell: res.path[0], trail: [res.path[0]] }
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const tick = () => {
      k += 1
      if (k >= res.path.length || reduce) {
        robotRef.current = { cell: res.path[res.path.length - 1], trail: res.path }
        setAnimating(false)
        draw()
        return
      }
      robotRef.current = { cell: res.path[k], trail: res.path.slice(0, k + 1) }
      draw()
      window.setTimeout(tick, 110)
    }
    window.setTimeout(tick, 110)
  }

  const onCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const cv = canvasRef.current
    if (!cv) return
    const rect = cv.getBoundingClientRect()
    const j = Math.floor(((e.clientX - rect.left) / rect.width) * 20)
    const i = Math.floor(((e.clientY - rect.top) / rect.height) * 20)
    if (i < 0 || j < 0 || i >= 20 || j >= 20) return
    const grid = rows.map((r) => [...r])
    const ch = grid[i][j]
    if (tool === 'wall') {
      if (ch === 'A' || ch === 'B') return
      grid[i][j] = ch === '#' ? '.' : '#'
    } else {
      const mark = tool === 'start' ? 'A' : 'B'
      if (ch === 'A' || ch === 'B') return
      grid.forEach((r) => r.forEach((c2, k) => c2 === mark && (r[k] = '.')))
      grid[i][j] = mark
    }
    setRunning(false)
    setRows(grid.map((r) => r.join('')))
  }

  const btn =
    'rounded-lg px-3 py-2 text-sm font-semibold transition-colors disabled:opacity-50 focus-visible:outline-3'
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,560px)_1fr]">
      <div>
        <canvas
          ref={canvasRef}
          onClick={onCanvasClick}
          className="cursor-crosshair rounded-xl border border-[var(--line)]"
          role="img"
          aria-label="Malla interactiva: haz clic para editar obstáculos; el color muestra el mapa de calor de visitas y las flechas la política greedy"
        />
        <p className="mt-2 text-xs opacity-80">
          Clic en la malla: {tool === 'wall' ? 'poner/quitar obstáculo' : tool === 'start' ? 'mover el inicio A' : 'mover la meta B'}.
          Color = mapa de calor del jugador (visitas recientes, escala log).
        </p>
      </div>
      <div className="flex flex-col gap-4">
        {error && <p className="rounded-lg bg-red-900/60 p-2 text-sm">{error}</p>}
        <div className="flex flex-wrap gap-2" role="group" aria-label="Algoritmo">
          {(['qlearning', 'sarsa'] as const).map((a) => (
            <button
              key={a}
              type="button"
              aria-pressed={algo === a}
              onClick={() => setAlgo(a)}
              className={`${btn} ${algo === a ? 'bg-gold-500 text-navy-900' : 'border border-[var(--line)]'}`}
            >
              {a === 'qlearning' ? 'Q-learning (off-policy)' : 'SARSA (on-policy)'}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap gap-2">
          <button type="button" onClick={() => setRunning((r) => !r)} disabled={animating || !!error} className={`${btn} bg-gold-500 text-navy-900 min-w-28`}>
            {running ? '⏸ Pausar' : '▶ Entrenar'}
          </button>
          <button type="button" onClick={animate} disabled={animating || !!error} className={`${btn} border border-[var(--line)]`}>
            ⚽ Ver jugada
          </button>
          <button type="button" onClick={() => (setRunning(false), reset())} className={`${btn} border border-[var(--line)]`}>
            ↺ Reiniciar Q
          </button>
          <button
            type="button"
            onClick={() => (setRunning(false), setRows([...SMUGGLER_MAP]))}
            className={`${btn} border border-[var(--line)]`}
          >
            Mapa original
          </button>
        </div>
        <fieldset className="grid grid-cols-2 gap-x-5 gap-y-3">
          <legend className="sr-only">Hiperparámetros</legend>
          <Slider id="pg-alpha" label="α (tasa de aprendizaje)" value={alpha} min={0.01} max={1} step={0.01} onChange={setAlpha} />
          <Slider id="pg-gamma" label="γ (descuento)" value={gamma} min={0.5} max={0.99} step={0.01} onChange={setGamma} />
          <Slider id="pg-eps" label="ε (exploración)" value={epsilon} min={0} max={0.5} step={0.01} onChange={setEpsilon} />
          <Slider id="pg-slip" label="p_slip (resbalón)" value={pSlip} min={0} max={0.4} step={0.01} onChange={setPSlip} />
        </fieldset>
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <label htmlFor="pg-speed">Partidos por cuadro</label>
          <select id="pg-speed" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="rounded-md border border-[var(--line)] bg-[var(--bg)] px-2 py-1">
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={showPolicy} onChange={(e) => setShowPolicy(e.target.checked)} /> Flechas de política
          </label>
        </div>
        <div className="flex flex-wrap gap-2" role="group" aria-label="Herramienta del editor">
          {(
            [
              ['wall', '🧱 Obstáculo'],
              ['start', 'A Inicio'],
              ['goal', 'B Meta'],
            ] as const
          ).map(([t, l]) => (
            <button
              key={t}
              type="button"
              aria-pressed={tool === t}
              onClick={() => setTool(t)}
              className={`${btn} ${tool === t ? 'bg-navy-600 text-white' : 'border border-[var(--line)]'}`}
            >
              {l}
            </button>
          ))}
        </div>
        <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
          {[
            ['Partidos', stats.episodes.toLocaleString('es-MX')],
            ['Media últimos 100', stats.episodes ? stats.ma.toFixed(1) : '—'],
            ['Goles (últ. 100)', stats.episodes ? `${Math.round(stats.goals * 100)} %` : '—'],
            ['Choques (últ. 100)', stats.episodes ? `${Math.round(stats.crashes * 100)} %` : '—'],
          ].map(([k, v]) => (
            <div key={k} className="card px-3 py-2">
              <dt className="text-xs opacity-80">{k}</dt>
              <dd className="font-display text-2xl text-[var(--accent)]">{v}</dd>
            </div>
          ))}
        </dl>
        <div>
          <p className="mb-1 text-xs opacity-80">Media móvil (100) de la recompensa por partido</p>
          <canvas ref={sparkRef} className="h-[72px] w-full rounded-md border border-[var(--line)]" aria-hidden />
        </div>
      </div>
    </div>
  )
}
