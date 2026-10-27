import { OrbitControls, Stars } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { BASE } from '../../data'
import { blip, chime, cheer, engineLoop, explosion, isMuted, setMuted, unlockAudio } from '../sfx'
import { Earth, Fx, Helipad, Lander, makeHeight, OX, Terrain, type FxHandle, type LanderHandle } from './LunarScene'
import { FPS, LunarLanderSim, heuristic, type Action, type StepResult } from './physics'
import { QPolicy, type PolicyJson } from './policy'

type Pilot = 'dqn' | 'dqn_nofuel' | 'human' | 'heuristic'
type Phase = 'title' | 'countdown' | 'flying' | 'result'

const PILOTS: { id: Pilot; label: string }[] = [
  { id: 'dqn', label: '🤖 Autopiloto DQN' },
  { id: 'dqn_nofuel', label: '⛽ DQN sin costo' },
  { id: 'human', label: '🎮 Pilota tú' },
  { id: 'heuristic', label: '📐 Heurística' },
]
const ACTION_LABEL = ['Nada', 'Prop. izq.', 'Principal', 'Prop. der.']
const ACTION_ICON = ['·', '◀', '▲', '▶']
const SPEEDS = [0.5, 1, 2, 4]

interface Hud {
  t: number
  score: number
  fuel: number
  alt: number
  vx: number
  vy: number
  angle: number
  legs: [boolean, boolean]
  action: Action
  q: number[] | null
}

interface Result {
  outcome: 'landed' | 'crashed' | 'timeout'
  score: number
  fuel: number
  time: number
}

const policyCache = new Map<string, Promise<QPolicy>>()
function loadPolicy(tag: string) {
  if (!policyCache.has(tag))
    policyCache.set(
      tag,
      fetch(`${BASE}models/${tag}_policy.json`)
        .then((r) => r.json())
        .then((j: PolicyJson) => new QPolicy(j)),
    )
  return policyCache.get(tag)!
}

interface DirectorProps {
  sim: LunarLanderSim
  phase: Phase
  pilot: Pilot
  policy: QPolicy | null
  speed: number
  keys: React.RefObject<Set<string>>
  lander: React.RefObject<LanderHandle | null>
  fx: React.RefObject<FxHandle | null>
  freeCam: boolean
  controls: React.RefObject<{ target: THREE.Vector3 } | null>
  ground: (x: number) => number
  onHud: (h: Hud) => void
  onEnd: (r: StepResult) => void
}

function Director({ sim, phase, pilot, policy, speed, keys, lander, fx, freeCam, controls, ground, onHud, onEnd }: DirectorProps) {
  const { camera, size } = useThree()
  const acc = useRef(0)
  const last = useRef<{ action: Action; q: number[] | null }>({ action: 0, q: null })
  const hudAcc = useRef(0)
  const shake = useRef(0)
  const engine = useRef<ReturnType<typeof engineLoop>>(null)
  const ended = useRef(false)
  const c = useMemo(() => new THREE.Color(), [])

  useEffect(() => {
    engine.current = engineLoop()
    return () => engine.current?.stop()
  }, [])
  useEffect(() => {
    ended.current = false
    acc.current = 0
    lander.current?.setVisible(true)
    lander.current?.sync(sim)
    last.current = { action: 0, q: null }
  }, [sim, lander])

  const choose = (obs: number[]): Action => {
    const q = policy ? policy.qValues(obs) : null
    let a: Action = 0
    if (pilot === 'human') {
      const k = keys.current!
      if (k.has('main')) a = 2
      else if (k.has('left')) a = 1
      else if (k.has('right')) a = 3
    } else if (pilot === 'heuristic') a = heuristic(obs)
    else if (q) a = q.indexOf(Math.max(...q)) as Action
    last.current = { action: a, q }
    return a
  }

  useFrame((_, dtRaw) => {
    const dt = Math.min(dtRaw, 0.05)
    let firedMain = false
    let firedSide = false
    if (phase === 'flying' && !sim.done) {
      acc.current += dt * speed
      let n = 0
      while (acc.current >= 1 / FPS && n < 12 && !sim.done) {
        acc.current -= 1 / FPS
        n++
        const r = sim.step(choose(sim.lastObs))
        const v = sim.lander.getLinearVelocity()
        if (r.main) {
          firedMain = true
          lander.current?.flame('main', r.main.x, r.main.y, r.main.dx, r.main.dy)
          for (let k = 0; k < 4; k++) {
            c.setHSL(0.08 + Math.random() * 0.06, 1, 0.55)
            const sp = 4 + Math.random() * 3
            fx.current?.emit(r.main.x + r.main.dx * 0.3, r.main.y + r.main.dy * 0.3, (Math.random() - 0.5) * 0.3, v.x + r.main.dx * sp + (Math.random() - 0.5), v.y + r.main.dy * sp + (Math.random() - 0.5), (Math.random() - 0.5) * 1.2, c, 0.5 + Math.random() * 0.3)
          }
          const alt = sim.lander.getPosition().y - ground(sim.lander.getPosition().x)
          if (alt < 3.5) {
            const gx = sim.lander.getPosition().x
            for (let k = 0; k < 3; k++) {
              c.setRGB(0.55, 0.55, 0.6)
              const dir = Math.random() < 0.5 ? -1 : 1
              fx.current?.emit(gx + (Math.random() - 0.5) * 0.4, ground(gx) + 0.05, (Math.random() - 0.5) * 1.4, dir * (2 + Math.random() * 3) * (1 - alt / 3.5), 0.4 + Math.random() * 0.8, (Math.random() - 0.5) * 2, c, 0.9)
            }
          }
        }
        if (r.side) {
          firedSide = true
          lander.current?.flame('side', r.side.x, r.side.y, r.side.dx, r.side.dy)
          c.set('#7cf5ff')
          fx.current?.emit(r.side.x, r.side.y, 0, v.x + r.side.dx * 3, v.y + r.side.dy * 3, (Math.random() - 0.5) * 0.6, c, 0.3)
        }
        if (sim.done && !ended.current) {
          ended.current = true
          const p = sim.lander.getPosition()
          if (r.crashed) {
            fx.current?.burst(p.x, p.y, ['#ffd200', '#ff8a3d', '#ff4d4d', '#ffffff'], 380, 7, 1.4, 3)
            fx.current?.debris(p.x, p.y)
            lander.current?.setVisible(false)
            if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) shake.current = 1.3
            explosion()
          } else if (r.landed) {
            fx.current?.burst(p.x, p.y + 0.6, ['#ffd200', '#ffffff', '#0b1f3a', '#7cc4ff'], 260, 4.5, 2.4, 1.6)
            chime()
            cheer()
          }
          onEnd(r)
        }
      }
    }
    if (!sim.done || !ended.current) lander.current?.sync(sim)
    engine.current?.set(phase === 'flying' ? (firedMain ? 0.55 : firedSide ? 0.18 : 0) : 0)

    // Cámara cinematográfica: sigue al módulo y se acerca al suelo
    if (!freeCam) {
      const p = sim.lander.getPosition()
      const x = p.x - OX
      const padX = 0
      const alt = Math.max(0, p.y - ground(p.x))
      const tx = x * 0.75 + padX * 0.25
      const ty = Math.max(p.y * 0.75 + 0.8, 3)
      const dist = (8.5 + Math.min(alt, 10) * 0.6) * (size.width < size.height ? 1.7 : 1)
      const want = new THREE.Vector3(tx + 1.2, ty + 0.9, dist)
      camera.position.lerp(want, 1 - Math.exp(-dt * 2.2))
      const ctl = controls.current
      if (ctl) ctl.target.lerp(new THREE.Vector3(tx, ty - 0.5, 0), 1 - Math.exp(-dt * 3))
      else camera.lookAt(tx, ty - 0.5, 0)
    }
    if (shake.current > 0) {
      shake.current = Math.max(0, shake.current - dt * 1.5)
      const k = shake.current
      camera.position.x += (Math.random() - 0.5) * k * 0.6
      camera.position.y += (Math.random() - 0.5) * k * 0.4
    }

    hudAcc.current += dt
    if (hudAcc.current > 0.1) {
      hudAcc.current = 0
      const p = sim.lander.getPosition()
      const v = sim.lander.getLinearVelocity()
      const o = sim.lastObs
      onHud({
        t: sim.t / FPS,
        score: sim.totalReward,
        fuel: sim.fuelUsed,
        alt: p.y - ground(p.x),
        vx: v.x,
        vy: v.y,
        angle: (sim.lander.getAngle() * 180) / Math.PI,
        legs: [sim.legContact[0], sim.legContact[1]],
        action: last.current.action,
        q: last.current.q ?? (policy && o.length ? policy.qValues(o) : null),
      })
    }
  })
  return null
}

export default function LunarGame() {
  const [phase, setPhase] = useState<Phase>('title')
  const [pilot, setPilot] = useState<Pilot>('dqn')
  const [policy, setPolicy] = useState<QPolicy | null>(null)
  const [speedIdx, setSpeedIdx] = useState(1)
  const [fuelCost, setFuelCost] = useState(true)
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 1e6))
  const [run, setRun] = useState(0)
  const [hud, setHud] = useState<Hud | null>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [count, setCount] = useState(3)
  const [autoReplay, setAutoReplay] = useState(true)
  const [freeCam, setFreeCam] = useState(false)
  const [muted, setMutedState] = useState(isMuted())
  const [visible, setVisible] = useState(true)
  const [record, setRecord] = useState<Record<Pilot, { tries: number; landed: number; best: number }>>({
    dqn: { tries: 0, landed: 0, best: -Infinity },
    dqn_nofuel: { tries: 0, landed: 0, best: -Infinity },
    human: { tries: 0, landed: 0, best: -Infinity },
    heuristic: { tries: 0, landed: 0, best: -Infinity },
  })

  const wrap = useRef<HTMLDivElement>(null)
  const keys = useRef(new Set<string>())
  const lander = useRef<LanderHandle>(null)
  const fx = useRef<FxHandle>(null)
  const controls = useRef<{ target: THREE.Vector3 } | null>(null)

  // La red siempre se carga (en modo manual muestra lo que haría el DQN)
  useEffect(() => {
    let alive = true
    loadPolicy(pilot === 'dqn_nofuel' ? 'dqn_nofuel' : 'dqn').then((p) => alive && setPolicy(p))
    return () => {
      alive = false
    }
  }, [pilot])

  const sim = useMemo(() => {
    const s = new LunarLanderSim({ fuelCost })
    s.reset(seed)
    return s
    // run fuerza un episodio nuevo con la misma semilla
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seed, run, fuelCost])
  const heightFn = useMemo(() => makeHeight(sim, seed), [sim, seed])
  const ground = heightFn.profile

  useEffect(() => {
    const el = wrap.current
    if (!el) return
    const io = new IntersectionObserver((e) => setVisible(e.some((x) => x.isIntersecting)), { threshold: 0.05 })
    io.observe(el)
    return () => io.disconnect()
  }, [])

  // Cuenta regresiva
  useEffect(() => {
    if (phase !== 'countdown') return
    setCount(3)
    blip(523, 0.12)
    let n = 3
    const id = setInterval(() => {
      n -= 1
      if (n > 0) {
        setCount(n)
        blip(523, 0.12)
      } else {
        clearInterval(id)
        blip(1047, 0.25)
        setPhase('flying')
      }
    }, pilot === 'human' ? 650 : 280)
    return () => clearInterval(id)
  }, [phase, sim, pilot])

  const launch = useCallback((newTerrain: boolean) => {
    setResult(null)
    if (newTerrain) setSeed(Math.floor(Math.random() * 1e6))
    else setRun((r) => r + 1)
    setPhase('countdown')
    wrap.current?.focus()
  }, [])

  const onEnd = useCallback(
    (r: StepResult) => {
      const res: Result = { outcome: r.landed ? 'landed' : r.crashed ? 'crashed' : 'timeout', score: sim.totalReward, fuel: sim.fuelUsed, time: sim.t / FPS }
      setResult(res)
      setPhase('result')
      setRecord((rec) => ({
        ...rec,
        [pilot]: { tries: rec[pilot].tries + 1, landed: rec[pilot].landed + (r.landed ? 1 : 0), best: Math.max(rec[pilot].best, res.score) },
      }))
    },
    [sim, pilot],
  )

  // Repetición automática en modos de exhibición
  useEffect(() => {
    if (phase !== 'result' || pilot === 'human' || !autoReplay) return
    const id = setTimeout(() => launch(true), 3200)
    return () => clearTimeout(id)
  }, [phase, pilot, autoReplay, launch])

  const keyName = (code: string) =>
    ({ ArrowUp: 'main', KeyW: 'main', Space: 'main', ArrowLeft: 'left', KeyA: 'left', ArrowRight: 'right', KeyD: 'right' })[code]
  const onKeyDown = (e: React.KeyboardEvent) => {
    const k = keyName(e.code)
    if (k && pilot === 'human') {
      e.preventDefault()
      keys.current.add(k)
    }
    if (e.code === 'Enter' && (phase === 'result' || phase === 'title')) start()
  }
  const onKeyUp = (e: React.KeyboardEvent) => {
    const k = keyName(e.code)
    if (k) keys.current.delete(k)
  }

  const start = () => {
    unlockAudio()
    launch(true)
  }

  const choosePilot = (p: Pilot) => {
    setPilot(p)
    setFuelCost(p !== 'dqn_nofuel')
    keys.current.clear()
    if (phase !== 'title') launch(true)
  }

  const btn = 'rounded-lg px-3 py-1.5 text-sm font-semibold transition-colors focus-visible:outline-3'
  const seg = (on: boolean) => `${btn} ${on ? 'bg-gold-500 text-navy-900' : 'border border-[var(--line)] hover:border-gold-500'}`
  const hold = (k: string) => ({
    onPointerDown: (e: React.PointerEvent) => (e.preventDefault(), keys.current.add(k)),
    onPointerUp: () => keys.current.delete(k),
    onPointerLeave: () => keys.current.delete(k),
    onPointerCancel: () => keys.current.delete(k),
  })
  const rec = record[pilot]
  const qmax = hud?.q ? Math.max(...hud.q) : 0
  const qmin = hud?.q ? Math.min(...hud.q) : 0

  return (
    <div
      ref={wrap}
      tabIndex={0}
      onKeyDown={onKeyDown}
      onKeyUp={onKeyUp}
      onBlur={() => keys.current.clear()}
      className="game-wrap relative overflow-hidden rounded-2xl border border-[var(--line)] bg-black text-cream-100 outline-none"
      aria-label="Juego 3D Alunizaje con DQN"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 bg-navy-950 px-3 py-2">
        <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Piloto">
          {PILOTS.map((p) => (
            <button key={p.id} type="button" role="tab" aria-selected={pilot === p.id} onClick={() => choosePilot(p.id)} className={seg(pilot === p.id)}>
              {p.label}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap gap-1.5">
          <button type="button" aria-pressed={freeCam} onClick={() => setFreeCam((f) => !f)} className={seg(freeCam)}>
            🎥 {freeCam ? 'Cámara libre' : 'Cámara cine'}
          </button>
          <button type="button" onClick={() => (setMuted(!muted), setMutedState(!muted))} className={seg(false)} aria-label={muted ? 'Activar sonido' : 'Silenciar'}>
            {muted ? '🔇' : '🔊'}
          </button>
          <button
            type="button"
            onClick={() => (document.fullscreenElement ? void document.exitFullscreen() : void wrap.current?.requestFullscreen?.())}
            className={seg(false)}
            aria-label="Pantalla completa"
          >
            ⛶
          </button>
        </div>
      </div>

      <div className="relative h-[min(68vh,620px)] min-h-[400px] w-full">
        <Canvas
          shadows="percentage"
          dpr={[1, 1.75]}
          frameloop={visible ? 'always' : 'never'}
          camera={{ position: [2, 9, 16], fov: 45 }}
          aria-label="Superficie lunar 3D con el módulo, la plataforma entre dos banderas y la Tierra al fondo"
        >
          <color attach="background" args={['#02040a']} />
          <ambientLight intensity={0.25} />
          <directionalLight position={[-8, 14, 10]} intensity={2.1} castShadow shadow-mapSize={[1024, 1024]} shadow-camera-left={-14} shadow-camera-right={14} shadow-camera-top={14} shadow-camera-bottom={-6} />
          <pointLight position={[0, 6, 6]} intensity={8} color="#7cc4ff" distance={30} />
          <Stars radius={90} depth={40} count={2500} factor={4} saturation={0} fade speed={0.4} />
          <Earth />
          <Terrain sim={sim} seed={seed} />
          <Helipad x1={sim.helipadX1} x2={sim.helipadX2} y={sim.helipadY} />
          <Lander ref={lander} />
          <Fx ref={fx} ground={ground} />
          <Director
            sim={sim}
            phase={phase}
            pilot={pilot}
            policy={policy}
            speed={SPEEDS[speedIdx]}
            keys={keys}
            lander={lander}
            fx={fx}
            freeCam={freeCam}
            controls={controls}
            ground={ground}
            onHud={setHud}
            onEnd={onEnd}
          />
          {freeCam && <OrbitControls ref={controls as never} makeDefault enableDamping target={[0, 4, 0]} maxDistance={40} />}
        </Canvas>

        {/* HUD de misión */}
        {hud && phase !== 'title' && (
          <div className="pointer-events-none absolute left-2 top-2 grid grid-cols-2 gap-x-4 gap-y-0.5 rounded-lg border border-gold-500/50 bg-navy-950/80 px-2 py-1.5 font-mono text-[11px] sm:px-3 sm:py-2 sm:text-sm">
            <span className="opacity-80">PUNTAJE</span>
            <span className={`text-right font-bold ${hud.score >= 0 ? 'text-gold-400' : 'text-red-400'}`}>{hud.score.toFixed(1)}</span>
            <span className="opacity-80">COMBUSTIBLE</span>
            <span className="text-right">−{hud.fuel.toFixed(1)}</span>
            <span className="opacity-80 max-sm:hidden">TIEMPO</span>
            <span className="text-right max-sm:hidden">{hud.t.toFixed(1)} s</span>
            <span className="opacity-80">ALTITUD</span>
            <span className="text-right">{Math.max(0, hud.alt).toFixed(2)}</span>
            <span className="opacity-80 max-sm:hidden">VEL. X / Y</span>
            <span className={`text-right max-sm:hidden ${Math.abs(hud.vy) > 2 ? 'text-red-400' : ''}`}>
              {hud.vx.toFixed(2)} / {hud.vy.toFixed(2)}
            </span>
            <span className="opacity-80">ÁNGULO</span>
            <span className={`text-right ${Math.abs(hud.angle) > 20 ? 'text-red-400' : ''}`}>{hud.angle.toFixed(0)}°</span>
            <span className="opacity-80">PATAS</span>
            <span className="text-right">
              <span className={hud.legs[0] ? 'text-green-400' : 'opacity-40'}>●</span> <span className={hud.legs[1] ? 'text-green-400' : 'opacity-40'}>●</span>
            </span>
          </div>
        )}

        {/* Q-values en vivo */}
        {hud?.q && phase !== 'title' && (
          <div className="pointer-events-none absolute bottom-2 left-2 w-44 rounded-lg border border-gold-500/50 bg-navy-950/80 px-2 py-1.5 text-[11px] sm:w-64 sm:px-3 sm:py-2 sm:text-xs">
            <p className="mb-1 font-semibold">
              Q(s, a) de la red {pilot === 'human' ? '· el DQN haría:' : ''}{' '}
              <span className="text-gold-400">{ACTION_LABEL[hud.q.indexOf(qmax)]}</span>
            </p>
            {hud.q.map((v, a) => {
              const w = qmax === qmin ? 100 : 15 + (85 * (v - qmin)) / (qmax - qmin)
              const chosen = a === hud.action && phase === 'flying'
              return (
                <div key={a} className="flex items-center gap-2">
                  <span className="w-4 text-center">{ACTION_ICON[a]}</span>
                  <div className="h-2.5 flex-1 rounded bg-white/10">
                    <div className={`h-full rounded ${v === qmax ? 'bg-gold-500' : 'bg-sky-400/70'} ${chosen ? 'ring-2 ring-white' : ''}`} style={{ width: `${w}%` }} />
                  </div>
                  <span className="w-12 text-right font-mono">{v.toFixed(1)}</span>
                </div>
              )
            })}
          </div>
        )}

        {/* Récord de la sesión */}
        {phase !== 'title' && (
          <div className="pointer-events-none absolute right-2 top-2 rounded-lg border border-gold-500/50 bg-navy-950/80 px-2 py-1 text-right font-display text-sm tracking-wide sm:px-3 sm:py-2 sm:text-lg">
            <div>
              ALUNIZAJES {rec.landed}/{rec.tries}
            </div>
            <div className="text-sm text-gold-400">RÉCORD {rec.tries ? rec.best.toFixed(0) : '—'}</div>
            <div className="font-sans text-[11px] opacity-80">Misión #{seed}</div>
          </div>
        )}

        {/* Cuenta regresiva */}
        {phase === 'countdown' && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center">
            <span key={count} className="banner-pop font-display text-8xl text-gold-400 drop-shadow-[0_0_20px_rgba(255,210,0,0.6)]">
              {count}
            </span>
          </div>
        )}

        {/* Resultado */}
        {phase === 'result' && result && (
          <div className="absolute inset-0 grid place-items-center p-4">
            <div className="banner-in rounded-2xl border border-gold-500/70 bg-navy-950/90 px-6 py-5 text-center shadow-2xl">
              <p
                className={`font-display text-5xl tracking-wider sm:text-6xl ${result.outcome === 'landed' ? 'text-gold-400' : result.outcome === 'crashed' ? 'text-red-400' : 'text-sky-300'}`}
              >
                {result.outcome === 'landed' ? '¡ALUNIZAJE!' : result.outcome === 'crashed' ? '¡CHOQUE!' : '¡TIEMPO!'}
              </p>
              <p className="mt-2 font-mono text-sm">
                Puntaje <strong className="text-gold-400">{result.score.toFixed(1)}</strong> · combustible −{result.fuel.toFixed(1)} · {result.time.toFixed(1)} s
              </p>
              <p className="mt-1 text-xs opacity-80">{result.score >= 200 ? 'Nivel "resuelto" (≥ 200)' : 'El umbral de LunarLander es 200 puntos'}</p>
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                <button type="button" onClick={() => launch(true)} className={`${btn} bg-gold-500 text-navy-900`}>
                  🚀 Nueva misión
                </button>
                <button type="button" onClick={() => launch(false)} className={seg(false)}>
                  ↺ Mismo terreno
                </button>
              </div>
              {pilot !== 'human' && autoReplay && <p className="mt-2 text-[11px] opacity-70">Siguiente misión automática en 3 s…</p>}
            </div>
          </div>
        )}

        {/* Título */}
        {phase === 'title' && (
          <div className="absolute inset-0 grid place-items-center bg-gradient-to-b from-black/10 via-black/40 to-black/85 p-4 text-center">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.3em] text-gold-400">Juego 2 · Deep Q-Learning</p>
              <h3 className="mt-2 font-display text-5xl leading-none tracking-wide sm:text-7xl">Alunizaje DQN</h3>
              <p className="mx-auto mt-3 max-w-md text-sm opacity-90">
                La física de LunarLander portada a tu navegador y la red neuronal entrenada en Python al mando. Mira sus Q-values en
                vivo o toma los controles (▲ motor principal, ◀ ▶ propulsores).
              </p>
              <button type="button" onClick={start} className="start-pulse mt-6 rounded-xl bg-gold-500 px-8 py-3 font-display text-3xl tracking-widest text-navy-900 shadow-2xl">
                ▶ START
              </button>
            </div>
          </div>
        )}

        {/* Controles táctiles */}
        {pilot === 'human' && phase !== 'title' && (
          <div className="absolute bottom-2 right-2 flex gap-2" role="group" aria-label="Controles del módulo">
            {(
              [
                ['left', '◀', 'Propulsor izquierdo'],
                ['main', '▲', 'Motor principal'],
                ['right', '▶', 'Propulsor derecho'],
              ] as const
            ).map(([k, l, aria]) => (
              <button key={k} type="button" aria-label={aria} {...hold(k)} className="h-14 w-14 touch-none select-none rounded-xl bg-navy-800/85 text-2xl font-bold text-gold-400 ring-1 ring-gold-500/60 active:bg-gold-500 active:text-navy-900">
                {l}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-white/10 bg-navy-950 p-3 text-sm">
        <span className="opacity-80">Velocidad</span>
        <div className="flex gap-1" role="group" aria-label="Velocidad de simulación">
          {SPEEDS.map((s, i) => (
            <button key={s} type="button" aria-pressed={speedIdx === i} onClick={() => setSpeedIdx(i)} className={seg(speedIdx === i)}>
              {s}×
            </button>
          ))}
        </div>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={fuelCost} onChange={(e) => setFuelCost(e.target.checked)} disabled={phase === 'flying'} /> Cobrar combustible
        </label>
        {pilot !== 'human' && (
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={autoReplay} onChange={(e) => setAutoReplay(e.target.checked)} /> Misiones automáticas
          </label>
        )}
        <span className="opacity-80">
          {pilot === 'human'
            ? 'Teclas: ↑/W/Espacio motor principal · ←/A y →/D propulsores · Enter: nueva misión'
            : pilot === 'heuristic'
              ? 'Controlador clásico de Gymnasium (reglas fijas, sin aprendizaje), como referencia.'
              : 'La red elige la acción con mayor Q(s, a) en cada paso, a 50 Hz como en Gymnasium.'}
        </span>
      </div>
    </div>
  )
}
