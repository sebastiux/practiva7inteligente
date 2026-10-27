import { OrbitControls } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { TabularData } from '../../data'
import type { Algo } from '../../rl/agent'
import { SMUGGLER_MAP, parseMap, type MapSpec } from '../../rl/gridworld'
import { cheer, isMuted, setMuted, thud, unlockAudio, whistle, blip } from '../sfx'
import { cellPos, Crowd, DangerMark, Defenders, Goal, Particles, Pitch, PolicyArrows, Stadium, StartSpot, type CrowdHandle, type DefendersHandle, type ParticlesHandle } from './GridScene'
import { Robot, type RobotHandle } from './Robot'
import { Runner, type StepEvent } from './runner'

type Mode = 'duel' | 'train' | 'play'

/** En pantallas verticales la cancha no cabe a lo ancho: alejamos y elevamos la cámara. */
const narrow = () => typeof window !== 'undefined' && window.innerWidth < 640
const TV_POS = (): [number, number, number] => (narrow() ? [0, 27, 21] : [0, 13, 16.5])
type View = 'tv' | 'top' | 'follow'

const TEAM = {
  qlearning: { name: 'Q-LEARNING', short: 'Q', color: '#ffd200' },
  sarsa: { name: 'SARSA', short: 'S', color: '#7cc4ff' },
  human: { name: 'TÚ', short: 'TÚ', color: '#ffffff' },
}

const SPEEDS = [0.5, 1, 2, 4, 8]
const BASE_DUR = { move: 0.32, stay: 0.3, crash: 0.7, goal: 0.55 }
const PAUSE = { goal: 1.5, crash: 1.6, timeout: 0.9 }

interface Slot {
  key: string
  team: keyof typeof TEAM
  runner: Runner
  robot: React.RefObject<RobotHandle | null>
  pause: number
}

interface Banner {
  id: number
  text: string
  tone: 'goal' | 'card' | 'info'
}

interface Settings {
  playing: boolean
  speed: number
  turbo: boolean
  view: View
}

/* ---------- Director: avanza la simulación y dispara efectos dentro del loop de R3F ---------- */
interface DirectorHandle {
  humanStep: (a: number) => void
}

const Director = forwardRef<
  DirectorHandle,
  {
    slots: Slot[]
    map: MapSpec
    settings: React.RefObject<Settings>
    particles: React.RefObject<ParticlesHandle | null>
    defenders: React.RefObject<DefendersHandle | null>
    crowd: React.RefObject<CrowdHandle | null>
    shake: React.RefObject<THREE.Group | null>
    controls: React.RefObject<{ target: THREE.Vector3; enabled: boolean } | null>
    onTick: () => void
    onBanner: (b: Omit<Banner, 'id'>) => void
  }
>(function Director({ slots, map, settings, particles, defenders, crowd, shake, controls, onTick, onBanner }, ref) {
  const { camera } = useThree()
  const shakeAmt = useRef(0)
  const tickAcc = useRef(0)
  const started = useRef(false)
  const pending = useRef<number | null>(null)

  const launch = useCallback(
    (slot: Slot, ev: StepEvent) => {
      const sp = settings.current!.speed
      const kind = ev.outcome === 'crash' ? 'crash' : ev.outcome === 'goal' ? 'goal' : ev.from === ev.to ? 'stay' : 'move'
      slot.robot.current?.go(ev, map.cols, BASE_DUR[kind] / sp)
      if (ev.slipped && kind !== 'crash' && sp <= 2) blip(220, 0.06)
    },
    [map.cols, settings],
  )

  useImperativeHandle(ref, () => ({
    humanStep: (a) => {
      const slot = slots[0]
      if (!slot || slot.team !== 'human') return
      // si el robot sigue animando, guardamos la jugada (búfer de una tecla)
      if (slot.pause > 0 || slot.robot.current?.busy()) {
        pending.current = a
        return
      }
      if (!slot.runner.active) {
        slot.runner.start()
        slot.robot.current?.spawn(map.start[0], map.start[1])
      }
      launch(slot, slot.runner.step(a))
      onTick()
    },
  }))

  useEffect(() => {
    slots.forEach((s) => {
      s.runner.start()
      s.robot.current?.spawn(map.start[0], map.start[1])
      s.pause = 0
    })
    started.current = true
  }, [slots, map])

  useFrame((_, dtRaw) => {
    const dt = Math.min(dtRaw, 0.05)
    const st = settings.current!
    if (st.turbo && st.playing) {
      for (const s of slots) {
        s.runner.trainFast(Math.round(8 * st.speed))
        s.robot.current?.setVisible(false)
      }
    } else {
      for (const s of slots) {
        s.robot.current?.setVisible(true)
        if (s.pause > 0) {
          s.pause -= dt * st.speed
          if (s.pause <= 0) {
            s.runner.start()
            s.robot.current?.spawn(map.start[0], map.start[1])
            if (s.team === 'human') onTick()
          }
          continue
        }
        if (s.team === 'human') {
          if (pending.current !== null && !s.robot.current?.busy()) {
            const a = pending.current
            pending.current = null
            if (!s.runner.active) {
              s.runner.start()
              s.robot.current?.spawn(map.start[0], map.start[1])
            }
            launch(s, s.runner.step(a))
          }
          continue
        }
        if (!st.playing || s.robot.current?.busy()) continue
        if (!s.runner.active) {
          s.runner.start()
          s.robot.current?.spawn(map.start[0], map.start[1])
          continue
        }
        launch(s, s.runner.step())
      }
    }
    // sacudida de cámara
    const g = shake.current
    if (g) {
      shakeAmt.current = Math.max(0, shakeAmt.current - dt * 2.2)
      const k = shakeAmt.current
      g.position.set((Math.random() - 0.5) * k * 0.5, (Math.random() - 0.5) * k * 0.3, (Math.random() - 0.5) * k * 0.5)
    }
    // cámara que sigue al primer robot
    if (st.view === 'follow' && slots[0]?.robot.current && !st.turbo) {
      const p = slots[0].robot.current.position()
      const want = new THREE.Vector3(p.x - 5, 7.5, p.z + 7)
      camera.position.lerp(want, 1 - Math.exp(-dt * 3))
      const c = controls.current
      if (c) c.target.lerp(p, 1 - Math.exp(-dt * 4))
    }
    tickAcc.current += dt
    if (tickAcc.current > 0.2) {
      tickAcc.current = 0
      onTick()
    }
  })

  // Efectos al impactar / anotar
  useEffect(() => {
    slots.forEach((slot) => {
      const team = TEAM[slot.team]
      handlers.set(slot.key, {
        impact: (ev: StepEvent, at: THREE.Vector3) => {
          const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
          if (ev.outcome === 'crash') {
            particles.current?.burst(at.clone().setY(0.45), { colors: ['#ffd200', '#ff8a3d', '#ffffff'], count: 60, speed: 3.2, up: 0.8, gravity: 9, life: 0.7 })
            if (ev.hit) defenders.current?.hit(ev.hit[0], ev.hit[1])
            slot.robot.current?.showCard()
            if (!reduce) shakeAmt.current = 1
            thud()
            whistle()
            onBanner({ text: `¡ROJA a ${team.name}!`, tone: 'card' })
          } else if (ev.outcome === 'goal') {
            const p = cellPos(map.goal[0], map.goal[1]).setY(0.6)
            particles.current?.burst(p, { colors: ['#ffd200', '#0b1f3a', '#ffffff', team.color], count: reduce ? 60 : 220, speed: 4.2, up: 1.6, gravity: 5, life: 2.2 })
            cheer()
            crowd.current?.cheer()
            onBanner({ text: `¡GOOOL de ${team.name}!`, tone: 'goal' })
          }
        },
        done: (ev: StepEvent) => {
          if (!ev.outcome) return
          slot.pause = PAUSE[ev.outcome]
          if (ev.outcome === 'timeout') {
            whistle(true)
            onBanner({ text: `${team.name}: se acabó el tiempo`, tone: 'info' })
          }
          onTick()
        },
      })
    })
  }, [slots, map, particles, defenders, crowd, onBanner, onTick])

  return null
})

/** Manejadores de efectos por robot (el Robot llama a onImpact/onDone). */
const handlers = new Map<string, { impact: (ev: StepEvent, at: THREE.Vector3) => void; done: (ev: StepEvent) => void }>()

/* ---------- Componente principal ---------- */
export default function GridGame({ tab }: { tab: TabularData | null }) {
  const [mode, setMode] = useState<Mode>('duel')
  const [started, setStarted] = useState(false)
  const [rows, setRows] = useState<string[]>(() => [...SMUGGLER_MAP])
  const [algo, setAlgo] = useState<Algo>('qlearning')
  const [epsilon, setEpsilon] = useState(0.1)
  const [pSlip, setPSlip] = useState(0.1)
  const [alpha, setAlpha] = useState(0.1)
  const [gamma, setGamma] = useState(0.9)
  const [speedIdx, setSpeedIdx] = useState(2)
  const [turbo, setTurbo] = useState(false)
  const [playing, setPlaying] = useState(true)
  const [view, setView] = useState<View>('tv')
  const [heatOn, setHeatOn] = useState(false)
  const [arrowsOn, setArrowsOn] = useState(false)
  const [tool, setTool] = useState<'wall' | 'start' | 'goal'>('wall')
  const [muted, setMutedState] = useState(isMuted())
  const [, setTick] = useState(0)
  const [banners, setBanners] = useState<Banner[]>([])
  const [epoch, setEpoch] = useState(0) // fuerza reinicio de corredores
  const [visible, setVisible] = useState(true)

  const wrap = useRef<HTMLDivElement>(null)
  const director = useRef<DirectorHandle>(null)
  const particles = useRef<ParticlesHandle>(null)
  const defenders = useRef<DefendersHandle>(null)
  const crowd = useRef<CrowdHandle>(null)
  const shake = useRef<THREE.Group>(null)
  const controls = useRef<{ target: THREE.Vector3; enabled: boolean; object: THREE.Camera } | null>(null)
  const settings = useRef<Settings>({ playing: true, speed: SPEEDS[2], turbo: false, view: 'tv' })
  settings.current = { playing: playing && started, speed: SPEEDS[speedIdx], turbo: turbo && mode === 'train', view }

  const map = useMemo(() => parseMap(mode === 'train' ? rows : SMUGGLER_MAP), [rows, mode])
  const robotA = useRef<RobotHandle>(null)
  const robotB = useRef<RobotHandle>(null)

  const slots = useMemo<Slot[]>(() => {
    const params = { alpha, gamma, epsilon }
    if (mode === 'duel') {
      if (!tab) return []
      return [
        { key: 'q', team: 'qlearning', runner: new Runner(map, { algo: 'qlearning', params, q: tab.algos.qlearning.q_table_seed0, seed: 11, pSlip }), robot: robotA, pause: 0 },
        { key: 's', team: 'sarsa', runner: new Runner(map, { algo: 'sarsa', params, q: tab.algos.sarsa.q_table_seed0, seed: 12, pSlip }), robot: robotB, pause: 0 },
      ]
    }
    if (mode === 'train') return [{ key: 't', team: algo, runner: new Runner(map, { algo, params, learn: true, seed: 3, pSlip }), robot: robotA, pause: 0 }]
    return [{ key: 'h', team: 'human', runner: new Runner(map, { human: true, seed: Date.now() % 1e6, pSlip }), robot: robotA, pause: 0 }]
    // Los hiperparámetros se aplican en vivo abajo; sólo reiniciamos al cambiar modo/mapa/algoritmo.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, map, algo, tab, epoch])

  // Hiperparámetros en vivo
  useEffect(() => {
    for (const s of slots) {
      if (s.runner.agent) s.runner.agent.params = { alpha, gamma, epsilon }
      s.runner.env.pSlip = pSlip
    }
  }, [slots, alpha, gamma, epsilon, pSlip])

  // Pausar el render cuando el juego no está en pantalla
  useEffect(() => {
    const el = wrap.current
    if (!el) return
    const io = new IntersectionObserver((e) => setVisible(e.some((x) => x.isIntersecting)), { threshold: 0.05 })
    io.observe(el)
    return () => io.disconnect()
  }, [])

  const onTick = useCallback(() => setTick((t) => t + 1), [])
  const bannerId = useRef(0)
  const onBanner = useCallback((b: Omit<Banner, 'id'>) => {
    const id = ++bannerId.current
    setBanners((bs) => [...bs.slice(-2), { ...b, id }])
    setTimeout(() => setBanners((bs) => bs.filter((x) => x.id !== id)), 1800)
  }, [])

  // Teclado para "Juega tú"
  const KEYS: Record<string, number> = {
    ArrowUp: 0, KeyW: 0, Numpad8: 0,
    ArrowDown: 1, KeyS: 1, KeyX: 1, Numpad2: 1,
    ArrowRight: 2, KeyD: 2, Numpad6: 2,
    ArrowLeft: 3, KeyA: 3, Numpad4: 3,
    KeyE: 4, Numpad9: 4,
    KeyQ: 5, Numpad7: 5,
    KeyC: 6, Numpad3: 6,
    KeyZ: 7, Numpad1: 7,
  }
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (mode !== 'play' || !started) return
    const a = KEYS[e.code]
    if (a === undefined) return
    e.preventDefault()
    director.current?.humanStep(a)
  }

  const start = () => {
    unlockAudio()
    whistle(true)
    setStarted(true)
    setPlaying(true)
    wrap.current?.focus()
  }

  const switchMode = (m: Mode) => {
    setMode(m)
    setHeatOn(m === 'train')
    setArrowsOn(false)
    setTurbo(false)
    setEpoch((e) => e + 1)
    if (started) whistle()
    wrap.current?.focus()
  }

  const onTile = (r: number, c: number) => {
    if (mode !== 'train') return
    const g = rows.map((x) => [...x])
    const ch = g[r][c]
    if (tool === 'wall') {
      if (ch === 'A' || ch === 'B') return
      g[r][c] = ch === '#' ? '.' : '#'
    } else {
      const mark = tool === 'start' ? 'A' : 'B'
      if (ch !== '.') return
      g.forEach((row) => row.forEach((x, k) => x === mark && (row[k] = '.')))
      g[r][c] = mark
    }
    setRows(g.map((x) => x.join('')))
  }

  const toggleFull = () => {
    const el = wrap.current
    if (!el) return
    if (document.fullscreenElement) void document.exitFullscreen()
    else void el.requestFullscreen?.()
  }

  const setCam = (v: View) => {
    setView(v)
    const c = controls.current
    if (!c) return
    if (v === 'tv') {
      c.object.position.set(...TV_POS())
      c.target.set(0, 0, 1.5)
    } else if (v === 'top') {
      c.object.position.set(0, narrow() ? 40 : 27, 0.01)
      c.target.set(0, 0, 0)
    }
  }

  const heat = heatOn ? (slots[0]?.runner.visits ?? null) : null
  const policyAgent = arrowsOn ? slots.find((s) => s.runner.agent)?.runner.agent : undefined
  // argmax determinista (el desempate aleatorio haría parpadear las flechas)
  const policy = policyAgent
    ? Array.from({ length: map.rows * map.cols }, (_, st) => {
        let best = 0
        for (let a = 1; a < 8; a++) if (policyAgent.q(st, a) > policyAgent.q(st, best)) best = a
        return best
      })
    : null
  const gap = mode === 'train' ? [] : ([[6, 10]] as [number, number][])

  const humanSlot = mode === 'play' ? slots[0] : null
  const btn = 'rounded-lg px-3 py-1.5 text-sm font-semibold transition-colors focus-visible:outline-3'
  const seg = (on: boolean) => `${btn} ${on ? 'bg-gold-500 text-navy-900' : 'border border-[var(--line)] hover:border-gold-500'}`

  return (
    <div
      ref={wrap}
      tabIndex={0}
      onKeyDown={onKeyDown}
      className="game-wrap relative overflow-hidden rounded-2xl border border-[var(--line)] bg-navy-950 text-cream-100 outline-none"
      aria-label="Juego 3D El Paso del Contrabandista"
    >
      {/* Pestañas de modo */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 px-3 py-2">
        <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Modo de juego">
          {(
            [
              ['duel', '⚔ Duelo SARSA vs Q'],
              ['train', '🏋 Entrena en vivo'],
              ['play', '🎮 Juega tú'],
            ] as [Mode, string][]
          ).map(([m, l]) => (
            <button key={m} type="button" role="tab" aria-selected={mode === m} onClick={() => switchMode(m)} className={seg(mode === m)}>
              {l}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap gap-1.5">
          {(
            [
              ['tv', '📺 TV'],
              ['top', '🛰 Cenital'],
              ['follow', '🎥 Sigue'],
            ] as [View, string][]
          ).map(([v, l]) => (
            <button key={v} type="button" aria-pressed={view === v} onClick={() => setCam(v)} className={seg(view === v)}>
              {l}
            </button>
          ))}
          <button type="button" onClick={() => (setMuted(!muted), setMutedState(!muted))} className={seg(false)} aria-label={muted ? 'Activar sonido' : 'Silenciar'}>
            {muted ? '🔇' : '🔊'}
          </button>
          <button type="button" onClick={toggleFull} className={seg(false)} aria-label="Pantalla completa">
            ⛶
          </button>
        </div>
      </div>

      <div className="relative h-[min(68vh,620px)] min-h-[380px] w-full">
        <Canvas
          shadows="percentage"
          dpr={[1, 1.75]}
          frameloop={visible ? 'always' : 'never'}
          camera={{ position: TV_POS(), fov: 42 }}
          gl={{ antialias: true }}
          aria-label="Cancha 3D con la barrera de defensas, la brecha y los robots"
        >
          <color attach="background" args={['#061226']} />
          <fog attach="fog" args={['#061226', 38, 70]} />
          <hemisphereLight args={['#bcd7ff', '#1f7a3d', 0.7]} />
          <directionalLight
            position={[9, 18, 7]}
            intensity={2.1}
            castShadow
            shadow-mapSize={[1024, 1024]}
            shadow-camera-left={-14}
            shadow-camera-right={14}
            shadow-camera-top={14}
            shadow-camera-bottom={-14}
          />
          <group ref={shake}>
            <Stadium />
            <Crowd ref={crowd} />
            <Pitch heat={heat} onTile={mode === 'train' ? onTile : undefined} key={`p${map.rows}`} />
            <Defenders ref={defenders} map={map} />
            <Goal r={map.goal[0]} c={map.goal[1]} />
            <StartSpot r={map.start[0]} c={map.start[1]} />
            <DangerMark cells={gap} />
            <PolicyArrows map={map} policy={policy} color={TEAM[slots[0]?.team ?? 'qlearning'].color} />
            {slots.map((s) => (
              <Robot
                key={`${s.key}-${epoch}`}
                ref={s.robot}
                color={TEAM[s.team].color}
                label={TEAM[s.team].short}
                onImpact={(ev, at) => handlers.get(s.key)?.impact(ev, at)}
                onDone={(ev) => handlers.get(s.key)?.done(ev)}
              />
            ))}
            <Particles ref={particles} />
          </group>
          <Director
            ref={director}
            slots={slots}
            map={map}
            settings={settings}
            particles={particles}
            defenders={defenders}
            crowd={crowd}
            shake={shake}
            controls={controls}
            onTick={onTick}
            onBanner={onBanner}
          />
          <OrbitControls
            ref={controls as never}
            makeDefault
            enableDamping
            maxPolarAngle={Math.PI / 2.15}
            minDistance={6}
            maxDistance={60}
            target={[0, 0, 1.5]}
            autoRotate={!started}
            autoRotateSpeed={0.6}
            enabled={view !== 'follow'}
          />
        </Canvas>

        {/* Marcador estilo TV */}
        {started && (
          <div className="pointer-events-none absolute inset-x-0 top-2 flex justify-center px-2">
            <div className="flex items-stretch overflow-hidden whitespace-nowrap rounded-lg border border-gold-500/60 bg-navy-900/90 font-display text-base tracking-wider shadow-xl sm:text-2xl">
              {slots.map((s, i) => {
                const t = TEAM[s.team]
                const st = s.runner.stats
                return (
                  <div key={s.key} className={`flex items-center gap-1.5 px-2 py-1 sm:gap-2 sm:px-3 ${i ? 'border-l border-gold-500/40' : ''}`}>
                    <span className="h-3 w-3 rounded-full" style={{ background: t.color }} aria-hidden />
                    <span className="sm:hidden">{t.short}</span>
                    <span className="hidden sm:inline">{t.name}</span>
                    <span className="text-gold-400">⚽ {st.goals}</span>
                    <span className="text-red-400">🟥 {st.crashes}</span>
                    <span className="hidden text-base opacity-80 sm:inline">μ {st.episodes ? s.runner.meanReturn().toFixed(1) : "—"}</span>
                  </div>
                )
              })}
              <div className="flex items-center whitespace-nowrap bg-gold-500 px-2 text-navy-900 sm:px-3">
                P{Math.max(...slots.map((s) => s.runner.stats.episodes), 0) + 1} · {slots[0]?.runner.steps ?? 0}′
              </div>
            </div>
          </div>
        )}

        {/* Banners */}
        <div className="pointer-events-none absolute inset-0 grid place-items-center" aria-live="polite">
          {banners.slice(-1).map((b) => (
            <div
              key={b.id}
              className={`banner-pop rounded-xl px-6 py-3 text-center font-display text-4xl tracking-wider shadow-2xl sm:text-6xl ${
                b.tone === 'goal' ? 'bg-gold-500 text-navy-900' : b.tone === 'card' ? 'bg-red-600 text-white' : 'bg-navy-800 text-cream-100'
              }`}
            >
              {b.text}
            </div>
          ))}
        </div>

        {/* Pantalla de título */}
        {!started && (
          <div className="absolute inset-0 grid place-items-center bg-gradient-to-b from-navy-950/30 via-navy-950/55 to-navy-950/85 p-4 text-center">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.3em] text-gold-400">Juego 1 · Grid World 8-DOF</p>
              <h3 className="mt-2 font-display text-5xl leading-none tracking-wide sm:text-7xl">El Paso del Contrabandista</h3>
              <p className="mx-auto mt-3 max-w-md text-sm opacity-90">
                Mira a SARSA y Q-learning jugar el mismo partido con sus tablas Q reales, entrena un agente en vivo o toma tú el
                control del robot.
              </p>
              <button type="button" onClick={start} className="start-pulse mt-6 rounded-xl bg-gold-500 px-8 py-3 font-display text-3xl tracking-widest text-navy-900 shadow-2xl">
                ▶ START
              </button>
            </div>
          </div>
        )}

        {/* D-pad para "Juega tú" */}
        {started && mode === 'play' && (
          <div className="absolute bottom-3 right-3 grid grid-cols-3 gap-1" role="group" aria-label="Controles de dirección">
            {[5, 0, 4, 3, -1, 2, 7, 1, 6].map((a, i) =>
              a < 0 ? (
                <span key={i} />
              ) : (
                <button
                  key={i}
                  type="button"
                  onClick={() => director.current?.humanStep(a)}
                  className="h-11 w-11 rounded-lg bg-navy-800/85 text-xl font-bold text-gold-400 ring-1 ring-gold-500/50 active:bg-gold-500 active:text-navy-900"
                  aria-label={['Norte', 'Sur', 'Este', 'Oeste', 'Noreste', 'Noroeste', 'Sureste', 'Suroeste'][a]}
                >
                  {['↑', '↓', '→', '←', '↗', '↖', '↘', '↙'][a]}
                </button>
              ),
            )}
          </div>
        )}
      </div>

      {/* Panel de control */}
      <div className="grid gap-3 border-t border-white/10 p-3 text-sm md:grid-cols-[auto_1fr]">
        <div className="flex flex-wrap items-center gap-2">
          {mode !== 'play' && (
            <button type="button" onClick={() => (unlockAudio(), setStarted(true), setPlaying((p) => !p))} className={`${btn} min-w-24 bg-gold-500 text-navy-900`}>
              {playing && started ? '⏸ Pausa' : '▶ Jugar'}
            </button>
          )}
          <span className="opacity-80">Velocidad</span>
          <div className="flex gap-1" role="group" aria-label="Velocidad">
            {SPEEDS.map((s, i) => (
              <button key={s} type="button" aria-pressed={speedIdx === i && !turbo} onClick={() => (setSpeedIdx(i), setTurbo(false))} className={seg(speedIdx === i && !turbo)}>
                {s}×
              </button>
            ))}
            {mode === 'train' && (
              <button type="button" aria-pressed={turbo} onClick={() => setTurbo((t) => !t)} className={seg(turbo)}>
                ⚡ Turbo
              </button>
            )}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 md:justify-end">
          {mode !== 'play' && (
            <>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={heatOn} onChange={(e) => setHeatOn(e.target.checked)} /> Mapa de calor
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={arrowsOn} onChange={(e) => setArrowsOn(e.target.checked)} /> Flechas de política
              </label>
            </>
          )}
          <Range id="g-eps" label="ε" value={epsilon} min={0} max={0.5} step={0.01} onChange={setEpsilon} hidden={mode === 'play'} />
          <Range id="g-slip" label="p_slip" value={pSlip} min={0} max={0.4} step={0.01} onChange={setPSlip} />
          {mode === 'train' && (
            <>
              <Range id="g-alpha" label="α" value={alpha} min={0.01} max={1} step={0.01} onChange={setAlpha} />
              <Range id="g-gamma" label="γ" value={gamma} min={0.5} max={0.99} step={0.01} onChange={setGamma} />
            </>
          )}
        </div>
        {mode === 'train' && (
          <div className="flex flex-wrap items-center gap-2 md:col-span-2">
            {(['qlearning', 'sarsa'] as Algo[]).map((a) => (
              <button key={a} type="button" aria-pressed={algo === a} onClick={() => (setAlgo(a), setEpoch((e) => e + 1))} className={seg(algo === a)}>
                {a === 'qlearning' ? 'Q-learning' : 'SARSA'}
              </button>
            ))}
            <button
              type="button"
              onClick={() => (slots[0]?.runner.trainFast(1000), onTick(), blip(880))}
              className={`${btn} border border-gold-500 text-gold-400`}
            >
              🏟 Pretemporada +1,000
            </button>
            <button type="button" onClick={() => setEpoch((e) => e + 1)} className={seg(false)}>
              ↺ Reiniciar Q
            </button>
            <span className="ml-2 opacity-80">Editor (clic en la cancha):</span>
            {(
              [
                ['wall', '🧍 Defensa'],
                ['start', 'A Salida'],
                ['goal', 'B Portería'],
              ] as const
            ).map(([t, l]) => (
              <button key={t} type="button" aria-pressed={tool === t} onClick={() => setTool(t)} className={seg(tool === t)}>
                {l}
              </button>
            ))}
            <button type="button" onClick={() => setRows([...SMUGGLER_MAP])} className={seg(false)}>
              Mapa original
            </button>
          </div>
        )}
        {humanSlot && (
          <div className="flex flex-wrap items-center gap-4 md:col-span-2">
            <span>
              Partido en curso: <strong className="text-gold-400">{humanSlot.runner.ret.toFixed(1)}</strong> pts en {humanSlot.runner.steps} pasos
            </span>
            <span>
              Tu promedio: <strong className="text-gold-400">{humanSlot.runner.stats.episodes ? humanSlot.runner.meanReturn().toFixed(1) : '—'}</strong> en{' '}
              {humanSlot.runner.stats.episodes} partidos
            </span>
            {tab && (
              <span className="opacity-90">
                Rivales (ε = 0.1): SARSA {tab.algos.sarsa.eval_eps.mean_return.toFixed(1)} · Q-learning {tab.algos.qlearning.eval_eps.mean_return.toFixed(1)}
              </span>
            )}
            <span className="opacity-80">Teclas: flechas/WASD, diagonales Q E Z C (o el D-pad).</span>
          </div>
        )}
        {mode === 'duel' && (
          <p className="md:col-span-2 opacity-90">
            Ambos robots usan su tabla Q final de Python (semilla 0), juegan con ε-greedy y resbalón activos, y no aprenden. La
            brecha está marcada en rojo. Sube ε y verás cómo Q-learning se llena de tarjetas rojas.
          </p>
        )}
      </div>
    </div>
  )
}

function Range(p: { id: string; label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; hidden?: boolean }) {
  if (p.hidden) return null
  return (
    <label htmlFor={p.id} className="flex items-center gap-2">
      <span className="w-12 text-right">{p.label}</span>
      <input id={p.id} type="range" min={p.min} max={p.max} step={p.step} value={p.value} onChange={(e) => p.onChange(Number(e.target.value))} className="w-24" />
      <span className="w-9 font-mono text-gold-400">{p.value.toFixed(2)}</span>
    </label>
  )
}
