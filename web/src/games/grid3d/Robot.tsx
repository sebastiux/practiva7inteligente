import { useFrame } from '@react-three/fiber'
import { forwardRef, useImperativeHandle, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { cellPos, useBallTexture } from './GridScene'
import type { StepEvent } from './runner'

type Kind = 'move' | 'stay' | 'crash' | 'goal' | 'spawn'

interface Anim {
  kind: Kind
  from: THREE.Vector3
  to: THREE.Vector3
  t: number
  dur: number
  slipped: boolean
  heading0: number
  heading1: number
  impactFired: boolean
}

export interface RobotHandle {
  go: (ev: StepEvent, cols: number, dur: number) => void
  spawn: (r: number, c: number) => void
  busy: () => boolean
  position: () => THREE.Vector3
  showCard: () => void
  setVisible: (v: boolean) => void
}

const ease = (u: number) => (u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2)
const angDiff = (a: number, b: number) => Math.atan2(Math.sin(b - a), Math.cos(b - a))

export const Robot = forwardRef<
  RobotHandle,
  { color: string; label: string; onImpact?: (ev: StepEvent, at: THREE.Vector3) => void; onDone?: (ev: StepEvent) => void }
>(function Robot({ color, label, onImpact, onDone }, ref) {
  const group = useRef<THREE.Group>(null)
  const body = useRef<THREE.Group>(null)
  const ball = useRef<THREE.Mesh>(null)
  const wheels = useRef<THREE.Group>(null)
  const card = useRef<THREE.Mesh>(null)
  const anim = useRef<Anim | null>(null)
  const ev = useRef<StepEvent | null>(null)
  const cardT = useRef(0)
  const heading = useRef(Math.PI / 2)
  const ballTex = useBallTexture()
  const tag = useMemo(() => {
    const cv = document.createElement('canvas')
    cv.width = 128
    cv.height = 64
    const g = cv.getContext('2d')!
    g.fillStyle = color
    g.beginPath()
    g.roundRect(4, 8, 120, 48, 14)
    g.fill()
    g.fillStyle = '#0b1f3a'
    g.font = '800 34px Inter Variable, system-ui, sans-serif'
    g.textAlign = 'center'
    g.textBaseline = 'middle'
    g.fillText(label, 64, 34)
    const t = new THREE.CanvasTexture(cv)
    t.colorSpace = THREE.SRGBColorSpace
    return t
  }, [color, label])

  useImperativeHandle(ref, () => ({
    go: (e, cols, dur) => {
      const g = group.current!
      const from = cellPos(Math.floor(e.from / cols), e.from % cols)
      const to = cellPos(Math.floor(e.to / cols), e.to % cols)
      let kind: Kind = e.from === e.to ? 'stay' : 'move'
      let target = to
      if (e.outcome === 'crash' && e.hit) {
        kind = 'crash'
        target = cellPos(e.hit[0], e.hit[1])
      } else if (e.outcome === 'goal') kind = 'goal'
      const dir = target.clone().sub(from)
      const h1 = kind === 'stay' ? heading.current : Math.atan2(dir.x, dir.z)
      anim.current = { kind, from: g.position.clone(), to: kind === 'crash' ? target : to, t: 0, dur, slipped: e.slipped, heading0: heading.current, heading1: h1, impactFired: false }
      ev.current = e
    },
    spawn: (r, c) => {
      const p = cellPos(r, c)
      group.current!.position.copy(p)
      heading.current = Math.PI / 2
      anim.current = { kind: 'spawn', from: p, to: p, t: 0, dur: 0.35, slipped: false, heading0: heading.current, heading1: heading.current, impactFired: true }
      ev.current = null
      if (ball.current) ball.current.position.set(0, 0.12, 0.42)
    },
    busy: () => anim.current !== null,
    position: () => group.current!.position,
    showCard: () => {
      cardT.current = 1.6
    },
    setVisible: (v) => {
      if (group.current) group.current.visible = v
    },
  }))

  useFrame((_, dtRaw) => {
    const dt = Math.min(dtRaw, 0.05)
    const g = group.current
    const b = body.current
    if (!g || !b) return
    const a = anim.current
    let rollDist = 0
    if (a) {
      a.t += dt
      const u = Math.min(1, a.t / a.dur)
      const prev = g.position.clone()
      const h = a.heading0 + angDiff(a.heading0, a.heading1) * Math.min(1, u * 2.5)
      heading.current = h
      b.rotation.set(0, 0, 0)
      g.scale.setScalar(1)
      switch (a.kind) {
        case 'spawn':
          g.scale.setScalar(0.2 + 0.8 * ease(u))
          break
        case 'move':
        case 'goal': {
          g.position.lerpVectors(a.from, a.to, ease(u))
          if (a.slipped) {
            // patinazo: derrape lateral y giro
            b.rotation.y = Math.sin(u * Math.PI) * 0.9
            b.rotation.z = Math.sin(u * Math.PI * 3) * 0.12
          }
          if (a.kind === 'goal' && ball.current) ball.current.position.set(0, 0.12 + Math.sin(u * Math.PI) * 0.35, 0.42 + u * 1.2)
          break
        }
        case 'stay':
          b.rotation.y = Math.sin(u * Math.PI * 4) * 0.6
          b.position.y = Math.abs(Math.sin(u * Math.PI * 2)) * 0.06
          break
        case 'crash': {
          // embiste hasta la mitad y rebota
          const k = u < 0.4 ? ease(u / 0.4) * 0.48 : 0.48 * Math.exp(-(u - 0.4) * 9) * Math.cos((u - 0.4) * 22)
          g.position.lerpVectors(a.from, a.to, Math.max(0, k))
          b.rotation.x = u > 0.4 ? -Math.sin((u - 0.4) * 18) * 0.35 * (1 - u) : 0
          if (!a.impactFired && u >= 0.4) {
            a.impactFired = true
            onImpact?.(ev.current!, a.to)
          }
          break
        }
      }
      g.rotation.y = heading.current
      rollDist = g.position.distanceTo(prev)
      if (u >= 1) {
        anim.current = null
        b.rotation.set(0, 0, 0)
        b.position.y = 0
        if (a.kind === 'goal') onImpact?.(ev.current!, g.position.clone())
        if (ev.current) onDone?.(ev.current)
      }
    }
    if (ball.current && rollDist) ball.current.rotation.x += rollDist / 0.12
    wheels.current?.children.forEach((w) => (w.rotation.x += rollDist / 0.1 + (a?.kind === 'stay' ? dt * 25 : 0)))
    if (card.current) {
      cardT.current = Math.max(0, cardT.current - dt)
      const t = cardT.current
      card.current.visible = t > 0
      card.current.position.y = 1.0 + (1.6 - t) * 0.25
      card.current.rotation.y += dt * 6
      card.current.scale.setScalar(t > 1.45 ? (1.6 - t) / 0.15 : 1)
    }
  })

  return (
    <group ref={group}>
      <group ref={body}>
        {/* chasis */}
        <mesh position={[0, 0.24, 0]} castShadow>
          <boxGeometry args={[0.5, 0.26, 0.56]} />
          <meshStandardMaterial color={color} metalness={0.35} roughness={0.35} />
        </mesh>
        <mesh position={[0, 0.42, -0.04]} castShadow>
          <boxGeometry args={[0.34, 0.16, 0.3]} />
          <meshStandardMaterial color="#0b1f3a" metalness={0.4} roughness={0.4} />
        </mesh>
        {/* visor */}
        <mesh position={[0, 0.43, 0.115]}>
          <boxGeometry args={[0.26, 0.06, 0.02]} />
          <meshStandardMaterial color="#7cf5ff" emissive="#7cf5ff" emissiveIntensity={2} />
        </mesh>
        {/* antena */}
        <mesh position={[0.1, 0.6, -0.1]}>
          <cylinderGeometry args={[0.012, 0.012, 0.22, 6]} />
          <meshStandardMaterial color="#c8ccd6" />
        </mesh>
        <mesh position={[0.1, 0.72, -0.1]}>
          <sphereGeometry args={[0.035, 10, 8]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={2.5} />
        </mesh>
        <group ref={wheels}>
          {[
            [-0.27, 0.1, 0.17],
            [0.27, 0.1, 0.17],
            [-0.27, 0.1, -0.17],
            [0.27, 0.1, -0.17],
          ].map((p, i) => (
            <mesh key={i} position={p as [number, number, number]} rotation-z={Math.PI / 2} castShadow>
              <cylinderGeometry args={[0.1, 0.1, 0.07, 14]} />
              <meshStandardMaterial color="#1a1a1a" roughness={0.9} />
            </mesh>
          ))}
        </group>
        <mesh ref={ball} position={[0, 0.12, 0.42]} castShadow>
          <sphereGeometry args={[0.12, 20, 14]} />
          <meshStandardMaterial map={ballTex} roughness={0.5} />
        </mesh>
      </group>
      <mesh ref={card} visible={false}>
        <planeGeometry args={[0.24, 0.34]} />
        <meshStandardMaterial color="#e11d2e" emissive="#e11d2e" emissiveIntensity={0.8} side={THREE.DoubleSide} />
      </mesh>
      <sprite position={[0, 1.0, 0]} scale={[0.62, 0.31, 1]}>
        <spriteMaterial map={tag} depthWrite={false} />
      </sprite>
    </group>
  )
})
