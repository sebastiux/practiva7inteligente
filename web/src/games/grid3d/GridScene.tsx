import { useFrame, type ThreeEvent } from '@react-three/fiber'
import { forwardRef, useImperativeHandle, useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { ACTION_DELTAS, type MapSpec } from '../../rl/gridworld'

export const N = 20
/** Centro de la celda (r, c) en coordenadas del mundo (x = columna, z = fila). */
export const cellPos = (r: number, c: number) => new THREE.Vector3(c - (N - 1) / 2, 0, r - (N - 1) / 2)

const GRASS_A = new THREE.Color('#1f7a3d')
const GRASS_B = new THREE.Color('#2a8c48')
const HEAT_STOPS = ['#0b1f3a', '#1f4f8a', '#3f8fc9', '#a9c96a', '#ffd200'].map((c) => new THREE.Color(c))

function heatColor(t: number, out: THREE.Color) {
  const x = Math.max(0, Math.min(0.9999, t)) * (HEAT_STOPS.length - 1)
  const i = Math.floor(x)
  return out.copy(HEAT_STOPS[i]).lerp(HEAT_STOPS[i + 1], x - i)
}

/* ---------- Cancha: 400 losetas instanciadas (pasto o mapa de calor) ---------- */
export function Pitch({
  heat,
  onTile,
}: {
  heat: Float64Array | null
  onTile?: (r: number, c: number) => void
}) {
  const ref = useRef<THREE.InstancedMesh>(null)
  const tmp = useMemo(() => new THREE.Object3D(), [])
  const col = useMemo(() => new THREE.Color(), [])
  useLayoutEffect(() => {
    const m = ref.current!
    for (let r = 0; r < N; r++)
      for (let c = 0; c < N; c++) {
        const i = r * N + c
        tmp.position.copy(cellPos(r, c)).setY(-0.05)
        tmp.updateMatrix()
        m.setMatrixAt(i, tmp.matrix)
      }
    m.instanceMatrix.needsUpdate = true
  }, [tmp])
  useLayoutEffect(() => {
    const m = ref.current!
    let vmax = 0
    if (heat) for (let i = 0; i < heat.length; i++) vmax = Math.max(vmax, heat[i])
    const lmax = Math.log1p(vmax) || 1
    for (let r = 0; r < N; r++)
      for (let c = 0; c < N; c++) {
        const i = r * N + c
        if (heat && heat[i] > 0) heatColor(Math.log1p(heat[i]) / lmax, col)
        else col.copy(r % 2 ? GRASS_A : GRASS_B)
        m.setColorAt(i, col)
      }
    if (m.instanceColor) m.instanceColor.needsUpdate = true
  })
  return (
    <instancedMesh
      ref={ref}
      args={[undefined, undefined, N * N]}
      receiveShadow
      onClick={(e: ThreeEvent<MouseEvent>) => {
        if (!onTile || e.instanceId === undefined) return
        e.stopPropagation()
        onTile(Math.floor(e.instanceId / N), e.instanceId % N)
      }}
    >
      <boxGeometry args={[0.97, 0.1, 0.97]} />
      <meshStandardMaterial roughness={0.95} />
    </instancedMesh>
  )
}

/* ---------- Estadio: líneas, gradas, torres de luz ---------- */
export function Stadium() {
  const line = (x: number, z: number, w: number, d: number, k: string) => (
    <mesh key={k} position={[x, 0.006, z]} receiveShadow>
      <boxGeometry args={[w, 0.012, d]} />
      <meshStandardMaterial color="#f3f3e6" />
    </mesh>
  )
  const half = N / 2 + 0.15
  const stands = useMemo(() => {
    const out: { pos: [number, number, number]; size: [number, number, number]; color: string }[] = []
    for (let side = 0; side < 4; side++)
      for (let k = 0; k < 4; k++) {
        const dist = half + 2.2 + k * 1.1
        const h = 0.6 + k * 0.75
        const len = N + 6 + k * 2.2
        const color = k % 2 ? '#0b1f3a' : '#16365f'
        const along = side < 2
        const sign = side % 2 ? 1 : -1
        out.push({
          pos: along ? [0, h / 2 - 0.1, sign * dist] : [sign * dist, h / 2 - 0.1, 0],
          size: along ? [len, h, 1.1] : [1.1, h, len],
          color,
        })
      }
    return out
  }, [half])
  return (
    <group>
      <mesh rotation-x={-Math.PI / 2} position={[0, -0.11, 0]} receiveShadow>
        <planeGeometry args={[N + 7, N + 7]} />
        <meshStandardMaterial color="#17602f" roughness={1} />
      </mesh>
      {line(0, -half, N + 0.3, 0.08, 't')}
      {line(0, half, N + 0.3, 0.08, 'b')}
      {line(-half, 0, 0.08, N + 0.3, 'l')}
      {line(half, 0, 0.08, N + 0.3, 'r')}
      {stands.map((s, i) => (
        <mesh key={i} position={s.pos} castShadow receiveShadow>
          <boxGeometry args={s.size} />
          <meshStandardMaterial color={s.color} roughness={0.8} />
        </mesh>
      ))}
      {[
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
      ].map(([sx, sz]) => (
        <group key={`${sx}${sz}`} position={[sx * (half + 6), 0, sz * (half + 6)]}>
          <mesh position={[0, 5, 0]}>
            <cylinderGeometry args={[0.12, 0.18, 10, 8]} />
            <meshStandardMaterial color="#c8ccd6" metalness={0.6} roughness={0.4} />
          </mesh>
          <mesh position={[0, 10.2, 0]} rotation-y={Math.atan2(sx, sz)}>
            <boxGeometry args={[1.8, 0.9, 0.3]} />
            <meshStandardMaterial color="#fff6d5" emissive="#fff6d5" emissiveIntensity={2.2} />
          </mesh>
        </group>
      ))}
    </group>
  )
}

/* ---------- La barrera: defensas en cada obstáculo ---------- */
export interface DefendersHandle {
  hit: (r: number, c: number) => void
}

export const Defenders = forwardRef<DefendersHandle, { map: MapSpec }>(function Defenders({ map }, ref) {
  const bodies = useRef<THREE.InstancedMesh>(null)
  const heads = useRef<THREE.InstancedMesh>(null)
  const stripes = useRef<THREE.InstancedMesh>(null)
  const cells = useMemo(() => {
    const out: [number, number][] = []
    for (let i = 0; i < map.obstacles.length; i++) if (map.obstacles[i]) out.push([Math.floor(i / map.cols), i % map.cols])
    return out
  }, [map])
  const jolt = useRef(new Map<number, number>())
  useImperativeHandle(ref, () => ({
    hit: (r, c) => {
      const k = cells.findIndex(([a, b]) => a === r && b === c)
      if (k >= 0) jolt.current.set(k, 1)
    },
  }))
  const tmp = useMemo(() => new THREE.Object3D(), [])
  useFrame(({ clock }, dt) => {
    const t = clock.elapsedTime
    cells.forEach(([r, c], k) => {
      const p = cellPos(r, c)
      const j = jolt.current.get(k) ?? 0
      const bob = Math.sin(t * 3 + k * 1.7) * 0.03 + j * Math.abs(Math.sin(j * 9)) * 0.5
      const lean = j * Math.sin(j * 14) * 0.5
      tmp.position.set(p.x, 0.42 + bob, p.z)
      tmp.rotation.set(lean, Math.sin(t * 0.8 + k) * 0.3, 0)
      tmp.updateMatrix()
      bodies.current!.setMatrixAt(k, tmp.matrix)
      tmp.position.set(p.x, 0.47 + bob, p.z)
      tmp.updateMatrix()
      stripes.current!.setMatrixAt(k, tmp.matrix)
      tmp.position.set(p.x, 0.92 + bob, p.z)
      tmp.updateMatrix()
      heads.current!.setMatrixAt(k, tmp.matrix)
      if (j > 0) jolt.current.set(k, Math.max(0, j - dt * 1.6))
    })
    for (const m of [bodies.current!, heads.current!, stripes.current!]) {
      m.count = cells.length
      m.instanceMatrix.needsUpdate = true
    }
  })
  return (
    <group>
      <instancedMesh ref={bodies} args={[undefined, undefined, 400]} castShadow>
        <capsuleGeometry args={[0.22, 0.42, 4, 10]} />
        <meshStandardMaterial color="#0b1f3a" roughness={0.6} />
      </instancedMesh>
      <instancedMesh ref={stripes} args={[undefined, undefined, 400]}>
        <cylinderGeometry args={[0.232, 0.232, 0.09, 12]} />
        <meshStandardMaterial color="#ffd200" roughness={0.5} />
      </instancedMesh>
      <instancedMesh ref={heads} args={[undefined, undefined, 400]} castShadow>
        <sphereGeometry args={[0.15, 14, 10]} />
        <meshStandardMaterial color="#e2b48c" roughness={0.7} />
      </instancedMesh>
    </group>
  )
})

/* ---------- Portería en B y punto de salida en A ---------- */
export function Goal({ r, c }: { r: number; c: number }) {
  const p = cellPos(r, c)
  const netTex = useMemo(() => {
    const cv = document.createElement('canvas')
    cv.width = cv.height = 64
    const g = cv.getContext('2d')!
    g.strokeStyle = 'rgba(255,255,255,0.85)'
    g.lineWidth = 3
    for (let k = 0; k <= 64; k += 16) {
      g.beginPath()
      g.moveTo(k, 0)
      g.lineTo(k, 64)
      g.moveTo(0, k)
      g.lineTo(64, k)
      g.stroke()
    }
    const t = new THREE.CanvasTexture(cv)
    t.wrapS = t.wrapT = THREE.RepeatWrapping
    t.repeat.set(4, 3)
    return t
  }, [])
  const post = (x: number, z: number, h: number, key: string) => (
    <mesh key={key} position={[x, h / 2, z]} castShadow>
      <cylinderGeometry args={[0.04, 0.04, h, 8]} />
      <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.25} />
    </mesh>
  )
  // Boca de la portería hacia el oeste (hacia A); red hacia +x
  return (
    <group position={[p.x + 0.15, 0, p.z]}>
      {post(-0.3, -0.48, 0.8, 'p1')}
      {post(-0.3, 0.48, 0.8, 'p2')}
      <mesh position={[-0.3, 0.8, 0]} rotation-x={Math.PI / 2}>
        <cylinderGeometry args={[0.04, 0.04, 1.0, 8]} />
        <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.25} />
      </mesh>
      <mesh position={[0.08, 0.42, 0]} rotation-y={Math.PI / 2}>
        <planeGeometry args={[1, 0.84]} />
        <meshStandardMaterial map={netTex} transparent side={THREE.DoubleSide} alphaTest={0.1} />
      </mesh>
      <mesh position={[-0.11, 0.8, 0]} rotation-x={-Math.PI / 2}>
        <planeGeometry args={[0.38, 1]} />
        <meshStandardMaterial map={netTex} transparent side={THREE.DoubleSide} alphaTest={0.1} />
      </mesh>
      {[-0.48, 0.48].map((z) => (
        <mesh key={z} position={[-0.11, 0.42, z]}>
          <planeGeometry args={[0.38, 0.84]} />
          <meshStandardMaterial map={netTex} transparent side={THREE.DoubleSide} alphaTest={0.1} />
        </mesh>
      ))}
    </group>
  )
}

export function StartSpot({ r, c }: { r: number; c: number }) {
  const p = cellPos(r, c)
  return (
    <group position={[p.x, 0.012, p.z]}>
      <mesh rotation-x={-Math.PI / 2}>
        <ringGeometry args={[0.36, 0.44, 32]} />
        <meshStandardMaterial color="#fff6d5" />
      </mesh>
      <mesh rotation-x={-Math.PI / 2}>
        <circleGeometry args={[0.08, 16]} />
        <meshStandardMaterial color="#fff6d5" />
      </mesh>
    </group>
  )
}

/** Anillo pulsante que marca la brecha peligrosa. */
export function DangerMark({ cells }: { cells: [number, number][] }) {
  const ref = useRef<THREE.Group>(null)
  useFrame(({ clock }) => {
    const s = 1 + Math.sin(clock.elapsedTime * 4) * 0.08
    ref.current?.children.forEach((ch) => ch.scale.set(s, s, s))
  })
  return (
    <group ref={ref}>
      {cells.map(([r, c]) => {
        const p = cellPos(r, c)
        return (
          <mesh key={`${r}-${c}`} position={[p.x, 0.02, p.z]} rotation-x={-Math.PI / 2}>
            <ringGeometry args={[0.38, 0.47, 4, 1, Math.PI / 4]} />
            <meshBasicMaterial color="#ff4d4d" transparent opacity={0.9} />
          </mesh>
        )
      })}
    </group>
  )
}

/* ---------- Flechas de política ---------- */
export function PolicyArrows({ map, policy, color = '#fff6d5' }: { map: MapSpec; policy: number[] | null; color?: string }) {
  const ref = useRef<THREE.InstancedMesh>(null)
  const geo = useMemo(() => {
    const g = new THREE.ConeGeometry(0.13, 0.42, 3)
    g.rotateX(Math.PI / 2) // apunta a +z
    return g
  }, [])
  useLayoutEffect(() => {
    const m = ref.current
    if (!m || !policy) return
    const tmp = new THREE.Object3D()
    let k = 0
    for (let s = 0; s < policy.length; s++) {
      if (map.obstacles[s]) continue
      const r = Math.floor(s / map.cols)
      const c = s % map.cols
      if (r === map.goal[0] && c === map.goal[1]) continue
      const [dr, dc] = ACTION_DELTAS[policy[s]]
      tmp.position.copy(cellPos(r, c)).setY(0.09)
      tmp.rotation.set(0, Math.atan2(dc, dr), 0)
      tmp.updateMatrix()
      m.setMatrixAt(k++, tmp.matrix)
    }
    m.count = k
    m.instanceMatrix.needsUpdate = true
  }, [map, policy])
  if (!policy) return null
  return (
    <instancedMesh ref={ref} args={[geo, undefined, 400]}>
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.35} />
    </instancedMesh>
  )
}

/* ---------- Partículas (confeti, chispas, polvo) ---------- */
export interface ParticlesHandle {
  burst: (pos: THREE.Vector3, opts: { colors: string[]; count: number; speed: number; up?: number; gravity?: number; life?: number; size?: number }) => void
}

const MAXP = 900
export const Particles = forwardRef<ParticlesHandle>(function Particles(_, ref) {
  const pts = useRef<THREE.Points>(null)
  const state = useMemo(
    () => ({
      pos: new Float32Array(MAXP * 3),
      vel: new Float32Array(MAXP * 3),
      col: new Float32Array(MAXP * 3),
      life: new Float32Array(MAXP),
      grav: new Float32Array(MAXP),
      next: 0,
    }),
    [],
  )
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(state.pos, 3))
    g.setAttribute('color', new THREE.BufferAttribute(state.col, 3))
    return g
  }, [state])
  useImperativeHandle(ref, () => ({
    burst: (p, o) => {
      const c = new THREE.Color()
      for (let k = 0; k < o.count; k++) {
        const i = state.next
        state.next = (state.next + 1) % MAXP
        state.pos.set([p.x, p.y, p.z], i * 3)
        const th = Math.random() * Math.PI * 2
        const sp = o.speed * (0.4 + Math.random() * 0.6)
        state.vel.set([Math.cos(th) * sp, (o.up ?? 1) * (0.5 + Math.random()) * o.speed, Math.sin(th) * sp], i * 3)
        c.set(o.colors[k % o.colors.length])
        state.col.set([c.r, c.g, c.b], i * 3)
        state.life[i] = (o.life ?? 1.4) * (0.6 + Math.random() * 0.4)
        state.grav[i] = o.gravity ?? 6
      }
    },
  }))
  useFrame((_, dtRaw) => {
    const dt = Math.min(dtRaw, 0.05)
    const { pos, vel, life, grav } = state
    for (let i = 0; i < MAXP; i++) {
      if (life[i] <= 0) {
        pos[i * 3 + 1] = -50
        continue
      }
      life[i] -= dt
      vel[i * 3 + 1] -= grav[i] * dt
      pos[i * 3] += vel[i * 3] * dt
      pos[i * 3 + 1] += vel[i * 3 + 1] * dt
      pos[i * 3 + 2] += vel[i * 3 + 2] * dt
      if (pos[i * 3 + 1] < 0.02) {
        pos[i * 3 + 1] = 0.02
        vel[i * 3 + 1] *= -0.3
        vel[i * 3] *= 0.7
        vel[i * 3 + 2] *= 0.7
      }
    }
    geo.attributes.position.needsUpdate = true
    geo.attributes.color.needsUpdate = true
  })
  return (
    <points ref={pts} geometry={geo} frustumCulled={false}>
      <pointsMaterial size={0.11} vertexColors sizeAttenuation depthWrite={false} />
    </points>
  )
})

/* ---------- Balón con textura procedural ---------- */
export function useBallTexture() {
  return useMemo(() => {
    const cv = document.createElement('canvas')
    cv.width = 128
    cv.height = 64
    const g = cv.getContext('2d')!
    g.fillStyle = '#ffffff'
    g.fillRect(0, 0, 128, 64)
    g.fillStyle = '#111827'
    for (const [x, y] of [
      [16, 16], [48, 32], [80, 16], [112, 32], [32, 52], [96, 52], [64, 6], [0, 40], [128, 40],
    ]) {
      g.beginPath()
      for (let k = 0; k < 5; k++) {
        const a = (k / 5) * Math.PI * 2 - Math.PI / 2
        g.lineTo(x + Math.cos(a) * 8, y + Math.sin(a) * 8)
      }
      g.fill()
    }
    return new THREE.CanvasTexture(cv)
  }, [])
}

/* ---------- Afición en las gradas (hace "la ola" cuando hay gol) ---------- */
export interface CrowdHandle {
  cheer: () => void
}
export const Crowd = forwardRef<CrowdHandle>(function Crowd(_, ref) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const fans = useMemo(() => {
    const out: { x: number; y: number; z: number; phase: number; along: number }[] = []
    const half = N / 2 + 0.15
    for (let side = 0; side < 4; side++)
      for (let k = 0; k < 4; k++) {
        const dist = half + 2.2 + k * 1.1
        const h = 0.6 + k * 0.75
        const len = N + 4 + k * 2.2
        for (let t = -len / 2 + 0.4; t < len / 2 - 0.3; t += 0.55) {
          if (Math.random() < 0.18) continue
          const along = side < 2
          const sign = side % 2 ? 1 : -1
          out.push({
            x: along ? t : sign * dist,
            z: along ? sign * dist : t,
            y: h - 0.1 + 0.18,
            phase: Math.random() * Math.PI * 2,
            along: t,
          })
        }
      }
    return out
  }, [])
  const wave = useRef(-1)
  useImperativeHandle(ref, () => ({ cheer: () => (wave.current = 0) }))
  useLayoutEffect(() => {
    const m = mesh.current!
    const c = new THREE.Color()
    const palette = ['#ffd200', '#ffd200', '#0b1f3a', '#ffffff', '#1d4e89', '#ffe680']
    fans.forEach((_, i) => m.setColorAt(i, c.set(palette[i % palette.length])))
    if (m.instanceColor) m.instanceColor.needsUpdate = true
  }, [fans])
  const tmp = useMemo(() => new THREE.Object3D(), [])
  useFrame(({ clock }, dt) => {
    const m = mesh.current!
    const t = clock.elapsedTime
    if (wave.current >= 0) wave.current += dt
    if (wave.current > 3.2) wave.current = -1
    fans.forEach((f, i) => {
      let jump = Math.max(0, Math.sin(t * 2 + f.phase)) * 0.04
      if (wave.current >= 0) {
        const front = -16 + wave.current * 16
        const d = Math.abs(f.along - front)
        if (d < 2.2) jump += (1 - d / 2.2) * 0.45
        jump += Math.max(0, Math.sin(t * 14 + f.phase)) * 0.12
      }
      tmp.position.set(f.x, f.y + jump, f.z)
      tmp.updateMatrix()
      m.setMatrixAt(i, tmp.matrix)
    })
    m.instanceMatrix.needsUpdate = true
  })
  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, fans.length]}>
      <capsuleGeometry args={[0.13, 0.22, 3, 6]} />
      <meshStandardMaterial roughness={0.8} />
    </instancedMesh>
  )
})
