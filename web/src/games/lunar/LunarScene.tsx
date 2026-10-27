import { useFrame } from '@react-three/fiber'
import { forwardRef, useImperativeHandle, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { LANDER_POLY, LEG_H, LEG_W, SCALE, W, type LunarLanderSim } from './physics'

/** El mundo de Box2D va de x=0 a x=20; en la escena lo centramos en 0. */
export const OX = W / 2

function hash(x: number, z: number) {
  const s = Math.sin(x * 127.1 + z * 311.7) * 43758.5453
  return s - Math.floor(s)
}

/** Perfil físico del terreno (el mismo que colisiona en planck) + montañas decorativas atrás. */
export function makeHeight(sim: LunarLanderSim, seed: number) {
  const xs = sim.terrainX
  const ys = sim.terrainY
  const profile = (x: number) => {
    if (x <= xs[0]) return ys[0] + (xs[0] - x) * 0.55
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1] + (x - xs[xs.length - 1]) * 0.55
    const i = Math.min(xs.length - 2, Math.floor((x - xs[0]) / (xs[1] - xs[0])))
    const u = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] * (1 - u) + ys[i + 1] * u
  }
  const ph = seed * 0.37
  const mountains = (x: number, z: number) =>
    1.6 * Math.sin(0.33 * x + ph + 0.21 * z) +
    1.1 * Math.sin(0.61 * z - 0.27 * x + ph * 2) +
    0.6 * Math.sin(1.3 * x + 0.9 * z) +
    0.3 * (hash(Math.round(x * 2), Math.round(z * 2)) - 0.5)
  const craters: [number, number, number][] = Array.from({ length: 7 }, (_, k) => [
    (hash(seed + k, 1) - 0.5) * 30 + OX,
    -3 - hash(seed + k, 2) * 10,
    0.8 + hash(seed + k, 3) * 1.8,
  ])
  return { profile, height: (x: number, z: number) => {
    const base = profile(x)
    if (z > 0.9) return Math.max(base - (z - 0.9) * 0.9, -1.2) + 0.05 * Math.sin(x * 3 + z * 2)
    if (z >= -0.9) return base
    const k = Math.min(1, (-z - 0.9) / 9)
    let y = base + k * k * (1.4 + 0.75 * mountains(x, z))
    for (const [cx, cz, r] of craters) {
      const d = Math.hypot(x - cx, z - cz)
      if (d < r * 1.3) y += d < r ? -0.5 * (1 - (d / r) ** 2) : 0.25 * Math.sin(((d - r) / (0.3 * r)) * Math.PI)
    }
    return y
  } }
}

export function Terrain({ sim, seed }: { sim: LunarLanderSim; seed: number }) {
  const geo = useMemo(() => {
    const { height } = makeHeight(sim, seed)
    const nx = 140
    const nz = 60
    const g = new THREE.PlaneGeometry(36, 24, nx, nz)
    g.rotateX(-Math.PI / 2) // plano XZ
    g.translate(0, 0, -6) // de z=-18 a z=+6
    const pos = g.attributes.position as THREE.BufferAttribute
    const colors = new Float32Array(pos.count * 3)
    const c = new THREE.Color()
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i)
      const z = pos.getZ(i)
      const y = height(x + OX, z)
      pos.setY(i, y)
      const n = hash(Math.round(x * 3), Math.round(z * 3))
      // gris lunar; más oscuro al frente para que el módulo destaque
      const front = Math.max(0, Math.min(1, (z - 0.5) / 5))
      c.setHSL(0.62, 0.05, 0.24 + Math.min(0.12, Math.max(0, y) * 0.012) + n * 0.06 - front * 0.1)
      colors.set([c.r, c.g, c.b], i * 3)
    }
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    g.computeVertexNormals()
    return g
  }, [sim, seed])
  const rim = useMemo(() => {
    const pts = sim.terrainX.map((x, i) => new THREE.Vector3(x - OX, sim.terrainY[i] + 0.01, 0.95))
    return new THREE.BufferGeometry().setFromPoints(pts)
  }, [sim, seed]) // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <group>
      <mesh geometry={geo} receiveShadow>
        <meshStandardMaterial vertexColors flatShading roughness={1} metalness={0} />
      </mesh>
      <lineSegments geometry={rim}>
        <lineBasicMaterial color="#ffd200" transparent opacity={0.25} />
      </lineSegments>
    </group>
  )
}

export function Helipad({ x1, x2, y }: { x1: number; x2: number; y: number }) {
  const flags = useRef<THREE.Group>(null)
  useFrame(({ clock }) => {
    flags.current?.children.forEach((f, i) => {
      const cloth = f.children[1]
      if (cloth) cloth.rotation.y = Math.sin(clock.elapsedTime * 3 + i) * 0.35
    })
  })
  const w = x2 - x1
  return (
    <group>
      <mesh position={[(x1 + x2) / 2 - OX, y - 0.04, 0]} receiveShadow>
        <boxGeometry args={[w + 0.3, 0.08, 2]} />
        <meshStandardMaterial color="#0b1f3a" metalness={0.5} roughness={0.4} />
      </mesh>
      {[-0.35, 0.35].map((dz) => (
        <mesh key={dz} position={[(x1 + x2) / 2 - OX, y + 0.002, dz]}>
          <boxGeometry args={[w, 0.01, 0.08]} />
          <meshStandardMaterial color="#ffd200" emissive="#ffd200" emissiveIntensity={0.6} />
        </mesh>
      ))}
      <Beacons xs={[x1 - 0.1, x2 + 0.1]} y={y} />
      <group ref={flags}>
        {[x1, x2].map((x) => (
          <group key={x} position={[x - OX, y, 0.2]}>
            <mesh position={[0, 0.5, 0]}>
              <cylinderGeometry args={[0.02, 0.02, 1, 6]} />
              <meshStandardMaterial color="#e5e7eb" />
            </mesh>
            <mesh position={[0.2, 0.86, 0]}>
              <planeGeometry args={[0.4, 0.26]} />
              <meshStandardMaterial color="#ffd200" emissive="#ffd200" emissiveIntensity={0.4} side={THREE.DoubleSide} />
            </mesh>
          </group>
        ))}
      </group>
    </group>
  )
}

export function Earth() {
  const tex = useMemo(() => {
    const cv = document.createElement('canvas')
    cv.width = 256
    cv.height = 128
    const g = cv.getContext('2d')!
    g.fillStyle = '#1d4e89'
    g.fillRect(0, 0, 256, 128)
    for (let k = 0; k < 26; k++) {
      g.fillStyle = k % 3 ? '#3f7d3a' : '#c2a46b'
      g.beginPath()
      g.ellipse(hash(k, 1) * 256, 20 + hash(k, 2) * 88, 8 + hash(k, 3) * 26, 5 + hash(k, 4) * 14, hash(k, 5) * 3, 0, Math.PI * 2)
      g.fill()
    }
    g.fillStyle = 'rgba(255,255,255,0.7)'
    for (let k = 0; k < 40; k++) {
      g.beginPath()
      g.ellipse(hash(k, 7) * 256, hash(k, 8) * 128, 10 + hash(k, 9) * 20, 2 + hash(k, 10) * 4, 0, 0, Math.PI * 2)
      g.fill()
    }
    return new THREE.CanvasTexture(cv)
  }, [])
  const ref = useRef<THREE.Mesh>(null)
  useFrame((_, dt) => ref.current && (ref.current.rotation.y += dt * 0.03))
  return (
    <group position={[-26, 22, -70]}>
      <mesh ref={ref}>
        <sphereGeometry args={[5, 48, 32]} />
        <meshStandardMaterial map={tex} emissive="#1d4e89" emissiveIntensity={0.25} roughness={0.8} />
      </mesh>
      <mesh scale={1.08}>
        <sphereGeometry args={[5, 32, 24]} />
        <meshBasicMaterial color="#7cc4ff" transparent opacity={0.12} side={THREE.BackSide} />
      </mesh>
    </group>
  )
}

/* ---------- Módulo lunar ---------- */
export interface LanderHandle {
  sync: (sim: LunarLanderSim) => void
  flame: (kind: 'main' | 'side', x: number, y: number, dx: number, dy: number) => void
  setVisible: (v: boolean) => void
}

export const Lander = forwardRef<LanderHandle>(function Lander(_, ref) {
  const body = useRef<THREE.Group>(null)
  const legs = [useRef<THREE.Group>(null), useRef<THREE.Group>(null)]
  const mainFlame = useRef<THREE.Mesh>(null)
  const sideFlame = useRef<THREE.Mesh>(null)
  const ttl = useRef({ main: 0, side: 0 })
  const shape = useMemo(() => {
    const s = new THREE.Shape()
    LANDER_POLY.forEach(([x, y], i) => (i ? s.lineTo(x / SCALE, y / SCALE) : s.moveTo(x / SCALE, y / SCALE)))
    s.closePath()
    const g = new THREE.ExtrudeGeometry(s, { depth: 0.62, bevelEnabled: true, bevelSize: 0.03, bevelThickness: 0.03, bevelSegments: 2 })
    g.translate(0, 0, -0.31)
    return g
  }, [])
  const flameGeo = useMemo(() => {
    const g = new THREE.ConeGeometry(0.13, 1, 14, 1, true)
    g.rotateX(Math.PI)
    g.translate(0, -0.5, 0) // base ancha en la tobera (origen), punta hacia -y
    return g
  }, [])

  useImperativeHandle(ref, () => ({
    sync: (sim) => {
      const p = sim.pose()
      if (body.current) {
        body.current.position.set(p.lander.x - OX, p.lander.y, 0)
        body.current.rotation.z = p.lander.angle
      }
      p.legs.forEach((l, i) => {
        const g = legs[i].current
        if (g) {
          g.position.set(l.x - OX, l.y, 0)
          g.rotation.z = l.angle
        }
      })
    },
    flame: (kind, x, y, dx, dy) => {
      const m = kind === 'main' ? mainFlame.current : sideFlame.current
      if (!m) return
      const off = kind === 'main' ? 0.22 : 0.05
      m.position.set(x - OX + dx * off, y + dy * off, 0)
      // la punta del cono apunta en la dirección del chorro (dx, dy)
      m.rotation.set(0, 0, Math.atan2(dy, dx) + Math.PI / 2)
      ttl.current[kind] = 0.09
    },
    setVisible: (v) => {
      ;[body.current, legs[0].current, legs[1].current].forEach((g) => g && (g.visible = v))
      if (!v) ttl.current = { main: 0, side: 0 }
    },
  }))

  useFrame((_, dt) => {
    for (const k of ['main', 'side'] as const) {
      const m = k === 'main' ? mainFlame.current : sideFlame.current
      if (!m) continue
      ttl.current[k] -= dt
      m.visible = ttl.current[k] > 0
      const len = k === 'main' ? 0.85 + Math.random() * 0.5 : 0.35 + Math.random() * 0.15
      const w = k === 'main' ? 1.3 + Math.random() * 0.3 : 0.55
      m.scale.set(w, len, w)
    }
  })

  return (
    <group>
      <group ref={body}>
        <mesh geometry={shape} castShadow>
          <meshStandardMaterial color="#ffd200" metalness={0.35} roughness={0.32} emissive="#4a3a00" emissiveIntensity={0.35} />
        </mesh>
        <mesh position={[0, -0.12, 0]}>
          <boxGeometry args={[1.18, 0.1, 0.7]} />
          <meshStandardMaterial color="#0b1f3a" metalness={0.5} roughness={0.4} />
        </mesh>
        <mesh position={[0, 0.25, 0.345]}>
          <circleGeometry args={[0.12, 20]} />
          <meshStandardMaterial color="#7cf5ff" emissive="#7cf5ff" emissiveIntensity={1.6} />
        </mesh>
        <mesh position={[0.25, 0.68, 0]}>
          <cylinderGeometry args={[0.01, 0.01, 0.25, 6]} />
          <meshStandardMaterial color="#e5e7eb" />
        </mesh>
        <mesh position={[0, -0.36, 0]}>
          <cylinderGeometry args={[0.1, 0.16, 0.12, 14]} />
          <meshStandardMaterial color="#374151" metalness={0.8} roughness={0.3} />
        </mesh>
      </group>
      {legs.map((ref, i) => (
        <group key={i} ref={ref}>
          {[-0.28, 0.28].map((z) => (
            <group key={z} position={[0, 0, z]}>
              <mesh castShadow>
                <boxGeometry args={[(2 * LEG_W) / SCALE, (2 * LEG_H) / SCALE, 0.06]} />
                <meshStandardMaterial color="#c8ccd6" metalness={0.7} roughness={0.35} />
              </mesh>
              <mesh position={[0, -LEG_H / SCALE, 0]}>
                <cylinderGeometry args={[0.09, 0.11, 0.035, 12]} />
                <meshStandardMaterial color="#9ca3af" metalness={0.6} />
              </mesh>
            </group>
          ))}
        </group>
      ))}
      <mesh ref={mainFlame} geometry={flameGeo} visible={false}>
        <meshBasicMaterial color="#ffb02e" transparent opacity={0.9} blending={THREE.AdditiveBlending} depthWrite={false} />
      </mesh>
      <mesh ref={sideFlame} geometry={flameGeo} visible={false}>
        <meshBasicMaterial color="#7cf5ff" transparent opacity={0.9} blending={THREE.AdditiveBlending} depthWrite={false} />
      </mesh>
    </group>
  )
})

/* ---------- Partículas aditivas (gases, polvo, explosión, confeti) ---------- */
export interface FxHandle {
  emit: (x: number, y: number, z: number, vx: number, vy: number, vz: number, color: THREE.Color, life: number) => void
  burst: (x: number, y: number, colors: string[], n: number, speed: number, life: number, gravity?: number) => void
  debris: (x: number, y: number) => void
}

const MAX = 2400
export const Fx = forwardRef<FxHandle, { ground: (x: number) => number }>(function Fx({ ground }, ref) {
  const st = useMemo(
    () => ({
      pos: new Float32Array(MAX * 3),
      vel: new Float32Array(MAX * 3),
      col: new Float32Array(MAX * 3),
      base: new Float32Array(MAX * 3),
      life: new Float32Array(MAX),
      life0: new Float32Array(MAX),
      grav: new Float32Array(MAX),
      next: 0,
    }),
    [],
  )
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(st.pos, 3))
    g.setAttribute('color', new THREE.BufferAttribute(st.col, 3))
    return g
  }, [st])
  const sprite = useMemo(() => {
    const cv = document.createElement('canvas')
    cv.width = cv.height = 64
    const g = cv.getContext('2d')!
    const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32)
    grd.addColorStop(0, 'rgba(255,255,255,1)')
    grd.addColorStop(0.35, 'rgba(255,255,255,0.65)')
    grd.addColorStop(1, 'rgba(255,255,255,0)')
    g.fillStyle = grd
    g.fillRect(0, 0, 64, 64)
    return new THREE.CanvasTexture(cv)
  }, [])
  const debrisRef = useRef<THREE.InstancedMesh>(null)
  const deb = useRef(Array.from({ length: 14 }, () => ({ p: new THREE.Vector3(0, -99, 0), v: new THREE.Vector3(), r: new THREE.Euler(), w: new THREE.Vector3(), life: 0 })))
  const emit = (x: number, y: number, z: number, vx: number, vy: number, vz: number, color: THREE.Color, life: number, gravity = 0) => {
    const i = st.next
    st.next = (st.next + 1) % MAX
    st.pos.set([x - OX, y, z], i * 3)
    st.vel.set([vx, vy, vz], i * 3)
    st.base.set([color.r, color.g, color.b], i * 3)
    st.life[i] = st.life0[i] = life
    st.grav[i] = gravity
  }
  useImperativeHandle(ref, () => ({
    emit: (x, y, z, vx, vy, vz, color, life) => emit(x, y, z, vx, vy, vz, color, life),
    burst: (x, y, colors, n, speed, life, gravity = 4) => {
      const c = new THREE.Color()
      for (let k = 0; k < n; k++) {
        const th = Math.random() * Math.PI * 2
        const ph = Math.acos(2 * Math.random() - 1)
        const sp = speed * (0.3 + Math.random() * 0.7)
        c.set(colors[k % colors.length])
        emit(x, y, 0, Math.sin(ph) * Math.cos(th) * sp, Math.abs(Math.cos(ph)) * sp, Math.sin(ph) * Math.sin(th) * sp, c, life * (0.5 + Math.random() * 0.5), gravity)
      }
    },
    debris: (x, y) => {
      deb.current.forEach((d) => {
        d.p.set(x - OX + (Math.random() - 0.5) * 0.6, y + Math.random() * 0.4, (Math.random() - 0.5) * 0.6)
        d.v.set((Math.random() - 0.5) * 7, 2 + Math.random() * 6, (Math.random() - 0.5) * 4)
        d.w.set(Math.random() * 10, Math.random() * 10, Math.random() * 10)
        d.life = 3
      })
    },
  }))
  const tmp = useMemo(() => new THREE.Object3D(), [])
  useFrame((_, dtRaw) => {
    const dt = Math.min(dtRaw, 0.05)
    const { pos, vel, col, base, life, life0, grav } = st
    for (let i = 0; i < MAX; i++) {
      if (life[i] <= 0) {
        col[i * 3] = col[i * 3 + 1] = col[i * 3 + 2] = 0
        continue
      }
      life[i] -= dt
      const f = Math.max(0, life[i] / life0[i])
      vel[i * 3 + 1] -= grav[i] * dt
      pos[i * 3] += vel[i * 3] * dt
      pos[i * 3 + 1] += vel[i * 3 + 1] * dt
      pos[i * 3 + 2] += vel[i * 3 + 2] * dt
      const gy = ground(pos[i * 3] + OX)
      if (pos[i * 3 + 1] < gy && Math.abs(pos[i * 3 + 2]) < 1) {
        pos[i * 3 + 1] = gy
        vel[i * 3 + 1] *= -0.25
        vel[i * 3] *= 0.8
      }
      // desvanecer hacia negro funciona con mezcla aditiva
      col[i * 3] = base[i * 3] * f
      col[i * 3 + 1] = base[i * 3 + 1] * f
      col[i * 3 + 2] = base[i * 3 + 2] * f
    }
    geo.attributes.position.needsUpdate = true
    geo.attributes.color.needsUpdate = true
    const m = debrisRef.current
    if (m) {
      deb.current.forEach((d, k) => {
        if (d.life > 0) {
          d.life -= dt
          d.v.y -= 6 * dt
          d.p.addScaledVector(d.v, dt)
          const gy = ground(d.p.x + OX)
          if (d.p.y < gy + 0.05) {
            d.p.y = gy + 0.05
            d.v.multiplyScalar(0.5)
            d.v.y = Math.abs(d.v.y) * 0.4
          }
          d.r.x += d.w.x * dt
          d.r.y += d.w.y * dt
          d.r.z += d.w.z * dt
        }
        tmp.position.copy(d.life > 0 ? d.p : new THREE.Vector3(0, -99, 0))
        tmp.rotation.copy(d.r)
        tmp.updateMatrix()
        m.setMatrixAt(k, tmp.matrix)
      })
      m.instanceMatrix.needsUpdate = true
    }
  })
  return (
    <group>
      <points geometry={geo} frustumCulled={false}>
        <pointsMaterial map={sprite} size={0.3} vertexColors sizeAttenuation transparent depthWrite={false} blending={THREE.AdditiveBlending} />
      </points>
      <instancedMesh ref={debrisRef} args={[undefined, undefined, 14]} castShadow>
        <boxGeometry args={[0.16, 0.08, 0.12]} />
        <meshStandardMaterial color="#ffd200" metalness={0.8} roughness={0.3} />
      </instancedMesh>
    </group>
  )
})

function Beacons({ xs, y }: { xs: number[]; y: number }) {
  const mats = useRef<THREE.MeshStandardMaterial[]>([])
  useFrame(({ clock }) => {
    const on = Math.sin(clock.elapsedTime * 5) > 0.2
    mats.current.forEach((m, i) => (m.emissiveIntensity = on === (i % 2 === 0) ? 4 : 0.3))
  })
  return (
    <group>
      {xs.flatMap((x, i) =>
        [-0.9, 0.9].map((z, j) => (
          <mesh key={`${i}-${j}`} position={[x - OX, y + 0.06, z]}>
            <sphereGeometry args={[0.06, 10, 8]} />
            <meshStandardMaterial ref={(m) => void (m && (mats.current[i * 2 + j] = m))} color="#ffd200" emissive="#ffd200" />
          </mesh>
        )),
      )}
    </group>
  )
}
