// Port a TypeScript (planck.js = Box2D) de gymnasium/envs/box2d/lunar_lander.py,
// versión discreta, sin viento. Mismas constantes, misma recompensa y misma
// observación normalizada, para que la red DQN entrenada en Python funcione aquí.
import { Box, Edge, Polygon, RevoluteJoint, Vec2, World, type Body, type Contact } from 'planck'
import { mulberry32 } from '../../rl/gridworld'

export const FPS = 50
export const SCALE = 30
export const MAIN_ENGINE_POWER = 13.0
export const SIDE_ENGINE_POWER = 0.6
export const INITIAL_RANDOM = 1000.0
export const LANDER_POLY: [number, number][] = [[-14, 17], [-17, 0], [-17, -10], [17, -10], [17, 0], [14, 17]]
export const LEG_AWAY = 20
export const LEG_DOWN = 18
export const LEG_W = 2
export const LEG_H = 8
export const LEG_SPRING_TORQUE = 40
export const SIDE_ENGINE_HEIGHT = 14
export const SIDE_ENGINE_AWAY = 12
export const MAIN_ENGINE_Y_LOCATION = 4
export const VIEWPORT_W = 600
export const VIEWPORT_H = 400
export const W = VIEWPORT_W / SCALE // 20
export const H = VIEWPORT_H / SCALE // 13.33
export const CHUNKS = 11
export const MAX_STEPS = 1000 // TimeLimit de Gymnasium

/** Acciones: 0 nada, 1 propulsor izquierdo, 2 motor principal, 3 propulsor derecho. */
export type Action = 0 | 1 | 2 | 3

export interface EngineFx {
  /** Punto de aplicación del impulso (coordenadas del mundo) */
  x: number
  y: number
  /** Dirección del chorro (unitaria, sentido de salida de los gases) */
  dx: number
  dy: number
}

export interface StepResult {
  obs: number[]
  reward: number
  terminated: boolean
  truncated: boolean
  landed: boolean
  crashed: boolean
  main?: EngineFx
  side?: EngineFx
}

export interface BodyPose {
  x: number
  y: number
  angle: number
}

export class LunarLanderSim {
  rng: () => number
  fuelCost: boolean
  world!: World
  lander!: Body
  legs: Body[] = []
  legContact = [false, false]
  gameOver = false
  prevShaping: number | null = null
  helipadX1 = 0
  helipadX2 = 0
  helipadY = 0
  terrainX: number[] = []
  terrainY: number[] = []
  t = 0
  totalReward = 0
  mainFirings = 0
  sideFirings = 0
  done = false
  /** Última observación (la que ve el agente para decidir) */
  lastObs: number[] = []

  constructor(opts: { seed?: number; fuelCost?: boolean } = {}) {
    this.rng = mulberry32(opts.seed ?? 0)
    this.fuelCost = opts.fuelCost ?? true
  }

  private uniform(a: number, b: number) {
    return a + (b - a) * this.rng()
  }

  reset(seed?: number): number[] {
    if (seed !== undefined) this.rng = mulberry32(seed)
    this.world = new World({ gravity: Vec2(0, -10) })
    this.gameOver = false
    this.prevShaping = null
    this.legContact = [false, false]
    this.t = 0
    this.totalReward = 0
    this.mainFirings = 0
    this.sideFirings = 0
    this.done = false

    // Terreno
    const height = Array.from({ length: CHUNKS + 1 }, () => this.uniform(0, H / 2))
    const chunkX = Array.from({ length: CHUNKS }, (_, i) => (W / (CHUNKS - 1)) * i)
    this.helipadX1 = chunkX[Math.floor(CHUNKS / 2) - 1]
    this.helipadX2 = chunkX[Math.floor(CHUNKS / 2) + 1]
    this.helipadY = H / 4
    for (let k = -2; k <= 2; k++) height[Math.floor(CHUNKS / 2) + k] = this.helipadY
    // Igual que en Python: height[i-1] con i=0 toma el último elemento.
    const at = (i: number) => height[(i + height.length) % height.length]
    const smoothY = chunkX.map((_, i) => 0.33 * (at(i - 1) + at(i) + at(i + 1)))
    this.terrainX = chunkX
    this.terrainY = smoothY

    const moon = this.world.createBody()
    moon.createFixture(new Edge(Vec2(0, 0), Vec2(W, 0)))
    for (let i = 0; i < CHUNKS - 1; i++) {
      moon.createFixture({ shape: new Edge(Vec2(chunkX[i], smoothY[i]), Vec2(chunkX[i + 1], smoothY[i + 1])), density: 0, friction: 0.1 })
    }

    // Módulo
    const initialY = VIEWPORT_H / SCALE
    const initialX = VIEWPORT_W / SCALE / 2
    this.lander = this.world.createDynamicBody({ position: Vec2(initialX, initialY), angle: 0 })
    this.lander.createFixture({
      shape: new Polygon(LANDER_POLY.map(([x, y]) => Vec2(x / SCALE, y / SCALE))),
      density: 5.0,
      friction: 0.1,
      filterCategoryBits: 0x0010,
      filterMaskBits: 0x001,
      restitution: 0.0,
    })
    this.lander.applyForceToCenter(Vec2(this.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), this.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)), true)

    // Patas
    this.legs = []
    for (const i of [-1, 1]) {
      const leg = this.world.createDynamicBody({ position: Vec2(initialX - (i * LEG_AWAY) / SCALE, initialY), angle: i * 0.05 })
      leg.createFixture({
        shape: new Box(LEG_W / SCALE, LEG_H / SCALE),
        density: 1.0,
        restitution: 0.0,
        filterCategoryBits: 0x0020,
        filterMaskBits: 0x001,
      })
      this.world.createJoint(
        new RevoluteJoint({
            bodyA: this.lander,
            bodyB: leg,
            localAnchorA: Vec2(0, 0),
            localAnchorB: Vec2((i * LEG_AWAY) / SCALE, LEG_DOWN / SCALE),
            referenceAngle: 0,
            enableMotor: true,
            enableLimit: true,
            maxMotorTorque: LEG_SPRING_TORQUE,
            motorSpeed: 0.3 * i,
            lowerAngle: i === -1 ? 0.9 - 0.5 : -0.9,
            upperAngle: i === -1 ? 0.9 : -0.9 + 0.5,
          }),
      )
      this.legs.push(leg)
    }

    this.world.on('begin-contact', (c: Contact) => this.contact(c, true))
    this.world.on('end-contact', (c: Contact) => this.contact(c, false))

    // Gymnasium devuelve la observación tras un primer paso sin acción (no cuenta para el TimeLimit).
    const first = this.physicsStep(0)
    this.lastObs = first.obs
    return first.obs
  }

  private contact(c: Contact, begin: boolean) {
    const a = c.getFixtureA().getBody()
    const b = c.getFixtureB().getBody()
    if (begin && (a === this.lander || b === this.lander)) this.gameOver = true
    this.legs.forEach((leg, i) => {
      if (leg === a || leg === b) this.legContact[i] = begin
    })
  }

  private physicsStep(action: Action): StepResult {
    const angle = this.lander.getAngle()
    const tip = [Math.sin(angle), Math.cos(angle)]
    const side = [-tip[1], tip[0]]
    const dispersion = [this.uniform(-1, 1) / SCALE, this.uniform(-1, 1) / SCALE]
    const pos0 = this.lander.getPosition()
    let mPower = 0
    let sPower = 0
    let main: EngineFx | undefined
    let sideFx: EngineFx | undefined

    if (action === 2) {
      mPower = 1.0
      const ox = tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
      const oy = -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
      const ip = Vec2(pos0.x + ox, pos0.y + oy)
      this.lander.applyLinearImpulse(Vec2(-ox * MAIN_ENGINE_POWER * mPower, -oy * MAIN_ENGINE_POWER * mPower), ip, true)
      const n = Math.hypot(ox, oy) || 1
      main = { x: ip.x, y: ip.y, dx: ox / n, dy: oy / n }
    }

    if (action === 1 || action === 3) {
      const direction = action - 2
      sPower = 1.0
      const ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + (direction * SIDE_ENGINE_AWAY) / SCALE)
      const oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + (direction * SIDE_ENGINE_AWAY) / SCALE)
      const p = this.lander.getPosition()
      const ip = Vec2(p.x + ox - (tip[0] * 17) / SCALE, p.y + oy + (tip[1] * SIDE_ENGINE_HEIGHT) / SCALE)
      this.lander.applyLinearImpulse(Vec2(-ox * SIDE_ENGINE_POWER * sPower, -oy * SIDE_ENGINE_POWER * sPower), ip, true)
      const n = Math.hypot(ox, oy) || 1
      sideFx = { x: ip.x, y: ip.y, dx: ox / n, dy: oy / n }
    }

    this.world.step(1.0 / FPS, 6 * 30, 2 * 30)

    const pos = this.lander.getPosition()
    const vel = this.lander.getLinearVelocity()
    const obs = [
      (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
      (pos.y - (this.helipadY + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
      (vel.x * (VIEWPORT_W / SCALE / 2)) / FPS,
      (vel.y * (VIEWPORT_H / SCALE / 2)) / FPS,
      this.lander.getAngle(),
      (20.0 * this.lander.getAngularVelocity()) / FPS,
      this.legContact[0] ? 1 : 0,
      this.legContact[1] ? 1 : 0,
    ]

    let reward = 0
    const shaping =
      -100 * Math.sqrt(obs[0] * obs[0] + obs[1] * obs[1]) -
      100 * Math.sqrt(obs[2] * obs[2] + obs[3] * obs[3]) -
      100 * Math.abs(obs[4]) +
      10 * obs[6] +
      10 * obs[7]
    if (this.prevShaping !== null) reward = shaping - this.prevShaping
    this.prevShaping = shaping
    if (this.fuelCost) {
      reward -= mPower * 0.3
      reward -= sPower * 0.03
    }

    let terminated = false
    let crashed = false
    let landed = false
    if (this.gameOver || Math.abs(obs[0]) >= 1.0) {
      terminated = true
      crashed = true
      reward = -100
    }
    if (!this.lander.isAwake()) {
      terminated = true
      crashed = false
      landed = true
      reward = 100
    }
    return { obs, reward, terminated, truncated: false, landed, crashed, main, side: sideFx }
  }

  step(action: Action): StepResult {
    if (this.done) throw new Error('Episodio terminado: llama reset()')
    const r = this.physicsStep(action)
    this.t += 1
    this.totalReward += r.reward
    if (action === 2) this.mainFirings++
    if (action === 1 || action === 3) this.sideFirings++
    r.truncated = !r.terminated && this.t >= MAX_STEPS
    this.done = r.terminated || r.truncated
    this.lastObs = r.obs
    return r
  }

  /** Combustible "gastado" en puntos de recompensa (0.3 por motor principal, 0.03 por lateral). */
  get fuelUsed() {
    return this.mainFirings * 0.3 + this.sideFirings * 0.03
  }

  pose(): { lander: BodyPose; legs: BodyPose[] } {
    const p = this.lander.getPosition()
    return {
      lander: { x: p.x, y: p.y, angle: this.lander.getAngle() },
      legs: this.legs.map((l) => ({ x: l.getPosition().x, y: l.getPosition().y, angle: l.getAngle() })),
    }
  }
}

/** Controlador heurístico de Gymnasium (para validar el port). */
export function heuristic(s: number[]): Action {
  let angleTarg = s[0] * 0.5 + s[2] * 1.0
  angleTarg = Math.max(-0.4, Math.min(0.4, angleTarg))
  const hoverTarg = 0.55 * Math.abs(s[0])
  let angleTodo = (angleTarg - s[4]) * 0.5 - s[5] * 1.0
  let hoverTodo = (hoverTarg - s[1]) * 0.5 - s[3] * 0.5
  if (s[6] || s[7]) {
    angleTodo = 0
    hoverTodo = -s[3] * 0.5
  }
  if (hoverTodo > Math.abs(angleTodo) && hoverTodo > 0.05) return 2
  if (angleTodo < -0.05) return 3
  if (angleTodo > 0.05) return 1
  return 0
}
