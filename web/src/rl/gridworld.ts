// Port a TypeScript de python/gridworld.py (mismas reglas, mismas recompensas).

export const ACTION_NAMES = ['N', 'S', 'E', 'O', 'NE', 'NO', 'SE', 'SO'] as const
export const ACTION_DELTAS: ReadonlyArray<readonly [number, number]> = [
  [-1, 0], [1, 0], [0, 1], [0, -1], [-1, 1], [-1, -1], [1, 1], [1, -1],
]
export const ACTION_ARROWS = ['↑', '↓', '→', '←', '↗', '↖', '↘', '↙']
/** Rosa de los vientos en sentido horario: N, NE, E, SE, S, SO, O, NO. */
const COMPASS = [0, 4, 2, 6, 1, 7, 3, 5]
const COMPASS_POS = COMPASS.reduce<number[]>((acc, a, i) => ((acc[a] = i), acc), [])
export const STAY = -1

export const R_GOAL = 100
export const R_CRASH = -100
export const COST_CARDINAL = -1.0
export const COST_DIAGONAL = -1.41

export const SMUGGLER_MAP = [
  '..........#.........',
  '..........#.........',
  '..........#.........',
  '..........#.........',
  '......A...#.........',
  '..........#.........',
  '....................',
  '..........#....B....',
  '..........#.........',
  '..........#.........',
  '..........#.........',
  '..........#.........',
  '....................',
  '....................',
  '....................',
  '....................',
  '....................',
  '....................',
  '....................',
  '....................',
]

export type Pos = readonly [number, number]

export function rotate(action: number, steps: number): number {
  return COMPASS[(((COMPASS_POS[action] + steps) % 8) + 8) % 8]
}

export function stepCost(action: number): number {
  return action < 4 ? COST_CARDINAL : COST_DIAGONAL
}

/** PRNG determinista (mulberry32) para que los experimentos sean reproducibles. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export interface MapSpec {
  rows: number
  cols: number
  obstacles: boolean[]
  start: Pos
  goal: Pos
}

export function parseMap(rows: string[]): MapSpec {
  const obstacles: boolean[] = []
  let start: Pos | null = null
  let goal: Pos | null = null
  rows.forEach((row, i) =>
    [...row].forEach((c, j) => {
      obstacles.push(c === '#')
      if (c === 'A') start = [i, j]
      if (c === 'B') goal = [i, j]
    }),
  )
  if (!start || !goal) throw new Error("El mapa necesita 'A' y 'B'")
  return { rows: rows.length, cols: rows[0].length, obstacles, start, goal }
}

export interface StepResult {
  state: number
  reward: number
  done: boolean
  truncated: boolean
  realAction: number
}

export class GridWorld {
  map: MapSpec
  pSlip: number
  maxSteps: number
  rng: () => number
  pos: Pos
  t = 0

  constructor(map: MapSpec, opts: { pSlip?: number; maxSteps?: number; seed?: number } = {}) {
    this.map = map
    this.pSlip = opts.pSlip ?? 0.1
    this.maxSteps = opts.maxSteps ?? 400
    this.rng = mulberry32(opts.seed ?? 0)
    this.pos = map.start
  }

  get nStates() {
    return this.map.rows * this.map.cols
  }

  toState(p: Pos) {
    return p[0] * this.map.cols + p[1]
  }

  toPos(s: number): Pos {
    return [Math.floor(s / this.map.cols), s % this.map.cols]
  }

  blocked(r: number, c: number) {
    if (r < 0 || c < 0 || r >= this.map.rows || c >= this.map.cols) return true
    return this.map.obstacles[r * this.map.cols + c]
  }

  /** Distribución de movimientos reales: [probabilidad, acción real]. */
  outcomes(action: number): Array<[number, number]> {
    const p = this.pSlip
    const out: Array<[number, number]> = [[1 - p, action]]
    if (p > 0) out.push([p / 3, rotate(action, -1)], [p / 3, rotate(action, 1)], [p / 3, STAY])
    return out
  }

  /** Regla de colisión: sólo choca si la celda DESTINO está bloqueada o fuera de la malla. */
  transition(pos: Pos, real: number, commanded: number): [Pos, number, boolean] {
    if (real === STAY) return [pos, stepCost(commanded), false]
    const [dr, dc] = ACTION_DELTAS[real]
    const nr = pos[0] + dr
    const nc = pos[1] + dc
    if (this.blocked(nr, nc)) return [pos, R_CRASH, true]
    if (nr === this.map.goal[0] && nc === this.map.goal[1]) return [[nr, nc], R_GOAL, true]
    return [[nr, nc], stepCost(commanded), false]
  }

  reset(): number {
    this.pos = this.map.start
    this.t = 0
    return this.toState(this.pos)
  }

  sampleRealAction(action: number): number {
    if (this.rng() >= this.pSlip) return action
    const k = Math.floor(this.rng() * 3)
    return [rotate(action, -1), rotate(action, 1), STAY][k]
  }

  step(action: number): StepResult {
    const realAction = this.sampleRealAction(action)
    const [pos, reward, done] = this.transition(this.pos, realAction, action)
    this.pos = pos
    this.t += 1
    const truncated = !done && this.t >= this.maxSteps
    return { state: this.toState(pos), reward, done, truncated, realAction }
  }
}
