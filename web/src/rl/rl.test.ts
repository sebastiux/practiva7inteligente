import { describe, expect, it } from 'vitest'
import { AgenteRL } from './agent'
import {
  COST_CARDINAL, COST_DIAGONAL, GridWorld, R_CRASH, R_GOAL, SMUGGLER_MAP, STAY, parseMap, rotate,
} from './gridworld'

const map = parseMap(SMUGGLER_MAP)
const N = 0, E = 2, NE = 4, SE = 6

describe('GridWorld (port TS)', () => {
  it('parsea el mapa del Paso del Contrabandista', () => {
    expect(map.rows).toBe(20)
    expect(map.cols).toBe(20)
    expect(map.start).toEqual([4, 6])
    expect(map.goal).toEqual([7, 15])
    expect(map.obstacles[6 * 20 + 10]).toBe(false) // la brecha
    expect(map.obstacles[5 * 20 + 10]).toBe(true)
  })

  it('rota acciones ±45°', () => {
    expect(rotate(N, 1)).toBe(NE)
    expect(rotate(N, -1)).toBe(5) // NO
    expect(rotate(E, 1)).toBe(SE)
  })

  it('cobra -1 en cardinales y -1.41 en diagonales', () => {
    const env = new GridWorld(map, { pSlip: 0 })
    expect(env.transition([10, 3], E, E)[1]).toBe(COST_CARDINAL)
    expect(env.transition([10, 3], SE, SE)[1]).toBe(COST_DIAGONAL)
  })

  it('choque con pared u obstáculo: -100, termina y no se mueve', () => {
    const env = new GridWorld(map, { pSlip: 0 })
    expect(env.transition([0, 0], N, N)).toEqual([[0, 0], R_CRASH, true])
    expect(env.transition([5, 9], E, E)).toEqual([[5, 9], R_CRASH, true])
  })

  it('la diagonal puede rozar esquinas para cruzar la brecha', () => {
    const env = new GridWorld(map, { pSlip: 0 })
    const [p, r, done] = env.transition([5, 9], SE, SE)
    expect(p).toEqual([6, 10])
    expect(done).toBe(false)
    expect(r).toBe(COST_DIAGONAL)
    expect(env.transition([6, 10], SE, SE)[0]).toEqual([7, 11])
  })

  it('llegar a la meta da +100 y termina', () => {
    const env = new GridWorld(map, { pSlip: 0 })
    expect(env.transition([7, 14], E, E)).toEqual([[7, 15], R_GOAL, true])
  })

  it('resbala con frecuencia ≈ p_slip a celdas adyacentes o se queda', () => {
    const env = new GridWorld(map, { pSlip: 0.1, seed: 3 })
    const counts = new Map<number, number>()
    const n = 30000
    for (let i = 0; i < n; i++) {
      const a = env.sampleRealAction(E)
      counts.set(a, (counts.get(a) ?? 0) + 1)
    }
    expect([...counts.keys()].sort()).toEqual([STAY, E, NE, SE].sort())
    expect((counts.get(E) ?? 0) / n).toBeCloseTo(0.9, 1)
    expect((counts.get(STAY) ?? 0) / n).toBeCloseTo(0.1 / 3, 1)
  })
})

describe('AgenteRL (port TS)', () => {
  it('Q-learning usa máx y SARSA usa a′ en el blanco', () => {
    const q = new AgenteRL(400, 'qlearning', { alpha: 0.1, gamma: 0.9, epsilon: 0.1 })
    const s = new AgenteRL(400, 'sarsa', { alpha: 0.1, gamma: 0.9, epsilon: 0.1 })
    for (const ag of [q, s]) {
      ag.Q[5 * 8 + 0] = 10
      ag.Q[5 * 8 + 1] = -20
    }
    expect(q.target(-1, 5, 1, false)).toBeCloseTo(-1 + 0.9 * 10)
    expect(s.target(-1, 5, 1, false)).toBeCloseTo(-1 + 0.9 * -20)
    expect(q.target(100, 5, 1, true)).toBe(100)
  })

  it('actualiza Q(s,a) ← Q + α (blanco − Q)', () => {
    const ag = new AgenteRL(400, 'qlearning', { alpha: 0.1, gamma: 0.9, epsilon: 0 })
    ag.Q[2 * 8 + 3] = 4
    const td = ag.update(1, 0, -1, 2, 0, false)
    expect(td).toBeCloseTo(-1 + 0.9 * 4 - 0)
    expect(ag.Q[1 * 8 + 0]).toBeCloseTo(0.1 * td)
  })

  it('aprende a llegar a la meta en el mapa del contrabandista', () => {
    const env = new GridWorld(map, { pSlip: 0.1, seed: 1 })
    const ag = new AgenteRL(env.nStates, 'sarsa', { alpha: 0.1, gamma: 0.9, epsilon: 0.1 }, 1)
    for (let i = 0; i < 3000; i++) ag.runEpisode(env)
    let goals = 0
    for (let i = 0; i < 100; i++) if (ag.runEpisode(env, { learn: false, epsilon: 0 }).outcome === 'goal') goals++
    expect(goals).toBeGreaterThan(80)
  })
})
