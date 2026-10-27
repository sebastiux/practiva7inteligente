import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { LunarLanderSim, heuristic, type Action } from './physics'
import { QPolicy, type PolicyJson } from './policy'

describe('LunarLander (port planck.js)', () => {
  it('reset devuelve una observación de 8 valores cerca del techo y centrada', () => {
    const sim = new LunarLanderSim()
    const obs = sim.reset(1)
    expect(obs).toHaveLength(8)
    expect(Math.abs(obs[0])).toBeLessThan(0.1)
    expect(obs[1]).toBeGreaterThan(1.2)
    expect(obs[6]).toBe(0)
    expect(sim.terrainY).toHaveLength(11)
    expect(sim.terrainY[5]).toBeCloseTo(0.99 * sim.helipadY, 5)
  })

  it('sin motores cae y termina en choque con −100', () => {
    const sim = new LunarLanderSim()
    sim.reset(3)
    let r
    do r = sim.step(0)
    while (!sim.done)
    expect(r.terminated).toBe(true)
    expect(r.crashed).toBe(true)
    expect(r.reward).toBe(-100)
  })

  it('el motor principal frena la caída y cobra 0.3 de combustible', () => {
    const a = new LunarLanderSim()
    const b = new LunarLanderSim()
    a.reset(7)
    b.reset(7)
    for (let k = 0; k < 20; k++) {
      a.step(0)
      b.step(2)
    }
    expect(b.lander.getLinearVelocity().y).toBeGreaterThan(a.lander.getLinearVelocity().y)
    expect(b.fuelUsed).toBeCloseTo(20 * 0.3)
  })

  it('la heurística de Gymnasium aterriza casi siempre', () => {
    const sim = new LunarLanderSim()
    let landed = 0
    for (let ep = 0; ep < 10; ep++) {
      let obs = sim.reset(100 + ep)
      let r
      do {
        r = sim.step(heuristic(obs))
        obs = r.obs
      } while (!sim.done)
      if (r.landed) landed++
    }
    expect(landed).toBeGreaterThanOrEqual(7)
  })
})

// Validación estadística contra Gymnasium (escribe public/data/lunar_validation.json):
//   VALIDATE_LUNAR=1 npx vitest run lunar
const policies = ['dqn', 'dqn_nofuel']
describe.skipIf(!process.env.VALIDATE_LUNAR)('validación contra Python', () => {
  const run = (pick: (obs: number[]) => Action, fuelCost = true, episodes = 100) => {
    const sim = new LunarLanderSim({ fuelCost })
    const rewards: number[] = []
    let landed = 0
    let steps = 0
    for (let ep = 0; ep < episodes; ep++) {
      let obs = sim.reset(9000 + ep)
      let r
      do {
        r = sim.step(pick(obs))
        obs = r.obs
      } while (!sim.done)
      rewards.push(sim.totalReward)
      landed += r.landed ? 1 : 0
      steps += sim.t
    }
    const mean = rewards.reduce((x, y) => x + y, 0) / episodes
    const std = Math.sqrt(rewards.reduce((x, y) => x + (y - mean) ** 2, 0) / episodes)
    return { mean, std, landing_rate: landed / episodes, mean_steps: steps / episodes }
  }
  it('heurística y DQN', { timeout: 600_000 }, () => {
    const out: Record<string, unknown> = { heuristic: run(heuristic) }
    for (const tag of policies) {
      const f = `public/models/${tag}_policy.json`
      if (!existsSync(f)) continue
      const pol = new QPolicy(JSON.parse(readFileSync(f, 'utf8')) as PolicyJson)
      out[tag] = run((o) => pol.act(o), tag !== 'dqn_nofuel')
    }
    const python = JSON.parse(readFileSync('public/data/lunar_baselines.json', 'utf8'))
    writeFileSync('public/data/lunar_validation.json', JSON.stringify({ python, browser: out }, null, 1))
    console.log(JSON.stringify(out, null, 1))
  })
})
