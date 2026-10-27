// Ejecuta episodios del Grid World paso a paso para poder animarlos.
import { AgenteRL, type Algo, type AgentParams } from '../../rl/agent'
import { ACTION_DELTAS, GridWorld, STAY, type MapSpec } from '../../rl/gridworld'

export type Outcome = 'goal' | 'crash' | 'timeout'

export interface StepEvent {
  from: number
  to: number
  intended: number
  real: number
  slipped: boolean
  reward: number
  done: boolean
  outcome?: Outcome
  /** Celda contra la que chocó (puede estar fuera de la malla) */
  hit?: [number, number]
}

export interface RunnerStats {
  episodes: number
  goals: number
  crashes: number
  timeouts: number
  returns: number[]
}

export const emptyStats = (): RunnerStats => ({ episodes: 0, goals: 0, crashes: 0, timeouts: 0, returns: [] })

export class Runner {
  env: GridWorld
  agent: AgenteRL | null
  learn: boolean
  s = 0
  a = 0
  ret = 0
  steps = 0
  active = false
  stats = emptyStats()
  visits: Float64Array

  constructor(
    map: MapSpec,
    opts: { algo?: Algo; params?: AgentParams; q?: number[][]; learn?: boolean; seed?: number; pSlip?: number; human?: boolean },
  ) {
    this.env = new GridWorld(map, { pSlip: opts.pSlip ?? 0.1, seed: opts.seed ?? 1 })
    this.agent = opts.human
      ? null
      : new AgenteRL(this.env.nStates, opts.algo ?? 'qlearning', opts.params ?? { alpha: 0.1, gamma: 0.9, epsilon: 0.1 }, opts.seed ?? 1)
    if (this.agent && opts.q) this.agent.Q = Float64Array.from(opts.q.flat())
    this.learn = opts.learn ?? false
    this.visits = new Float64Array(this.env.nStates)
  }

  start() {
    this.s = this.env.reset()
    this.a = this.agent ? this.agent.act(this.s) : 0
    this.ret = 0
    this.steps = 0
    this.active = true
    this.visits[this.s] += 1
    return this.s
  }

  /** Un paso. Para el jugador humano se pasa la acción elegida. */
  step(humanAction?: number): StepEvent {
    if (!this.active) this.start()
    const intended = this.agent ? this.a : (humanAction ?? 0)
    const from = this.s
    const { state: s2, reward, done, truncated, realAction } = this.env.step(intended)
    let hit: [number, number] | undefined
    if (done && reward < 0 && realAction !== STAY) {
      const [r, c] = this.env.toPos(from)
      const [dr, dc] = ACTION_DELTAS[realAction]
      hit = [r + dr, c + dc]
    }
    if (this.agent) {
      const a2 = this.agent.act(s2)
      if (this.learn) this.agent.update(this.s, this.a, reward, s2, a2, done)
      this.a = a2
    }
    this.s = s2
    this.ret += reward
    this.steps += 1
    if (s2 !== from) this.visits[s2] += 1
    const ev: StepEvent = { from, to: s2, intended, real: realAction, slipped: realAction !== intended, reward, done: done || truncated, hit }
    if (done || truncated) {
      ev.outcome = done && reward > 0 ? 'goal' : done ? 'crash' : 'timeout'
      this.finish(ev.outcome)
    }
    return ev
  }

  private finish(outcome: Outcome) {
    this.active = false
    this.stats.episodes += 1
    if (outcome === 'goal') this.stats.goals += 1
    else if (outcome === 'crash') this.stats.crashes += 1
    else this.stats.timeouts += 1
    this.stats.returns.push(this.ret)
    if (this.stats.returns.length > 500) this.stats.returns.shift()
  }

  /** Entrenamiento rápido sin animación ("pretemporada"). */
  trainFast(episodes: number) {
    if (!this.agent) return
    for (let k = 0; k < episodes; k++) {
      const res = this.agent.runEpisode(this.env, { visits: this.visits })
      this.stats.episodes += 1
      if (res.outcome === 'goal') this.stats.goals += 1
      else if (res.outcome === 'crash') this.stats.crashes += 1
      else this.stats.timeouts += 1
      this.stats.returns.push(res.ret)
      if (this.stats.returns.length > 500) this.stats.returns.shift()
    }
    this.active = false
  }

  meanReturn(last = 100) {
    const r = this.stats.returns.slice(-last)
    return r.length ? r.reduce((a, b) => a + b, 0) / r.length : 0
  }
}
