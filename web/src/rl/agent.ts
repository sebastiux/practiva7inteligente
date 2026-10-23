// Port a TypeScript de python/agents.py.
import { mulberry32, type GridWorld } from './gridworld'

export type Algo = 'sarsa' | 'qlearning'

export interface AgentParams {
  alpha: number
  gamma: number
  epsilon: number
}

export type Outcome = 'goal' | 'crash' | 'timeout'

export interface EpisodeResult {
  ret: number
  steps: number
  outcome: Outcome
  path: number[]
}

export class AgenteRL {
  algo: Algo
  params: AgentParams
  nActions = 8
  Q: Float64Array
  rng: () => number

  constructor(nStates: number, algo: Algo, params: AgentParams, seed = 0) {
    this.algo = algo
    this.params = { ...params }
    this.Q = new Float64Array(nStates * this.nActions)
    this.rng = mulberry32(seed + 7919)
  }

  q(s: number, a: number) {
    return this.Q[s * this.nActions + a]
  }

  maxQ(s: number) {
    let m = -Infinity
    for (let a = 0; a < this.nActions; a++) m = Math.max(m, this.q(s, a))
    return m
  }

  greedy(s: number): number {
    const m = this.maxQ(s)
    const best: number[] = []
    for (let a = 0; a < this.nActions; a++) if (this.q(s, a) === m) best.push(a)
    return best.length === 1 ? best[0] : best[Math.floor(this.rng() * best.length)]
  }

  act(s: number, epsilon = this.params.epsilon): number {
    if (this.rng() < epsilon) return Math.floor(this.rng() * this.nActions)
    return this.greedy(s)
  }

  /** Blanco TD: ec. (1) para Q-learning (máx) y ec. (2) para SARSA (a' elegida). */
  target(r: number, s2: number, a2: number, done: boolean): number {
    if (done) return r
    if (this.algo === 'qlearning') return r + this.params.gamma * this.maxQ(s2)
    return r + this.params.gamma * this.q(s2, a2)
  }

  update(s: number, a: number, r: number, s2: number, a2: number, done: boolean): number {
    const i = s * this.nActions + a
    const td = this.target(r, s2, a2, done) - this.Q[i]
    this.Q[i] += this.params.alpha * td
    return td
  }

  runEpisode(env: GridWorld, opts: { learn?: boolean; epsilon?: number; visits?: Float64Array } = {}): EpisodeResult {
    const learn = opts.learn ?? true
    const eps = opts.epsilon ?? this.params.epsilon
    let s = env.reset()
    let a = this.act(s, eps)
    let ret = 0
    let steps = 0
    const path = [s]
    if (opts.visits) opts.visits[s] += 1
    for (;;) {
      const { state: s2, reward: r, done, truncated } = env.step(a)
      ret += r
      steps += 1
      const a2 = this.act(s2, eps)
      if (learn) this.update(s, a, r, s2, a2, done)
      if (s2 !== s) {
        path.push(s2)
        if (opts.visits) opts.visits[s2] += 1
      }
      if (done || truncated) {
        const outcome: Outcome = r > 0 && done ? 'goal' : done ? 'crash' : 'timeout'
        return { ret, steps, outcome, path }
      }
      s = s2
      a = a2
    }
  }

  policy(): number[] {
    const n = this.Q.length / this.nActions
    return Array.from({ length: n }, (_, s) => this.greedy(s))
  }
}
