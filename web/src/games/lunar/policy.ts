// Red DQN (8-128-128-4, ReLU) exportada desde PyTorch, evaluada en el navegador.
import type { Action } from './physics'

export interface PolicyJson {
  tag: string
  layers: { W: number[][]; b: number[] }[]
}

export class QPolicy {
  private layers: { W: Float32Array; b: Float32Array; nOut: number; nIn: number }[]

  constructor(json: PolicyJson) {
    this.layers = json.layers.map(({ W, b }) => ({
      W: Float32Array.from(W.flat()),
      b: Float32Array.from(b),
      nOut: W.length,
      nIn: W[0].length,
    }))
  }

  qValues(obs: number[]): number[] {
    let x = Float32Array.from(obs)
    this.layers.forEach((L, li) => {
      const y = new Float32Array(L.nOut)
      for (let o = 0; o < L.nOut; o++) {
        let s = L.b[o]
        const row = o * L.nIn
        for (let i = 0; i < L.nIn; i++) s += L.W[row + i] * x[i]
        y[o] = li < this.layers.length - 1 ? Math.max(0, s) : s
      }
      x = y
    })
    return Array.from(x)
  }

  act(obs: number[]): Action {
    const q = this.qValues(obs)
    let best = 0
    for (let a = 1; a < q.length; a++) if (q[a] > q[best]) best = a
    return best as Action
  }
}
