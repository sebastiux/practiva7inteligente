import { describe, expect, it } from 'vitest'
import { SMUGGLER_MAP, parseMap } from '../../rl/gridworld'
import { Runner } from './runner'

const map = parseMap(SMUGGLER_MAP)

describe('Runner del juego 3D', () => {
  it('un humano que va al norte choca con la pared superior', () => {
    const r = new Runner(map, { human: true, pSlip: 0 })
    r.start()
    let ev
    for (let k = 0; k < 10 && r.active; k++) ev = r.step(0)
    expect(ev!.outcome).toBe('crash')
    expect(ev!.hit).toEqual([-1, 6])
    expect(r.stats.crashes).toBe(1)
  })

  it('marca resbalones cuando la acción real difiere', () => {
    const r = new Runner(map, { human: true, pSlip: 1, seed: 4 })
    r.start()
    const ev = r.step(6)
    expect(ev.slipped).toBe(true)
  })

  it('aprende en vivo y la pretemporada acumula estadísticas', () => {
    const r = new Runner(map, { algo: 'sarsa', learn: true, seed: 2 })
    r.trainFast(2000)
    expect(r.stats.episodes).toBe(2000)
    expect(r.meanReturn()).toBeGreaterThan(0)
    let n = 0
    r.start()
    while (r.active && n++ < 500) r.step()
    expect(r.stats.episodes).toBe(2001)
  })
})
