import { useEffect, useState } from 'react'

export const BASE = import.meta.env.BASE_URL

export interface EvalStats {
  mean_return: number
  std_return: number
  success_rate: number
  crash_rate: number
  timeout_rate: number
  gap_rate: number
  mean_steps: number
}

export interface AlgoResult {
  reward_mean: number[]
  reward_std: number[]
  ma100_mean: number[]
  ma100_std: number[]
  success_ma100: number[]
  crash_ma100: number[]
  final_ma100: { mean: number; std: number; per_seed: number[] }
  train_mean_return: number
  visits: { eval_eps: number[][]; eval_greedy: number[][]; train_last: number[][] }
  eval_eps: EvalStats
  eval_greedy: EvalStats
  eval_per_seed: { seed: number; eps: EvalStats; greedy: EvalStats }[]
  q_table_seed0: number[][]
  policy_seed0: number[][]
  v_seed0: number[][]
  v_mean: number[][]
}

export interface TabularData {
  config: {
    episodes: number
    alpha: number
    gamma: number
    epsilon: number
    p_slip: number
    last_n: number
    eval_episodes: number
    seeds: number[]
    map: string[]
    start: [number, number]
    goal: [number, number]
    gap_cells: [number, number][]
  }
  algos: { qlearning: AlgoResult; sarsa: AlgoResult }
}

export interface DqnEpisode {
  reward: number
  reward_with_fuel: number
  steps: number
  main_engine: number
  side_engine: number
  landed: boolean
  timeout: boolean
}

export interface DqnData {
  env_id: string
  gymnasium_version: string
  torch_version: string
  config: Record<string, number | string | boolean> & { n_params: number; eps_decay: number }
  solved_at: number | null
  episodes_run: number
  train_time_s: number
  history: {
    reward: number[]
    reward_with_fuel: number[]
    epsilon: number[]
    steps: number[]
    main_engine: number[]
    side_engine: number[]
    landed: boolean[]
    ma100: number[]
  }
  eval: {
    episodes: DqnEpisode[]
    mean: number
    std: number
    mean_with_fuel: number
    std_with_fuel: number
    landing_rate: number
    mean_steps: number
    mean_main_engine: number
    mean_side_engine: number
  }
  video: { reward: number; seed: number; frames: number } | null
}

const cache = new Map<string, Promise<unknown>>()

/** Los JSON pesan cientos de KB: se piden después de la carga inicial para no competir con la primera pintura. */
const afterLoad = new Promise<void>((resolve) => {
  const go = () => setTimeout(resolve, 50)
  if (document.readyState === 'complete') go()
  else window.addEventListener('load', go, { once: true })
})

export function loadJson<T>(name: string): Promise<T> {
  if (!cache.has(name)) {
    cache.set(
      name,
      afterLoad.then(() => fetch(`${BASE}data/${name}`)).then((r) => {
        if (!r.ok) throw new Error(`No se pudo cargar ${name}`)
        return r.json()
      }),
    )
  }
  return cache.get(name) as Promise<T>
}

export function useJson<T>(name: string): T | null {
  const [data, setData] = useState<T | null>(null)
  useEffect(() => {
    let alive = true
    loadJson<T>(name).then((d) => alive && setData(d), console.error)
    return () => {
      alive = false
    }
  }, [name])
  return data
}

export const fmt = (x: number, d = 1) =>
  x.toLocaleString('es-MX', { minimumFractionDigits: d, maximumFractionDigits: d })
export const pct = (x: number, d = 0) => `${fmt(100 * x, d)} %`
