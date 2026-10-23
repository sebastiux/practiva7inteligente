"""Entrena SARSA y Q-learning en "El Paso del Contrabandista" y exporta JSON.

Uso:
    python train_tabular.py --episodes 10000 --seeds 0 1 2 3 4 \
        --out ../web/public/data/tabular.json
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from agents import AgenteRL
from gridworld import ACTION_NAMES, SMUGGLER_MAP, GridWorld

ALGOS = ("qlearning", "sarsa")
# Celda de la brecha.
GAP_CELLS = [(6, 10)]


def moving_average(x: np.ndarray, w: int = 100) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        lo = max(0, i + 1 - w)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def evaluate(agent, env, episodes, epsilon):
    visits = np.zeros(env.n_states)
    gap_states = {env.to_state(p) for p in GAP_CELLS}
    stats = {"goal": 0, "crash": 0, "timeout": 0}
    returns, gap_uses, steps_list = [], 0, []
    for _ in range(episodes):
        ep_visits = np.zeros(env.n_states)
        ret, steps, outcome = agent.run_episode(env, learn=False, epsilon=epsilon, visits=ep_visits)
        visits += ep_visits
        stats[outcome] += 1
        returns.append(ret)
        steps_list.append(steps)
        if any(ep_visits[s] > 0 for s in gap_states):
            gap_uses += 1
    return visits, {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": stats["goal"] / episodes,
        "crash_rate": stats["crash"] / episodes,
        "timeout_rate": stats["timeout"] / episodes,
        "gap_rate": gap_uses / episodes,
        "mean_steps": float(np.mean(steps_list)),
    }


def run(job):
    algo, seed, cfg = job
    env = GridWorld(p_slip=cfg["p_slip"], seed=seed)
    agent = AgenteRL(env.n_states, env.n_actions, algo, cfg["alpha"], cfg["gamma"], cfg["epsilon"], seed=seed)
    n = cfg["episodes"]
    returns = np.zeros(n)
    outcomes = {"goal": np.zeros(n, bool), "crash": np.zeros(n, bool)}
    last_visits = np.zeros(env.n_states)
    last_n = cfg["last_n"]
    t0 = time.time()
    for ep in range(n):
        vis = last_visits if ep >= n - last_n else None
        ret, _, outcome = agent.run_episode(env, learn=True, visits=vis)
        returns[ep] = ret
        if outcome in outcomes:
            outcomes[outcome][ep] = True
    train_time = time.time() - t0
    eval_env = GridWorld(p_slip=cfg["p_slip"], seed=10_000 + seed)
    vis_eps, ev_eps = evaluate(agent, eval_env, cfg["eval_episodes"], cfg["epsilon"])
    vis_greedy, ev_greedy = evaluate(agent, eval_env, cfg["eval_episodes"], 0.0)
    print(f"{algo:9s} seed={seed} train={train_time:.1f}s  "
          f"eval(eps)={ev_eps['mean_return']:.1f} gap={ev_eps['gap_rate']:.2f}  "
          f"eval(greedy)={ev_greedy['mean_return']:.1f} gap={ev_greedy['gap_rate']:.2f}", flush=True)
    return {
        "algo": algo,
        "seed": seed,
        "returns": returns,
        "goal": outcomes["goal"],
        "crash": outcomes["crash"],
        "visits_train_last": last_visits,
        "visits_eval_eps": vis_eps,
        "visits_eval_greedy": vis_greedy,
        "eval_eps": ev_eps,
        "eval_greedy": ev_greedy,
        "Q": agent.Q,
    }


def r(x, d=2):
    return np.round(np.asarray(x, dtype=float), d).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--p-slip", type=float, default=0.1)
    ap.add_argument("--last-n", type=int, default=1000)
    ap.add_argument("--eval-episodes", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "web/public/data/tabular.json"))
    args = ap.parse_args()
    cfg = {
        "episodes": args.episodes, "alpha": args.alpha, "gamma": args.gamma,
        "epsilon": args.epsilon, "p_slip": args.p_slip, "last_n": args.last_n,
        "eval_episodes": args.eval_episodes,
    }
    jobs = [(a, s, cfg) for a in ALGOS for s in args.seeds]
    with Pool(args.workers) as pool:
        results = pool.map(run, jobs)

    env = GridWorld()
    out = {
        "config": {**cfg, "seeds": args.seeds, "map": SMUGGLER_MAP, "start": env.start,
                   "goal": env.goal, "gap_cells": GAP_CELLS, "actions": ACTION_NAMES,
                   "rewards": {"goal": 100, "crash": -100, "cardinal": -1.0, "diagonal": -1.41}},
        "algos": {},
    }
    for algo in ALGOS:
        rs = [x for x in results if x["algo"] == algo]
        R = np.stack([x["returns"] for x in rs])
        MA = np.stack([moving_average(x) for x in R])
        first = next(x for x in rs if x["seed"] == args.seeds[0])
        Q = first["Q"]
        mean_q = np.mean([x["Q"] for x in rs], axis=0)
        last100 = R[:, -100:].mean(axis=1)
        out["algos"][algo] = {
            "reward_mean": r(R.mean(0), 1),
            "reward_std": r(R.std(0), 1),
            "ma100_mean": r(MA.mean(0)),
            "ma100_std": r(MA.std(0)),
            "success_ma100": r(moving_average(np.stack([x["goal"] for x in rs]).mean(0)), 3),
            "crash_ma100": r(moving_average(np.stack([x["crash"] for x in rs]).mean(0)), 3),
            "final_ma100": {"mean": float(last100.mean()), "std": float(last100.std()),
                             "per_seed": r(last100)},
            "train_mean_return": float(R.mean()),
            "visits": {
                "eval_eps": r(sum(x["visits_eval_eps"] for x in rs).reshape(env.n_rows, env.n_cols), 0),
                "eval_greedy": r(sum(x["visits_eval_greedy"] for x in rs).reshape(env.n_rows, env.n_cols), 0),
                "train_last": r(sum(x["visits_train_last"] for x in rs).reshape(env.n_rows, env.n_cols), 0),
            },
            "eval_eps": {k: float(np.mean([x["eval_eps"][k] for x in rs])) for k in rs[0]["eval_eps"]},
            "eval_greedy": {k: float(np.mean([x["eval_greedy"][k] for x in rs])) for k in rs[0]["eval_greedy"]},
            "eval_per_seed": [{"seed": x["seed"], "eps": x["eval_eps"], "greedy": x["eval_greedy"]} for x in rs],
            "q_table_seed0": r(Q, 2),
            "policy_seed0": Q.argmax(1).reshape(env.n_rows, env.n_cols).tolist(),
            "v_seed0": r(Q.max(1).reshape(env.n_rows, env.n_cols), 2),
            "v_mean": r(mean_q.max(1).reshape(env.n_rows, env.n_cols), 2),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("Escrito", args.out)
    for algo in ALGOS:
        a = out["algos"][algo]
        print(f"{algo}: MA100 final={a['final_ma100']['mean']:.2f}±{a['final_ma100']['std']:.2f} "
              f"| eval eps: {a['eval_eps']['mean_return']:.1f} gap={a['eval_eps']['gap_rate']:.2f} "
              f"crash={a['eval_eps']['crash_rate']:.3f} | greedy: {a['eval_greedy']['mean_return']:.1f} "
              f"gap={a['eval_greedy']['gap_rate']:.2f}")


if __name__ == "__main__":
    main()
