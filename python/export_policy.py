"""Exporta las redes DQN (.pt) a JSON para correrlas en el navegador y mide
líneas base en Gymnasium para validar el port TypeScript de LunarLander.

Uso:
    python export_policy.py                     # exporta dqn y dqn_nofuel
    python export_policy.py --episodes 100      # episodios por línea base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.box2d.lunar_lander import heuristic

from dqn_lunar import QNetwork, fuel_cost, make_env

ROOT = Path(__file__).resolve().parent
WEB = ROOT.parent / "web" / "public"


def export(tag: str) -> QNetwork | None:
    path = ROOT / "models" / f"{tag}_lunarlander.pt"
    if not path.exists():
        return None
    net = QNetwork()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    layers = []
    for lin in (net.fc1, net.fc2, net.out):
        layers.append({
            "W": np.round(lin.weight.detach().numpy().astype(float), 6).tolist(),  # (salidas, entradas)
            "b": np.round(lin.bias.detach().numpy().astype(float), 6).tolist(),
        })
    out = WEB / "models" / f"{tag}_policy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tag": tag, "architecture": "8-128-128-4", "activation": "relu", "layers": layers},
                              separators=(",", ":")))
    print(f"[{tag}] pesos -> {out} ({out.stat().st_size / 1024:.0f} KB)")
    return net


def rollout(policy, episodes: int, seed0: int, no_fuel_cost=False):
    env, _ = make_env()
    rewards, landed, steps = [], 0, []
    for k in range(episodes):
        s, _ = env.reset(seed=seed0 + k)
        total, n = 0.0, 0
        while True:
            a = policy(env, s)
            s, r, term, trunc, _ = env.step(a)
            total += r + (fuel_cost(a) if no_fuel_cost and not term else 0.0)
            n += 1
            if term or trunc:
                landed += int(term and r == 100)
                break
        rewards.append(total)
        steps.append(n)
    return {"episodes": episodes, "mean": float(np.mean(rewards)), "std": float(np.std(rewards)),
            "landing_rate": landed / episodes, "mean_steps": float(np.mean(steps))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--tags", nargs="+", default=["dqn", "dqn_nofuel"])
    args = ap.parse_args()
    torch.set_num_threads(1)

    baselines = {"gymnasium_version": gym.__version__,
                 "heuristic": rollout(lambda env, s: heuristic(env, s), args.episodes, 50_000)}
    print("heurística:", baselines["heuristic"])
    for tag in args.tags:
        net = export(tag)
        if net is None:
            continue

        @torch.no_grad()
        def pol(env, s, net=net):
            return int(net(torch.from_numpy(np.asarray(s, np.float32))).argmax())

        baselines[tag] = rollout(pol, args.episodes, 50_000, no_fuel_cost=(tag == "dqn_nofuel"))
        print(f"{tag}:", baselines[tag])
    (WEB / "data" / "lunar_baselines.json").write_text(json.dumps(baselines, indent=1))


if __name__ == "__main__":
    main()
