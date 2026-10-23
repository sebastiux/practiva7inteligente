"""Resume corridas DQN con varias semillas (dqn.json + dqn_seed*.json) en dqn_seeds.json.

Uso (después de `python dqn_lunar.py --seed N --tag dqn_seedN --no-video`):
    python summarize_dqn_seeds.py
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "web" / "public" / "data"

rows = []
for path in [DATA / "dqn.json", *sorted(DATA.glob("dqn_seed*.json"))]:
    d = json.loads(path.read_text())
    e = d["eval"]
    rows.append({"seed": d["config"]["seed"], "solved_at": d["solved_at"], "eval_mean": e["mean"],
                 "eval_std": e["std"], "landing_rate": e["landing_rate"], "mean_steps": e["mean_steps"],
                 "timeouts": sum(x["timeout"] for x in e["episodes"]), "train_time_s": d["train_time_s"]})
(DATA / "dqn_seeds.json").write_text(json.dumps(rows, indent=1))
for r in rows:
    print(r)
