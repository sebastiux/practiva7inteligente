# Práctica 7: SARSA, Q-learning y Deep Q-Learning

Sitio web que resuelve y explica paso a paso la práctica *"Sarsa, Q-learning y Deep Q-learning"*
(Drs. Alexandro López González y Julio Caballero Mora, 22 de octubre de 2026).

**Sitio:** https://sebastiux.github.io/practiva7inteligente/

```
python/
  gridworld.py              entorno 20x20, 8 acciones, resbalón p=0.1 ("El Paso del Contrabandista")
  agents.py                 AgenteRL configurable: "sarsa" | "qlearning"
  train_tabular.py          entrena ambos (5 semillas) y exporta web/public/data/tabular.json
  dqn_lunar.py              DQN (PyTorch) en LunarLander, exporta JSON + video + modelo .pt
  summarize_dqn_seeds.py    resume corridas DQN con varias semillas
  models/*.pt               pesos entrenados (state_dict)
  tests/                    pytest del entorno (resbalón, recompensas, colisiones)
web/                        Vite + React + TypeScript + Tailwind + Plotly.js + KaTeX
  src/rl/                   port TypeScript del entorno y del agente (+ pruebas Vitest)
  public/data/*.json        resultados precomputados
  public/media/*.mp4|gif    aterrizajes greedy
```

## Reproducir

```bash
# Python (3.10+)
cd python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q tests

python train_tabular.py                              # ~10 s con 4 núcleos
python dqn_lunar.py                                  # ~2-3 min en CPU, se detiene con MA100 >= 200
python dqn_lunar.py --no-fuel-cost --tag dqn_nofuel  # variante sin costo de combustible
for s in 1 2 3; do python dqn_lunar.py --seed $s --tag dqn_seed$s --no-video; done
python summarize_dqn_seeds.py                        # escribe dqn_seeds.json (puedes borrar los dqn_seedN.json)

# Web
cd ../web
npm install
npm run dev        # http://localhost:5173/practiva7inteligente/
npm test           # Vitest del port TS
npm run build
```

> `pip install torch` desde PyPI trae la build CUDA (varios GB). En una máquina sin GPU puedes usar
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Decisiones de diseño

- **Mapa.** Barrera vertical en la columna 10 (filas 0-11) con una brecha de una celda en (6, 10).
  A = (4, 6), B = (7, 15). El atajo cruza la brecha en diagonal y el rodeo pasa por el espacio abierto
  de las filas 12-19.
- **Resbalón.** Con probabilidad 0.1 el movimiento real es, de forma uniforme, la dirección girada
  −45°, la girada +45° o quedarse quieto.
- **Colisión diagonal.** Sólo choca si la celda destino es obstáculo o queda fuera de la malla. Una
  diagonal puede rozar dos esquinas.
- **Costo de paso.** Se cobra según la acción comandada, aunque el robot patine. Los episodios se
  truncan a 400 pasos; el truncamiento no se trata como estado terminal.
- **Mapa de calor.** Se mide en una evaluación con ε = 0.1 (1,000 partidos por semilla) para que el
  riesgo sea visible. También se exportan las visitas greedy y las de los últimos 1,000 partidos de
  entrenamiento.
- **DQN.** Red 8-128-128-4 con experience replay (100k, lote 64, aprendizaje cada 4 pasos), target
  network suave (τ = 1e-3), Adam (α = 5e-4), γ = 0.99 y ε 1.0 → 0.01 (×0.995 por episodio). Usa
  `LunarLander-v3`, porque `v2` no existe en Gymnasium 1.x.
