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
  export_policy.py          exporta las redes a JSON para el navegador + líneas base en Gymnasium
  models/*.pt               pesos entrenados (state_dict)
  tests/                    pytest del entorno (resbalón, recompensas, colisiones)
web/                        Vite + React + TypeScript + Tailwind + Plotly.js + KaTeX
  src/rl/                   port TypeScript del entorno y del agente (+ pruebas Vitest)
  src/games/grid3d/         Juego 1: El Paso del Contrabandista 3D (three.js / React Three Fiber)
  src/games/lunar/          Juego 2: Alunizaje DQN 3D (física LunarLander portada a planck.js)
  public/models/*.json      pesos de las redes DQN para el autopiloto en el navegador
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
python export_policy.py                              # pesos -> web/public/models + líneas base (100 episodios)

# Web
cd ../web
npm install
npm run dev        # http://localhost:5173/practiva7inteligente/
npm test           # Vitest del port TS
VALIDATE_LUNAR=1 npx vitest run lunar   # compara el port de LunarLander contra Gymnasium
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

## Juegos 3D

- **El Paso del Contrabandista 3D** tiene tres modos:
  - **Duelo:** SARSA y Q-learning juegan a la vez con sus tablas Q finales de Python.
  - **Entrena en vivo:** pretemporada, turbo, mapa de calor en el pasto y editor de cancha.
  - **Juega tú:** teclado o D-pad, con el mismo resbalón que el agente.
- **Alunizaje DQN.** La física de `LunarLander` está portada a TypeScript con planck.js (Box2D), con las mismas
  constantes, la misma recompensa y la misma observación. Así la red de PyTorch, exportada a JSON, pilota en el
  navegador a 50 Hz y muestra sus Q-values en vivo. También se puede pilotar a mano o ver la heurística de
  Gymnasium.
- **Validación del port** (100 episodios por lado; ver `web/public/data/lunar_validation.json`):
  - Heurística: 252.8 en Python contra 255.3 en el navegador.
  - DQN: 193.3 contra 196.4, con 77 % y 76 % de aterrizajes.
  - DQN sin costo: 229.8 contra 219.0.
