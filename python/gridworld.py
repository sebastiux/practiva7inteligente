"""Grid World 8-DOF "El Paso del Contrabandista".

Entorno tabular de la Práctica 7 (SARSA vs Q-learning).

Convenciones
------------
* Coordenadas (fila, columna); la fila 0 está arriba (Norte).
* Acciones (índice -> nombre): 0 N, 1 S, 2 E, 3 O, 4 NE, 5 NO, 6 SE, 7 SO.
* Resbalamiento (p_slip): con probabilidad 1 - p_slip se ejecuta la acción
  pedida. Con probabilidad p_slip el robot "patina": se elige de forma
  uniforme entre (a) la dirección rotada -45°, (b) la dirección rotada +45°
  (ambas caen en celdas adyacentes a la celda deseada) o (c) quedarse en su
  lugar.
* Regla de colisión diagonal: un movimiento (cardinal o diagonal) choca
  únicamente si la celda DESTINO es un obstáculo o queda fuera de la malla.
  Un movimiento diagonal puede pasar "rozando" las esquinas de dos obstáculos
  (no se revisan las celdas ortogonales intermedias). Esto es lo que permite
  cruzar la brecha diagonal y es justo lo que la vuelve peligrosa.
* Recompensas: meta +100 (termina), colisión -100 (termina, el robot no se
  mueve), paso cardinal -1.0, paso diagonal -1.41. El costo de paso se cobra
  según la acción COMANDADA (energía gastada) aunque el robot patine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ACTION_NAMES = ["N", "S", "E", "O", "NE", "NO", "SE", "SO"]
ACTION_DELTAS = np.array(
    [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)], dtype=int
)
# Orden en la rosa de los vientos (sentido horario) para rotar ±45°.
COMPASS = [0, 4, 2, 6, 1, 7, 3, 5]  # N, NE, E, SE, S, SO, O, NO
COMPASS_POS = {a: i for i, a in enumerate(COMPASS)}
STAY = -1  # "acción" interna: el robot se queda en su lugar al patinar

R_GOAL = 100.0
R_CRASH = -100.0
COST_CARDINAL = -1.0
COST_DIAGONAL = -1.41

# El Paso del Contrabandista (20x20).
#   '#' obstáculo   'A' inicio   'B' meta   '.' libre
# Barrera vertical (columna 10) desde el borde norte hasta la fila 11.
# La brecha es UNA celda, (6, 10): A está arriba-izquierda y B abajo-derecha,
# así que el atajo cruza la barrera en diagonal rozando las esquinas (corto y
# peligroso). Las filas 12-19 son el espacio abierto para rodear (largo y seguro).
SMUGGLER_MAP = [
    "..........#.........",
    "..........#.........",
    "..........#.........",
    "..........#.........",
    "......A...#.........",
    "..........#.........",
    "....................",
    "..........#....B....",
    "..........#.........",
    "..........#.........",
    "..........#.........",
    "..........#.........",
    "....................",
    "....................",
    "....................",
    "....................",
    "....................",
    "....................",
    "....................",
    "....................",
]


def rotate(action: int, steps: int) -> int:
    """Rota una acción `steps` pasos de 45° (positivo = horario)."""
    return COMPASS[(COMPASS_POS[action] + steps) % 8]


def step_cost(action: int) -> float:
    return COST_CARDINAL if action < 4 else COST_DIAGONAL


def parse_map(rows: list[str]):
    grid = np.array([[c == "#" for c in r] for r in rows], dtype=bool)
    start = goal = None
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            if c == "A":
                start = (i, j)
            elif c == "B":
                goal = (i, j)
    if start is None or goal is None:
        raise ValueError("El mapa necesita una celda 'A' y una 'B'")
    return grid, start, goal


@dataclass
class GridWorld:
    rows: list[str] = field(default_factory=lambda: list(SMUGGLER_MAP))
    p_slip: float = 0.1
    max_steps: int = 400
    seed: int | None = None

    def __post_init__(self):
        self.obstacles, self.start, self.goal = parse_map(self.rows)
        self.n_rows, self.n_cols = self.obstacles.shape
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 8
        self.rng = np.random.default_rng(self.seed)
        self.pos = self.start
        self.t = 0

    # -- utilidades ---------------------------------------------------------
    def to_state(self, pos) -> int:
        return pos[0] * self.n_cols + pos[1]

    def to_pos(self, s: int):
        return divmod(s, self.n_cols)

    def blocked(self, r: int, c: int) -> bool:
        if r < 0 or c < 0 or r >= self.n_rows or c >= self.n_cols:
            return True
        return bool(self.obstacles[r, c])

    def outcomes(self, action: int):
        """Distribución de movimientos reales: lista de (prob, acción_real)."""
        p = self.p_slip
        out = [(1.0 - p, action)]
        if p > 0:
            out += [(p / 3, rotate(action, -1)), (p / 3, rotate(action, 1)), (p / 3, STAY)]
        return out

    def transition(self, pos, real_action: int, commanded: int):
        """Aplica un movimiento real. Devuelve (nueva_pos, recompensa, terminal)."""
        if real_action == STAY:
            return pos, step_cost(commanded), False
        dr, dc = ACTION_DELTAS[real_action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if self.blocked(nr, nc):
            return pos, R_CRASH, True
        if (nr, nc) == self.goal:
            return (nr, nc), R_GOAL, True
        return (nr, nc), step_cost(commanded), False

    # -- API tipo Gym -------------------------------------------------------
    def reset(self) -> int:
        self.pos = self.start
        self.t = 0
        return self.to_state(self.pos)

    def sample_real_action(self, action: int) -> int:
        if self.rng.random() >= self.p_slip:
            return action
        k = self.rng.integers(3)
        return (rotate(action, -1), rotate(action, 1), STAY)[k]

    def step(self, action: int):
        real = self.sample_real_action(action)
        self.pos, reward, done = self.transition(self.pos, real, action)
        self.t += 1
        truncated = (not done) and self.t >= self.max_steps
        return self.to_state(self.pos), reward, done, truncated, {"real_action": real}
