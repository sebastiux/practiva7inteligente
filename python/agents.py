"""AgenteRL tabular configurable: "sarsa" (on-policy) | "qlearning" (off-policy)."""
from __future__ import annotations

import numpy as np


class AgenteRL:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        algo: str = "qlearning",
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        seed: int | None = None,
        q_init: float = 0.0,
    ):
        if algo not in ("sarsa", "qlearning"):
            raise ValueError("algo debe ser 'sarsa' o 'qlearning'")
        self.algo = algo
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.Q = np.full((n_states, n_actions), float(q_init))
        self.rng = np.random.default_rng(seed)

    def greedy(self, s: int) -> int:
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())  # desempate aleatorio
        return int(best[0] if len(best) == 1 else self.rng.choice(best))

    def act(self, s: int, epsilon: float | None = None) -> int:
        eps = self.epsilon if epsilon is None else epsilon
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(s)

    def target(self, r: float, s_next: int, a_next: int | None, done: bool) -> float:
        if done:
            return r
        if self.algo == "qlearning":
            # Ec. (1): máx_a' Q(s', a') -> independiente de la acción que se tomará
            return r + self.gamma * self.Q[s_next].max()
        # Ec. (2): Q(s', a') con la a' que la política epsilon-greedy SÍ eligió
        return r + self.gamma * self.Q[s_next, a_next]

    def update(self, s, a, r, s_next, a_next, done) -> float:
        td = self.target(r, s_next, a_next, done) - self.Q[s, a]
        self.Q[s, a] += self.alpha * td
        return td

    def run_episode(self, env, learn: bool = True, epsilon: float | None = None, visits=None):
        """Juega un episodio. Devuelve (retorno, pasos, resultado)."""
        s = env.reset()
        a = self.act(s, epsilon)
        total, steps = 0.0, 0
        if visits is not None:
            visits[s] += 1
        while True:
            s2, r, done, trunc, _ = env.step(a)
            total += r
            steps += 1
            a2 = self.act(s2, epsilon)
            if learn:
                # Nota: en un truncamiento por tiempo NO se trata s2 como terminal.
                self.update(s, a, r, s2, a2, done)
            if visits is not None and s2 != s:
                visits[s2] += 1
            if done or trunc:
                outcome = "goal" if r > 0 else ("crash" if done else "timeout")
                return total, steps, outcome
            s, a = s2, a2

    def policy(self) -> np.ndarray:
        return self.Q.argmax(axis=1)
