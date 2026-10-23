import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import AgenteRL  # noqa: E402
from gridworld import (  # noqa: E402
    COST_CARDINAL, COST_DIAGONAL, R_CRASH, R_GOAL, STAY, GridWorld, rotate,
)

N, S, E, O, NE, NO, SE, SO = range(8)


@pytest.fixture
def env():
    return GridWorld(p_slip=0.0, seed=0)


def test_map(env):
    assert env.obstacles.shape == (20, 20)
    assert env.start == (4, 6) and env.goal == (7, 15)
    assert not env.obstacles[6, 10]  # brecha de 1 celda
    assert env.obstacles[5, 10] and env.obstacles[7, 10]
    assert not env.obstacles[12:, :].any()  # espacio abierto para rodear


def test_rotation():
    assert rotate(N, 1) == NE and rotate(N, -1) == NO
    assert rotate(SO, 1) == O and rotate(E, -1) == NE


def test_step_costs(env):
    assert env.transition((10, 3), E, E)[1] == COST_CARDINAL
    assert env.transition((10, 3), SE, SE)[1] == COST_DIAGONAL
    # Al patinar y quedarse quieto se cobra la acción comandada.
    assert env.transition((10, 3), STAY, SE) == ((10, 3), COST_DIAGONAL, False)


def test_wall_and_border_collisions(env):
    assert env.transition((0, 0), N, N) == ((0, 0), R_CRASH, True)
    assert env.transition((19, 19), SE, SE) == ((19, 19), R_CRASH, True)
    assert env.transition((5, 9), E, E) == ((5, 9), R_CRASH, True)


def test_diagonal_corner_rule(env):
    # La diagonal sólo revisa la celda destino: puede rozar dos esquinas.
    assert env.transition((5, 9), SE, SE) == ((6, 10), COST_DIAGONAL, False)
    assert env.transition((6, 10), SE, SE) == ((7, 11), COST_DIAGONAL, False)
    assert env.transition((6, 10), NE, NE)[2] is False  # (5, 11) libre
    assert env.transition((6, 10), N, N) == ((6, 10), R_CRASH, True)


def test_goal(env):
    assert env.transition((7, 14), E, E) == ((7, 15), R_GOAL, True)


def test_slip_distribution():
    env = GridWorld(p_slip=0.1, seed=123)
    n = 60_000
    c = Counter(env.sample_real_action(E) for _ in range(n))
    assert set(c) == {E, NE, SE, STAY}
    assert c[E] / n == pytest.approx(0.9, abs=0.01)
    for a in (NE, SE, STAY):
        assert c[a] / n == pytest.approx(0.1 / 3, abs=0.005)


def test_outcomes_sum_to_one():
    env = GridWorld(p_slip=0.1)
    for a in range(8):
        assert sum(p for p, _ in env.outcomes(a)) == pytest.approx(1.0)


def test_episode_terminates_on_crash():
    env = GridWorld(p_slip=0.0, seed=0)
    env.reset()
    total = 0
    for _ in range(10):
        _, r, done, trunc, _ = env.step(N)
        total += r
        if done:
            break
    assert done and r == R_CRASH
    assert total == pytest.approx(4 * COST_CARDINAL + R_CRASH)


def test_targets_differ():
    q = AgenteRL(400, 8, "qlearning", gamma=0.9)
    s = AgenteRL(400, 8, "sarsa", gamma=0.9)
    for ag in (q, s):
        ag.Q[5] = [10, -20, 0, 0, 0, 0, 0, 0]
    assert q.target(-1, 5, 1, False) == pytest.approx(-1 + 0.9 * 10)
    assert s.target(-1, 5, 1, False) == pytest.approx(-1 + 0.9 * -20)
    assert q.target(100, 5, 1, True) == 100


def test_agent_learns():
    env = GridWorld(p_slip=0.1, seed=1)
    ag = AgenteRL(env.n_states, env.n_actions, "sarsa", seed=1)
    for _ in range(3000):
        ag.run_episode(env)
    outcomes = [ag.run_episode(env, learn=False, epsilon=0.0)[2] for _ in range(100)]
    assert outcomes.count("goal") > 80
    assert np.isfinite(ag.Q).all()
