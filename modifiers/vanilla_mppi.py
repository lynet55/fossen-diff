"""Vanilla MPPI for the fossen UUV --- one concrete, self-contained implementation.

Information-theoretic MPPI (Williams et al. 2017, https://arxiv.org/pdf/1707.02342):

    eps_k   ~ N(0, Sigma)                                k = 1..K samples
    v_k      = clip(U + eps_k)                            perturbed control sequences
    S_k      = phi(x_T) + sum_t [ q(x_t, v_t) + lambda * U_t^T Sigma^-1 eps_t ]
    w_k      = softmax(-S_k / lambda)
    U       <- U + sum_k w_k eps_k
    apply U_0, shift U left (warm start)

This file is deliberately NOT generalizable. Everything is hard-wired: the state
layout, the cost, the dynamics, the hyperparameters. To try a variant, copy the
file and edit the function you care about --- don't add config knobs. The only
shared dependency is the fossen model, which is imported.

State x = [nu(6), pos(3), R(9 flattened)]  (18,)   body vel, NED pos, body->NED rot
Control u = tau = [Fx, Fy, Fz, Tx, Ty, Tz]  (6,)

I/O (ROS topics, RViz rollouts, MarineGym UDP action) lives in ros_io.py; this file only
turns an ros_io.Observation into an ros_io.Command (tau, the velocity it produces, rollouts).

Obstacles enter the cost twice: a clearance (distance) field computed from the occupancy
grid on every map update (memory of walls out of view), plus raw sonar hits within
SCAN_RANGE (fresh, catches moving obstacles the decaying map blurs).

Two cost modes, picked per tick: while a nominal command is fresh the MPPI tracks it
(safety filter: pilot intent + sonar avoidance); otherwise it drives to /bluerov/goal.

Run as a ROS node:  uv run python modifiers/vanilla_mppi.py   (--mode test: one step, no ROS)
"""

import os
import sys

import jax
import jax.numpy as jnp

# Make `models` importable when this file is run directly (python modifiers/vanilla_mppi.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fossen_diff import fossen_rollout
import ros_io

# --------------------------------------------------------------------------- #
# Hyperparameters (edit these directly)
# --------------------------------------------------------------------------- #
DT = 0.05          # step size [s]
HORIZON = 30       # T, planning steps
NUM_SAMPLES = 1024  # K, rollouts per step
NU = 6             # control dimension (tau)

TEMPERATURE = 1.0   # lambda; lower -> greedier, higher -> closer to a plain average
NOISE_SIGMA = 10.0  # exploration std per control channel (Sigma = NOISE_SIGMA**2 * I)
U_MIN, U_MAX = -50.0, 50.0  # control bounds

# Velocity limits of the MarineGym kinematic block [u, v, w, r] (m/s, rad/s);
# must match task.controller.{u,v,w,r}_max, faster commands are clipped by the sim.
VEL_MAX = jnp.array([0.45, 0.30, 0.25, 0.5])
SONAR_RAYS = 256    # fixed sonar buffer size (padded/truncated) so mppi_step never retraces
SAFE_DIST = 0.6     # [m] horizontal clearance below which obstacles are penalized
SCAN_RANGE = 1.5    # [m] only sonar hits closer than this enter the cost; the grid covers the rest
OCC_THRESHOLD = 45  # grid cells >= this (0..100) are occupied; matches the sim planner's 0.45
VIS_ROLLOUTS = 64     # rollouts drawn, picked evenly over the cost ranking (best to worst)
VIS_LOG_RANGE = 20.0  # log-weight span (nats) mapped green -> red; weights below that are red


# --------------------------------------------------------------------------- #
# Problem: dynamics (fossen prediction model) and cost
# --------------------------------------------------------------------------- #
def dynamics(x, u):
    """One fossen step: x_t, tau -> x_{t+1}."""
    nu, pos, R = x[0:6], x[6:9], x[9:18].reshape(3, 3)
    out = fossen_rollout(nu, u[None], DT, init_pos=pos, init_R=R)
    return jnp.concatenate([out["nu"][0], out["pos"][0], out["R"][0].ravel()])


# ref = (goal(3) NED, w_goal, nominal(4) [u,v,w,r] FRD, w_track, obstacles(SONAR_RAYS, 2) NED xy,
#        grid = (clearance(H, W) [m], origin(2) map xy, resolution))
def _obstacle_cost(x, obstacles):
    d = jnp.linalg.norm(obstacles - x[6:8], axis=-1)  # horizontal: sonar is planar, walls vertical
    return 100.0 * jnp.sum(jnp.maximum(SAFE_DIST - d, 0.0) ** 2)


def _grid_cost(x, grid):
    clearance, origin, res = grid
    # NED (x, y) -> map (x, -y) -> fractional cell (row, col); bilinear, edge-clamped.
    col = (x[6] - origin[0]) / res
    row = (-x[7] - origin[1]) / res
    d = jax.scipy.ndimage.map_coordinates(clearance, [row[None], col[None]], order=1, mode="nearest")[0]
    return 100.0 * jnp.maximum(SAFE_DIST - d, 0.0) ** 2


def _collision_cost(x, ref):
    return _obstacle_cost(x, ref[4]) + _grid_cost(x, ref[5])


def running_cost(x, u, ref):
    goal, w_goal, nominal, w_track = ref[:4]
    vel = x[jnp.array([0, 1, 2, 5])]
    return (w_goal * jnp.sum((x[6:9] - goal) ** 2)
            + w_track * 10.0 * jnp.sum((vel - nominal) ** 2)
            + 100.0 * jnp.sum(jnp.maximum(jnp.abs(vel) - VEL_MAX, 0.0) ** 2)
            + _collision_cost(x, ref)
            + 1e-3 * jnp.sum(u**2))


def terminal_cost(x, ref):
    goal, w_goal = ref[:2]
    return 10.0 * w_goal * jnp.sum((x[6:9] - goal) ** 2) + _collision_cost(x, ref)


# --------------------------------------------------------------------------- #
# MPPI step
# --------------------------------------------------------------------------- #
def _rollout_cost(x0, controls, ref):
    """Total cost of one control sequence (T, nu) rolled out from x0, and its positions (T, 3)."""
    def step(x, u):
        x_next = dynamics(x, u)
        return x_next, (running_cost(x, u, ref), x_next[6:9])

    x_T, (step_costs, positions) = jax.lax.scan(step, x0, controls)
    return jnp.sum(step_costs) + terminal_cost(x_T, ref), positions


@jax.jit
def mppi_step(U, key, x0, ref):
    """One MPPI iteration. Returns (u0, U_next already shifted, key_next, vis).

    vis = (positions (VIS_ROLLOUTS, T+1, 3) NED incl. x0, log-weights (VIS_ROLLOUTS,)) for plotting.
    """
    key, noise_key = jax.random.split(key)

    # 1. Perturbed control sequences, respecting the bounds.
    eps = NOISE_SIGMA * jax.random.normal(noise_key, (NUM_SAMPLES, HORIZON, NU))
    V = jnp.clip(U[None] + eps, U_MIN, U_MAX)
    eps = V - U[None]  # effective noise after clipping

    # 2. Roll out all samples in parallel.
    state_costs, positions = jax.vmap(_rollout_cost, in_axes=(None, 0, None))(x0, V, ref)

    # 3. Control cost lambda * U^T Sigma^-1 eps (isotropic Sigma = sigma^2 I).
    control_costs = (TEMPERATURE / NOISE_SIGMA**2) * jnp.einsum("tu,ktu->k", U, eps)
    costs = state_costs + control_costs

    # 4. Weight, update, warm-start shift.
    weights = jax.nn.softmax(-costs / TEMPERATURE)
    U_new = jnp.clip(U + jnp.einsum("k,ktu->tu", weights, eps), U_MIN, U_MAX)
    u0 = U_new[0]
    U_next = jnp.concatenate([U_new[1:], U_new[-1:]], axis=0)

    # 5. Visualization subset: evenly spaced over the cost ranking so the color spread shows.
    idx = jnp.argsort(costs)[:: max(1, NUM_SAMPLES // VIS_ROLLOUTS)][:VIS_ROLLOUTS]
    start = jnp.broadcast_to(x0[6:9], (idx.shape[0], 1, 3))
    vis = (jnp.concatenate([start, positions[idx]], axis=1), jnp.log(weights[idx] + 1e-30))
    return u0, U_next, key, vis


# --------------------------------------------------------------------------- #
# ROS node
# --------------------------------------------------------------------------- #
# Before the first map: a 1x1 "everything is far" grid (the real map's shape retraces mppi_step once).
NO_GRID = (jnp.full((1, 1), 1e3), jnp.zeros(2), jnp.float32(1.0))


def clearance_grid(occ, origin, res):
    """Occupancy (H, W) 0..100 -> (clearance in metres to the nearest occupied cell, origin, res)."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    free = np.asarray(occ) < OCC_THRESHOLD
    clearance = distance_transform_edt(free, sampling=res) if not free.all() else np.full(free.shape, 1e3)
    return (jnp.asarray(clearance, jnp.float32), jnp.asarray(origin, jnp.float32), jnp.float32(res))


def make_ref(goal=None, nominal=None, obstacles=None, grid=None):
    """Pack the cost reference; a missing goal/nominal just zeroes its weight."""
    far = jnp.full((SONAR_RAYS, 2), 1e4)
    return (jnp.zeros(3) if goal is None else jnp.asarray(goal, jnp.float32),
            jnp.float32(goal is not None and nominal is None),
            jnp.zeros(4) if nominal is None else jnp.asarray(nominal, jnp.float32),
            jnp.float32(nominal is not None),
            far if obstacles is None else jnp.asarray(obstacles, jnp.float32),
            NO_GRID if grid is None else grid)


class Controller:
    """ros_io.Observation -> ros_io.Command. Holds the warm-started plan between ticks."""

    def __init__(self):
        import numpy as np
        self.np = np
        self.U = jnp.zeros((HORIZON, NU))
        self.key = jax.random.key(0)
        self.grid, self.grid_version = None, None
        self.predict = jax.jit(dynamics)  # eager fossen_rollout re-traces its scan every call (~170 ms)

    def __call__(self, obs):
        np = self.np
        if obs.grid is not None and obs.grid.version != self.grid_version:  # once per map update
            self.grid = clearance_grid(obs.grid.occ, obs.grid.origin, obs.grid.res)
            self.grid_version = obs.grid.version

        obstacles = None
        if obs.scan is not None:  # fixed size: hits within SCAN_RANGE, the rest far away (grid covers it)
            near = obs.scan[np.linalg.norm(obs.scan - obs.x[6:8], axis=-1) <= SCAN_RANGE][:SONAR_RAYS]
            obstacles = np.full((SONAR_RAYS, 2), 1e4)
            obstacles[: len(near)] = near

        x0 = jnp.asarray(obs.x, jnp.float32)
        ref = make_ref(obs.goal, obs.nominal, obstacles, self.grid)
        u0, self.U, self.key, (positions, log_w) = mppi_step(self.U, self.key, x0, ref)

        log_w = np.asarray(log_w)
        return ros_io.Command(
            tau=np.asarray(u0),
            nu_next=np.asarray(self.predict(x0, u0)[0:6]),
            rollouts=np.asarray(positions),
            scores=1.0 + (log_w - log_w.max()) / VIS_LOG_RANGE,
        )


def test():
    """One-step shape check without ROS: feed a dummy state through the MPPI and
    the model, and print the shapes so a change hasn't broken the wiring."""
    x0 = jnp.concatenate([jnp.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),  # nu
                          jnp.zeros(3),                                # pos
                          jnp.eye(3).ravel()])                         # R
    U = jnp.zeros((HORIZON, NU))
    key = jax.random.key(0)
    obstacles = jnp.full((SONAR_RAYS, 2), 1e4).at[0].set(jnp.array([1.0, 0.0]))
    occ = jnp.zeros((20, 40)).at[:, 15].set(100)  # wall across map x = 1.5 m (0.1 m cells)
    grid = clearance_grid(occ, origin=[0.0, -1.0], res=0.1)
    ref = make_ref(goal=[5.0, 0.0, 0.0], obstacles=obstacles, grid=grid)

    u0, U_next, key, (vis_pos, vis_logw) = mppi_step(U, key, x0, ref)
    x_next = dynamics(x0, u0)
    print(f"x0      {x0.shape}  (expect (18,))")
    print(f"u0      {u0.shape}  (expect ({NU},))   = {jnp.array_str(u0, precision=2)}")
    print(f"U_next  {U_next.shape}  (expect ({HORIZON}, {NU}))")
    print(f"x_next  {x_next.shape}  (expect (18,))")
    print(f"vis     {vis_pos.shape} {vis_logw.shape}  (expect ({VIS_ROLLOUTS}, {HORIZON + 1}, 3) ({VIS_ROLLOUTS},))"
          f"  log-w best {float(vis_logw[0]):.1f} worst {float(vis_logw[-1]):.1f}")
    for px in (0.5, 1.2, 1.5):
        x = x0.at[6].set(px)
        print(f"grid cost at x={px}  {float(_grid_cost(x, grid)):.2f}  (wall at 1.5, SAFE_DIST {SAFE_DIST})")

    u0, _, _, _ = mppi_step(U, key, x0, make_ref(nominal=[0.3, 0.0, 0.0, 0.0]))
    print(f"u0 (tracking nominal surge 0.3) = {jnp.array_str(u0, precision=2)}")


if __name__ == "__main__":
    import argparse

    # Tuning knobs for an agent to sweep; override the module constants above
    # (set before the first jit trace). Everything else is edited in the file.
    p = argparse.ArgumentParser(description="Vanilla MPPI for the fossen UUV.")
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--noise-sigma", type=float, default=NOISE_SIGMA)
    p.add_argument("--u-min", type=float, default=U_MIN)
    p.add_argument("--u-max", type=float, default=U_MAX)
    p.add_argument("--mode", choices=("node", "test"), default="node",
                   help="run as a ROS node, or do the one-step shape check")
    a = p.parse_args()

    HORIZON, NUM_SAMPLES = a.horizon, a.num_samples
    TEMPERATURE, NOISE_SIGMA = a.temperature, a.noise_sigma
    U_MIN, U_MAX = a.u_min, a.u_max

    if a.mode == "test":
        test()
    else:
        ros_io.run(Controller(), DT, sinks=ros_io.marinegym_sinks(VEL_MAX), name="vanilla_mppi")
