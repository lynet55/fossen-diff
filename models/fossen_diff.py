"""
JAX 6-DOF ROV model (BlueROV2) following Skaldebø, Amundsen, Su & Kelasidi,
"Modeling of Remotely Operated Vehicle (ROV) Operations for Aquaculture",
IEEE ICMA 2023, doi:10.1109/ICMA57826.2023.10215600.

Equation numbers refer to that paper:

    eta_dot = J(q) nu                                                    (1)
    M_RB nu_dot + C_RB(nu) nu + M_A nu_r_dot + C_A(nu_r) nu_r
        + D(nu_r) nu_r + g(eta) = tau                                    (2)

Usage
-----
    import jax
    jax.config.update("jax_enable_x64", True)   # optional; the module does not set it

    from fossen_jax import Vehicle, fossen_model, fossen_rollout

    nu = fossen_model(init_state, taus, dt)                          # (T, 6) velocities
    out = fossen_rollout(init_state, taus, dt, current_ned=[0.2, 0, 0])
    heavier = Vehicle(mass=12.0)                                     # BlueROV2 with one change
    nu = fossen_model(init_state, taus, dt, vehicle=heavier)

`Vehicle` is a frozen dataclass registered as a JAX pytree, so it can be passed
through jit / grad / vmap like any array argument (e.g. jax.grad w.r.t. mass).

Attitude is a rotation matrix integrated with the exact exponential map (Rodrigues)
instead of the paper's quaternion ODE; both represent R(q) in SO(3) without
Euler-angle singularities.
"""

import dataclasses
import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

__all__ = [
    "Vehicle",
    "BLUEROV2",
    "fossen_model",
    "fossen_rollout",
    "fossen_model_jit",
    "fossen_rollout_jit",
    "fossen_model_batched",
    "nu_dot_fn",
    "rigid_body_mass",
    "added_mass",
    "coriolis",
    "damping",
    "restoring",
]

GRAVITY = 9.81


# ----------------------------------------------------------------------------
# Vehicle parameters (BlueROV2, Tables III and IV)
# ----------------------------------------------------------------------------
@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["mass", "w_over_b", "r_g", "r_b", "inertia", "hydro"],
    meta_fields=[],
)
@dataclasses.dataclass(frozen=True)
class Vehicle:
    """Rigid-body, hydrostatic and hydrodynamic parameters. Defaults: BlueROV2.

    hydro layout (same order as the `parameters` argument):
      [0:6]   added mass      X_udot, Y_vdot, Z_wdot, K_pdot, M_qdot, N_rdot
      [6:12]  linear damping  X_u, Y_v, Z_w, K_p, M_q, N_r
      [12:18] quadratic damp. X_u|u|, Y_v|v|, Z_w|w|, K_p|p|, M_q|q|, N_r|r|

    Signs follow SNAME (all negative). Table III lists added mass as positive
    magnitudes, but eq. (10) M_A = -diag(X_udot, ...) with M_A > 0 requires the
    derivatives to be negative.
    """

    mass: float = 11.5          # m [kg]
    w_over_b: float = 0.98      # W/B [-]
    r_g: jnp.ndarray = dataclasses.field(            # CG in {b} [m]
        default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))
    r_b: jnp.ndarray = dataclasses.field(            # CB in {b} [m]
        default_factory=lambda: jnp.array([0.0, 0.0, -0.02]))
    inertia: jnp.ndarray = dataclasses.field(        # [Ix, Iy, Iz] [kg m^2]
        default_factory=lambda: jnp.array([0.16, 0.16, 0.16]))
    hydro: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.array([
            -5.5, -12.7, -14.57, -5.5, -5.5, -5.5,
            -4.03, -6.22, -5.18, -0.07, -0.07, -0.07,
            -18.18, -21.66, -39.99, -1.55, -1.55, -1.55,
        ]))

    def replace(self, **changes) -> "Vehicle":
        return dataclasses.replace(self, **changes)


BLUEROV2 = Vehicle()
DEFAULT_PARAMS = BLUEROV2.hydro


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def skew(v):
    """S(lambda), eq. (6)."""
    return jnp.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def rodrigues(phi):
    """exp(S(phi)) with a gradient-safe small-angle branch."""
    sq = jnp.dot(phi, phi)
    small = sq < 1e-24
    angle = jnp.sqrt(jnp.where(small, 1.0, sq))
    k = skew(phi / angle)
    full = jnp.eye(3) + jnp.sin(angle) * k + (1.0 - jnp.cos(angle)) * (k @ k)
    return jnp.where(small, jnp.eye(3) + skew(phi), full)


def orthonormalize(R):
    c0 = R[:, 0] / jnp.linalg.norm(R[:, 0])
    c1 = R[:, 1] - jnp.dot(R[:, 1], c0) * c0
    c1 = c1 / jnp.linalg.norm(c1)
    c2 = jnp.cross(c0, c1)
    return jnp.stack((c0, c1, c2), axis=1)


def _sanitize(x, clip):
    x = jnp.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
    return jnp.clip(x, -clip, clip)


# ----------------------------------------------------------------------------
# Model terms
# ----------------------------------------------------------------------------
def rigid_body_mass(veh: Vehicle):
    """M_RB, eq. (7)/(9)."""
    m = veh.mass
    s = skew(veh.r_g)
    return jnp.block([
        [m * jnp.eye(3), -m * s],
        [m * s, jnp.diag(veh.inertia)],
    ])


def added_mass(hydro):
    """M_A, eq. (10)."""
    return -jnp.diag(hydro[0:6])


def coriolis(M, nu):
    """C(nu) from a 6x6 mass matrix, eq. (13). Used for both RB and A."""
    v1, v2 = nu[0:3], nu[3:6]
    a = M[0:3, 0:3] @ v1 + M[0:3, 3:6] @ v2
    b = M[3:6, 0:3] @ v1 + M[3:6, 3:6] @ v2
    z = jnp.zeros((3, 3))
    return jnp.block([
        [z, -skew(a)],
        [-skew(a), -skew(b)],
    ])


def damping(hydro, nu_r):
    """D(nu_r) = D_L + D_NL(nu_r), eqs. (8), (11), (12)."""
    return -jnp.diag(hydro[6:12]) - jnp.diag(hydro[12:18] * jnp.abs(nu_r))


def restoring(veh: Vehicle, R):
    """g(eta), eq. (18). R maps {b} -> {n} (NED, z down)."""
    W = veh.mass * GRAVITY
    B = W / veh.w_over_b
    f_g = R.T @ jnp.array([0.0, 0.0, W])    # eq. (17), expressed in {b}
    f_b = R.T @ jnp.array([0.0, 0.0, -B])
    return -jnp.concatenate((
        f_g + f_b,
        jnp.cross(veh.r_g, f_g) + jnp.cross(veh.r_b, f_b),
    ))


def nu_dot_fn(nu, R, tau, hydro, veh: Vehicle, current_ned):
    """Solve eq. (2) for nu_dot.

    current_ned: ocean current velocity in {n}, assumed constant and irrotational.
    In {b}: nu_c = [R^T v_c; 0], so nu_c_dot = [-omega x (R^T v_c); 0] and
    M_A nu_r_dot = M_A nu_dot - M_A nu_c_dot.
    """
    M_RB = rigid_body_mass(veh)
    M_A = added_mass(hydro)

    v_c = R.T @ current_ned
    nu_c = jnp.concatenate((v_c, jnp.zeros(3)))
    nu_c_dot = jnp.concatenate((-jnp.cross(nu[3:6], v_c), jnp.zeros(3)))
    nu_r = nu - nu_c

    rhs = (
        tau
        - coriolis(M_RB, nu) @ nu
        - coriolis(M_A, nu_r) @ nu_r
        - damping(hydro, nu_r) @ nu_r
        - restoring(veh, R)
        + M_A @ nu_c_dot
    )
    return jnp.linalg.solve(M_RB + M_A, rhs)


# ----------------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------------
def _taus_from_inputs(inputs):
    # 6 columns: [Fx, Fy, Fz, Tx, Ty, Tz]; 12 columns: [state(6), tau(6)]
    return inputs[:, 0:6] if inputs.shape[-1] == 6 else inputs[:, 6:12]


def fossen_rollout(init_state, inputs, dt, parameters=None,
                   vehicle: Vehicle = BLUEROV2,
                   current_ned=None, init_pos=None, init_R=None,
                   clip: Optional[float] = 50.0):
    """
    init_state:  (6,) body velocities nu = [u, v, w, p, q, r]
    inputs:      (T, 6) generalized forces tau, or (T, 12) dataset rows [state, tau]
    dt:          step size [s]
    parameters:  (18,) hydrodynamic coefficients; overrides vehicle.hydro
    vehicle:     Vehicle (default BlueROV2)
    current_ned: (3,) current velocity in NED (default: no current)
    init_pos:    (3,) NED position (default 0); init_R: (3,3) body->NED (default I)
    clip:        None to disable, else the original +-clip / NaN guard
                 (gradient is zero wherever it saturates)
    returns dict with 'nu' (T,6), 'pos' (T,3), 'R' (T,3,3)
    """
    hydro = vehicle.hydro if parameters is None else jnp.asarray(parameters)
    taus = _taus_from_inputs(jnp.asarray(inputs))
    nu0 = jnp.asarray(init_state, dtype=taus.dtype)
    v_c = jnp.zeros(3) if current_ned is None else jnp.asarray(current_ned)
    pos0 = jnp.zeros(3) if init_pos is None else jnp.asarray(init_pos)
    R0 = jnp.eye(3) if init_R is None else jnp.asarray(init_R)

    def step(carry, tau):
        nu, pos, R = carry
        if clip is not None:
            nu = _sanitize(nu, clip)
        nu_dot = nu_dot_fn(nu, R, tau, hydro, vehicle, v_c)
        if clip is not None:
            nu_dot = _sanitize(nu_dot, clip)
        # Semi-implicit Euler, same ordering as the original:
        # velocity first, then kinematics (eq. 1) with the updated velocity.
        nu_next = nu + dt * nu_dot
        pos_next = pos + dt * (R @ nu_next[0:3])
        R_next = orthonormalize(R @ rodrigues(nu_next[3:6] * dt))
        return (nu_next, pos_next, R_next), (nu_next, pos_next, R_next)

    _, (nus, poss, Rs) = lax.scan(step, (nu0, pos0, R0), taus)
    return {"nu": nus, "pos": poss, "R": Rs}


def fossen_model(init_state, inputs, dt, parameters=None, **kwargs):
    """Drop-in for the original: returns body velocities (T, 6)."""
    return fossen_rollout(init_state, inputs, dt, parameters, **kwargs)["nu"]


def _batched(init_states, inputs, dt, parameters=None, **kwargs):
    f = lambda s, x: fossen_model(s, x, dt, parameters, **kwargs)
    return jax.vmap(f)(init_states, inputs)


# dt and clip are static (Python floats); everything else, including `vehicle`, is traced.
fossen_model_jit = jax.jit(fossen_model, static_argnums=(2,), static_argnames=("clip",))
fossen_rollout_jit = jax.jit(fossen_rollout, static_argnums=(2,), static_argnames=("clip",))
fossen_model_batched = jax.jit(_batched, static_argnums=(2,), static_argnames=("clip",))


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    T = 50
    u = jax.random.normal(jax.random.PRNGKey(0), (T, 6))
    init = jnp.zeros(6)
    print("nu[-1]:", fossen_model_jit(init, u, 0.05)[-1])

    loss = lambda veh: jnp.mean(fossen_model(init, u, 0.05, vehicle=veh) ** 2)
    g = jax.jit(jax.grad(loss))(BLUEROV2)
    print("d loss / d mass:", g.mass, " d loss / d r_b:", g.r_b)