# Implementation details

## JAX on CPU vs GPU

The GPU build of JAX is installed (`jax[cuda13]`), so JAX picks the GPU by default.

The propagator in `main.py` runs one tiny Fossen step per tick. At that size, copying data to and from the GPU can cost more than the computation, so the CPU may be faster. Time both:

```bash
JAX_PLATFORMS=cpu uv run python main.py
```

The GPU should pay off for batched or long rollouts, such as MPPI (`fossen_model_batched`).

## Implemented dynamics model

Implemented in [`models/fossen_diff.py`](../models/fossen_diff.py), following Skaldebø, Amundsen, Su and Kelasidi, [*Modeling of Remotely Operated Vehicle (ROV) Operations for Aquaculture*](https://doi.org/10.1109/ICMA57826.2023.10215600), IEEE ICMA 2023. It is a 6-DOF model of the BlueROV2. Equation numbers below are the paper's.

### State and frames

- Position $\mathbf{p} \in \mathbb{R}^3$ in the NED frame $\{n\}$ (z points down).
- Attitude $\mathbf{R} \in SO(3)$, mapping body $\{b\}$ to $\{n\}$.
- Body velocity $\boldsymbol{\nu} = [u, v, w, p, q, r]^\top$.
- Input $\boldsymbol{\tau} = [F_x, F_y, F_z, T_x, T_y, T_z]^\top$, generalized forces in $\{b\}$.

### Equations of motion

$$\dot{\mathbf{p}} = \mathbf{R}\,\boldsymbol{\nu}_{1:3} \tag{1}$$

$$\mathbf{M}_{RB}\dot{\boldsymbol{\nu}} + \mathbf{C}_{RB}(\boldsymbol{\nu})\boldsymbol{\nu} + \mathbf{M}_A\dot{\boldsymbol{\nu}}_r + \mathbf{C}_A(\boldsymbol{\nu}_r)\boldsymbol{\nu}_r + \mathbf{D}(\boldsymbol{\nu}_r)\boldsymbol{\nu}_r + \mathbf{g}(\mathbf{R}) = \boldsymbol{\tau} \tag{2}$$

The relative velocity is $\boldsymbol{\nu}_r = \boldsymbol{\nu} - \boldsymbol{\nu}_c$, where the ocean current is constant and irrotational: $\boldsymbol{\nu}_c = [\mathbf{R}^\top \mathbf{v}_c;\ \mathbf{0}]$. Without a current, $\boldsymbol{\nu}_r = \boldsymbol{\nu}$. The code solves (2) for $\dot{\boldsymbol{\nu}}$:

$$\dot{\boldsymbol{\nu}} = (\mathbf{M}_{RB} + \mathbf{M}_A)^{-1}\Big(\boldsymbol{\tau} - \mathbf{C}_{RB}\boldsymbol{\nu} - \mathbf{C}_A\boldsymbol{\nu}_r - \mathbf{D}\boldsymbol{\nu}_r - \mathbf{g} + \mathbf{M}_A\dot{\boldsymbol{\nu}}_c\Big)$$

### Terms

Rigid-body and added mass (eqs. 7, 9, 10):

$$\mathbf{M}_{RB} = \begin{bmatrix} m\mathbf{I}_3 & -m\mathbf{S}(\mathbf{r}_g) \\ m\mathbf{S}(\mathbf{r}_g) & \mathrm{diag}(I_x, I_y, I_z) \end{bmatrix}, \qquad \mathbf{M}_A = -\mathrm{diag}(X_{\dot u}, Y_{\dot v}, Z_{\dot w}, K_{\dot p}, M_{\dot q}, N_{\dot r})$$

Coriolis, the same form for the rigid-body and added-mass parts (eq. 13). Split $\mathbf{M}$ into $2\times2$ blocks and let $\mathbf{a} = \mathbf{M}_{11}\boldsymbol{\nu}_{1:3} + \mathbf{M}_{12}\boldsymbol{\nu}_{4:6}$ and $\mathbf{b} = \mathbf{M}_{21}\boldsymbol{\nu}_{1:3} + \mathbf{M}_{22}\boldsymbol{\nu}_{4:6}$:

$$\mathbf{C}(\boldsymbol{\nu}) = \begin{bmatrix} \mathbf{0} & -\mathbf{S}(\mathbf{a}) \\ -\mathbf{S}(\mathbf{a}) & -\mathbf{S}(\mathbf{b}) \end{bmatrix}$$

Damping, linear plus quadratic (eqs. 8, 11, 12):

$$\mathbf{D}(\boldsymbol{\nu}_r) = -\mathrm{diag}(X_u, Y_v, Z_w, K_p, M_q, N_r) - \mathrm{diag}\big(X_{u|u|}|u_r|, \dots, N_{r|r|}|r_r|\big)$$

Restoring forces (eqs. 17, 18), with weight $W = mg$ and buoyancy $B = W / (W/B)$:

$$\mathbf{f}_g = \mathbf{R}^\top [0, 0, W]^\top, \quad \mathbf{f}_b = \mathbf{R}^\top [0, 0, -B]^\top, \quad \mathbf{g} = -\begin{bmatrix} \mathbf{f}_g + \mathbf{f}_b \\ \mathbf{r}_g \times \mathbf{f}_g + \mathbf{r}_b \times \mathbf{f}_b \end{bmatrix}$$

$\mathbf{S}(\cdot)$ is the skew-symmetric matrix, $\mathbf{S}(\boldsymbol{\lambda})\mathbf{x} = \boldsymbol{\lambda} \times \mathbf{x}$ (eq. 6).

### Discretization

Per step of size $\Delta t$ (semi-implicit Euler: velocity first, then kinematics with the updated velocity):

$$\boldsymbol{\nu}_{k+1} = \boldsymbol{\nu}_k + \Delta t\,\dot{\boldsymbol{\nu}}_k, \qquad \mathbf{p}_{k+1} = \mathbf{p}_k + \Delta t\,\mathbf{R}_k\boldsymbol{\nu}_{k+1,1:3}, \qquad \mathbf{R}_{k+1} = \mathrm{orth}\big(\mathbf{R}_k \exp(\mathbf{S}(\boldsymbol{\nu}_{k+1,4:6}\Delta t))\big)$$

The attitude is a rotation matrix integrated with the exact exponential map (Rodrigues), then re-orthonormalized with Gram-Schmidt. This avoids Euler-angle singularities, as the paper's quaternion ODE does. $\boldsymbol{\nu}$ and $\dot{\boldsymbol{\nu}}$ are NaN-guarded and clipped to $\pm 50$.

### Properties

| Property | Value |
|---|---|
| Mass $m$ | 11.5 kg |
| $W/B$ | 0.98 (slightly buoyant) |
| $\mathbf{r}_g$ | $[0, 0, 0]$ m |
| $\mathbf{r}_b$ | $[0, 0, -0.02]$ m |
| Inertia $[I_x, I_y, I_z]$ | $[0.16, 0.16, 0.16]$ kg m² |
| Added mass $[X_{\dot u}, Y_{\dot v}, Z_{\dot w}, K_{\dot p}, M_{\dot q}, N_{\dot r}]$ | $[-5.5, -12.7, -14.57, -5.5, -5.5, -5.5]$ |
| Linear damping $[X_u, Y_v, Z_w, K_p, M_q, N_r]$ | $[-4.03, -6.22, -5.18, -0.07, -0.07, -0.07]$ |
| Quadratic damping $[X_{u\|u\|}, Y_{v\|v\|}, Z_{w\|w\|}, K_{p\|p\|}, M_{q\|q\|}, N_{r\|r\|}]$ | $[-18.18, -21.66, -39.99, -1.55, -1.55, -1.55]$ |

The hydrodynamic coefficients follow the SNAME sign convention, so they are all negative. The paper's Table III lists the added mass as positive magnitudes, but eq. (10) needs negative derivatives for $\mathbf{M}_A$ to be positive definite.

All of these are fields of the `Vehicle` dataclass, which is a JAX pytree. You can change them or differentiate through them, for example `jax.grad` with respect to the mass.
