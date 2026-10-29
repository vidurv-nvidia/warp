# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An example implementation of an FDTD solver.

A point source on the surface of a 3-D Luneburg lens launches a curved
wavefront into the lens. The lens straightens the wavefront into a parallel
beam that leaves the opposite side.

The example uses the finite-difference time-domain (FDTD) method on a Yee
staggered grid. It marches Maxwell's curl equations with a leapfrog scheme and
a matched loss rate ``gamma``:

    mu_0 * dH/dt = -curl E - gamma * mu_0 * H
    eps_0 * eps_r * dE/dt = curl H - gamma * eps_0 * eps_r * E - J

The three 2-D coordinate-plane views of a 3-D Yee cell are:

        x-y plane                 x-z plane                 y-z plane
           y                         z                         z
           ^                         ^                         ^
           |                         |                         |
      o---E_x---o               o---E_x---o               o---E_y---o
      |         |               |         |               |         |
      E_y H_z   E_y             E_z  H_y  E_z             E_z  H_x  E_z
      |         |               |         |               |         |
      o---E_x---o ---> x        o---E_x---o ---> x        o---E_y---o ---> y

The temporal staggering is as follows:

    time       (n - 1/2) dt          n dt          (n + 1/2) dt        (n + 1) dt
               ------------------------------------------------------------------>

    H          H^(n-1/2)  -- update_h using E^n --> H^(n+1/2)

    E                                  E^n  -- update_e using H^(n+1/2) --> E^(n+1)

The ``update_e`` kernel keeps the tangential electric field components at zero on the six
outer faces. The ``gamma`` profile attenuates outgoing waves toward zero before they reach
those faces, reducing boundary reflections. It ramps from zero at the inner edge of the attenuation
layer to ``GAMMA_MAX`` at the wall and vanishes elsewhere.

The lens is a graded-index sphere of radius ``R_LENS`` centered on the origin,

    eps_r(r) = n(r)^2 = 2 - (r / R_LENS)^2    for r <= R_LENS

the Luneburg profile. A soft current source on the ``-z`` surface launches a curved
wavefront into the lens. The graded index refracts it into a parallel beam leaving the
``+z`` surface, visible in the rendered ``E_x`` center plane. ``J`` is ramped up from rest.

The scheme is second-order accurate in space and time and stable for
``dt <= dx / (c * sqrt(3))`` in 3-D. Units are normalized so that
``c = eps_0 = mu_0 = lambda_0 = f_0 = T_0 = 1``, hence a vacuum impedance of 1.
The cubic domain is fixed at ``HALF_WIDTH`` wavelengths half-extent; ``--resolution``
sets how finely it is gridded and every other grid quantity derives from it.

Requires Matplotlib for interactive visualization (use --headless to run without it).

Usage:
    python example_fdtd_3d.py
"""

import itertools
import math

import numpy as np

import warp as wp

wp.config.enable_backward = False

# physical inputs, in normalized units (c = mu_0 = eps_0 = lambda_0 = f_0 = T_0 = 1)
HALF_WIDTH = 4.25  # half-extent of the cubic domain
R_LENS = 3.0  # lens radius in vacuum wavelengths
L_ABSORBER = 2.0 / 3.0  # depth of the attenuation layer lining each wall
E0 = 1.0  # source amplitude, tuned at the reference grid spacing DX_REF
DX_REF = 1.0 / 30.0  # grid spacing at which E0 and PLOT_E_SCALE were tuned
SOURCE_RAMP = 3.0  # periods over which the source envelope rises from rest
SIM_DURATION = 36.0  # default headless run length, in source periods
DIAG_INTERVAL = 100  # timesteps between electromagnetic energy diagnostics
PLOT_E_SCALE = 2.5e-4  # colorbar half-range for the E_x frames; SOURCE_SCALE keeps it valid at every resolution
GAMMA_MAX = -2.0 * math.log(1.0e-3) / L_ABSORBER

# resolution and derived quantities
N_GRID = 128  # stored nodes per axis across the fixed domain
DX = 2.0 * HALF_WIDTH / (N_GRID - 1)
DT = 0.95 * DX / math.sqrt(3.0)  # 3-D CFL limit with a 0.95 safety factor
SOURCE_SCALE = (DX_REF / DX) ** 2  # holds the dipole moment, fixed across resolutions
SOURCE_IJ = (N_GRID - 1) // 2
SOURCE_K = round((HALF_WIDTH - R_LENS) / DX)


@wp.func
def loss_coeffs(gamma: float) -> tuple[float, float]:
    """Return the matched-loss coefficients for a field component at its position on the Yee grid."""
    gamma_dt_half = 0.5 * DT * gamma
    return (1.0 - gamma_dt_half) / (1.0 + gamma_dt_half), 1.0 / (1.0 + gamma_dt_half)


@wp.func
def inv_eps_r(x: float, y: float, z: float) -> float:
    """Return the inverse relative permittivity of the Luneburg lens at a point given by (x, y, z)."""
    r_sq = x * x + y * y + z * z

    eps_r = float(1.0)
    if r_sq < R_LENS * R_LENS:
        eps_r = 2.0 - r_sq / (R_LENS * R_LENS)

    return 1.0 / eps_r


@wp.func
def source_amplitude(timestep_index: int) -> float:
    """Return the scaled point-source term ``DX * J_x`` sampled at ``t^{n+1/2}``.

    ``SOURCE_SCALE`` keeps the dipole moment ``J_x * DX^3``, and with it the radiated
    field and ``PLOT_E_SCALE``, independent of resolution.
    """
    t = (float(timestep_index) + 0.5) * DT
    return -E0 * SOURCE_SCALE * wp.sin(2.0 * wp.pi * t) * (1.0 - wp.exp(-((t / SOURCE_RAMP) ** 2.0)))


@wp.kernel
def update_h(
    e_x: wp.array3d[float],
    e_y: wp.array3d[float],
    e_z: wp.array3d[float],
    gamma_integer: wp.array[float],
    gamma_half_offset: wp.array[float],
    h_x: wp.array3d[float],
    h_y: wp.array3d[float],
    h_z: wp.array3d[float],
):
    """Advance ``H`` from ``t = (n - 1/2) dt`` to ``t = (n + 1/2) dt`` using ``E^n``.

    The representative component update (the others follow by cyclic permutation) is::

        H_x^{n+1/2}|_{i, j+1/2, k+1/2} = a * H_x^{n-1/2}
            - (b * dt / dx) * [(E_z|_{j+1} - E_z|_j) - (E_y|_{k+1} - E_y|_k)]

    with the matched-loss coefficients of ``loss_coeffs`` sampled from the 1-D profiles
    ``gamma_integer`` and ``gamma_half_offset`` at each component's staggered coordinates
    (the grid is cubic, so all three axes index the same two profiles).
    """
    i, j, k = wp.tid()

    dt_dx = DT / DX
    # last index whose half-offset indices are still inside the walls
    last_staggered = N_GRID - 2

    # H_x at (i, j+1/2, k+1/2)
    if j <= last_staggered and k <= last_staggered:
        a_x, b_x = loss_coeffs(gamma_integer[i] + gamma_half_offset[j] + gamma_half_offset[k])
        curl_x = (e_z[i, j + 1, k] - e_z[i, j, k]) - (e_y[i, j, k + 1] - e_y[i, j, k])
        h_x[i, j, k] = a_x * h_x[i, j, k] - b_x * dt_dx * curl_x

    # H_y at (i+1/2, j, k+1/2)
    if i <= last_staggered and k <= last_staggered:
        a_y, b_y = loss_coeffs(gamma_half_offset[i] + gamma_integer[j] + gamma_half_offset[k])
        curl_y = (e_x[i, j, k + 1] - e_x[i, j, k]) - (e_z[i + 1, j, k] - e_z[i, j, k])
        h_y[i, j, k] = a_y * h_y[i, j, k] - b_y * dt_dx * curl_y

    # H_z at (i+1/2, j+1/2, k)
    if i <= last_staggered and j <= last_staggered:
        a_z, b_z = loss_coeffs(gamma_half_offset[i] + gamma_half_offset[j] + gamma_integer[k])
        curl_z = (e_y[i + 1, j, k] - e_y[i, j, k]) - (e_x[i, j + 1, k] - e_x[i, j, k])
        h_z[i, j, k] = a_z * h_z[i, j, k] - b_z * dt_dx * curl_z


@wp.kernel
def update_e(
    h_x: wp.array3d[float],
    h_y: wp.array3d[float],
    h_z: wp.array3d[float],
    gamma_integer: wp.array[float],
    gamma_half_offset: wp.array[float],
    timestep_counter: wp.array[wp.int32],
    e_x: wp.array3d[float],
    e_y: wp.array3d[float],
    e_z: wp.array3d[float],
):
    """Advance ``E`` from ``t = n dt`` to ``t = (n + 1) dt`` using ``H^{n+1/2}``.

    The representative component update (the others follow by cyclic permutation) is::

        E_x^{n+1}|_{i+1/2, j, k} = a * E_x^n + (b * dt / (eps_r * dx))
            * [(H_z|_{j+1/2} - H_z|_{j-1/2}) - (H_y|_{k+1/2} - H_y|_{k-1/2}) - dx * J_x]

    where ``eps_r`` is the Luneburg profile of ``inv_eps_r``, evaluated analytically. A
    point source drives ``J_x`` at the single node ``(SOURCE_IJ, SOURCE_IJ, SOURCE_K)``,
    with ``source_amplitude`` supplying ``dx * J_x``.

    """
    i, j, k = wp.tid()

    timestep_index = timestep_counter[0]

    dt_dx = DT / DX

    last_staggered = N_GRID - 2
    last_interior = N_GRID - 2

    x_node = float(i) * DX - HALF_WIDTH
    y_node = float(j) * DX - HALF_WIDTH
    z_node = float(k) * DX - HALF_WIDTH
    half_dx = 0.5 * DX

    # E_x at (i+1/2, j, k), carrying the point source
    if i <= last_staggered and 1 <= j <= last_interior and 1 <= k <= last_interior:
        a_x, b_x = loss_coeffs(gamma_half_offset[i] + gamma_integer[j] + gamma_integer[k])
        curl_x = (h_z[i, j, k] - h_z[i, j - 1, k]) - (h_y[i, j, k] - h_y[i, j, k - 1])
        if i == SOURCE_IJ and j == SOURCE_IJ and k == SOURCE_K:
            curl_x -= source_amplitude(timestep_index)
        inv_eps_x = inv_eps_r(x_node + half_dx, y_node, z_node)
        new_e_x = a_x * e_x[i, j, k] + b_x * dt_dx * inv_eps_x * curl_x
        e_x[i, j, k] = new_e_x

    # E_y at (i, j+1/2, k)
    if 1 <= i <= last_interior and j <= last_staggered and 1 <= k <= last_interior:
        a_y, b_y = loss_coeffs(gamma_integer[i] + gamma_half_offset[j] + gamma_integer[k])
        curl_y = (h_x[i, j, k] - h_x[i, j, k - 1]) - (h_z[i, j, k] - h_z[i - 1, j, k])
        inv_eps_y = inv_eps_r(x_node, y_node + half_dx, z_node)
        new_e_y = a_y * e_y[i, j, k] + b_y * dt_dx * inv_eps_y * curl_y
        e_y[i, j, k] = new_e_y

    # E_z at (i, j, k+1/2)
    if 1 <= i <= last_interior and 1 <= j <= last_interior and k <= last_staggered:
        a_z, b_z = loss_coeffs(gamma_integer[i] + gamma_integer[j] + gamma_half_offset[k])
        curl_z = (h_y[i, j, k] - h_y[i - 1, j, k]) - (h_x[i, j, k] - h_x[i, j - 1, k])
        inv_eps_z = inv_eps_r(x_node, y_node, z_node + half_dx)
        new_e_z = a_z * e_z[i, j, k] + b_z * dt_dx * inv_eps_z * curl_z
        e_z[i, j, k] = new_e_z


# The specialized kernels retain the fixed physical constants of the 128-node example.
HALF_WIDTH_128 = 4.25
N_GRID_128 = 128
DX_128 = 2.0 * HALF_WIDTH_128 / (N_GRID_128 - 1)
DT_128 = 0.95 * DX_128 / math.sqrt(3.0)


@wp.func_native(
    snippet="""
#if defined(__CUDA_ARCH__)
    return static_cast<wp::int32>(blockIdx.x * blockDim.x + threadIdx.x);
#else
    return -1;
#endif
"""
)
def native_linear_thread_index() -> int: ...


@wp.func
def loss_coeffs_128(gamma: float) -> tuple[float, float]:
    gamma_dt_half = 0.5 * DT_128 * gamma
    return ((1.0 - gamma_dt_half) / (1.0 + gamma_dt_half), 1.0 / (1.0 + gamma_dt_half))


@wp.kernel(grid_stride=False)
def update_h_128(
    e_x: wp.array3d[float],
    e_y: wp.array3d[float],
    e_z: wp.array3d[float],
    gamma_integer: wp.array[float],
    gamma_half_offset: wp.array[float],
    h_x: wp.array3d[float],
    h_y: wp.array3d[float],
    h_z: wp.array3d[float],
):
    packed = native_linear_thread_index()
    if packed < 0:
        launch_i, launch_j, launch_k = wp.tid()
        packed = (launch_i * N_GRID_128 + launch_j) * N_GRID_128 + launch_k
    if packed >= N_GRID_128 * N_GRID_128 * N_GRID_128 // 2:
        return
    linear = packed << 1
    i = packed >> 13
    j = (packed >> 6) & (N_GRID_128 - 1)
    k = (packed & (N_GRID_128 // 2 - 1)) << 1
    ex_flat = wp.array(ptr=e_x.ptr, shape=(N_GRID_128 * N_GRID_128 * N_GRID_128,), dtype=wp.float32)
    ey_flat = wp.array(ptr=e_y.ptr, shape=(N_GRID_128 * N_GRID_128 * N_GRID_128,), dtype=wp.float32)
    integer_flat = wp.array(ptr=gamma_integer.ptr, shape=(N_GRID_128,), dtype=wp.float32)
    half_flat = wp.array(ptr=gamma_half_offset.ptr, shape=(N_GRID_128,), dtype=wp.float32)
    dt_dx = DT_128 / DX_128
    last_staggered = N_GRID_128 - 2
    i_next_word = packed + wp.min(1, N_GRID_128 - 1 - i) * (N_GRID_128 * N_GRID_128 // 2)
    j_next_word = packed + wp.min(1, N_GRID_128 - 1 - j) * (N_GRID_128 // 2)
    k_next_scalar = linear + wp.min(2, N_GRID_128 - 1 - k)
    ex_words = wp.array(ptr=e_x.ptr, shape=(N_GRID_128 * N_GRID_128 * N_GRID_128 // 2,), dtype=wp.uint64)
    ey_words = wp.array(ptr=e_y.ptr, shape=(N_GRID_128 * N_GRID_128 * N_GRID_128 // 2,), dtype=wp.uint64)
    ez_words = wp.array(ptr=e_z.ptr, shape=(N_GRID_128 * N_GRID_128 * N_GRID_128 // 2,), dtype=wp.uint64)
    hx_words = wp.array(ptr=h_x.ptr, shape=(N_GRID_128 * N_GRID_128 * N_GRID_128 // 2,), dtype=wp.uint64)
    hy_words = wp.array(ptr=h_y.ptr, shape=(N_GRID_128 * N_GRID_128 * N_GRID_128 // 2,), dtype=wp.uint64)
    hz_words = wp.array(ptr=h_z.ptr, shape=(N_GRID_128 * N_GRID_128 * N_GRID_128 // 2,), dtype=wp.uint64)
    ex_center = wp.cast(ex_words[packed], wp.vec2)
    ey_center = wp.cast(ey_words[packed], wp.vec2)
    ez_center = wp.cast(ez_words[packed], wp.vec2)
    gamma_i = integer_flat[i]
    gamma_j = integer_flat[j]
    integer_words = wp.array(ptr=gamma_integer.ptr, shape=(N_GRID_128 // 2,), dtype=wp.uint64)
    half_words = wp.array(ptr=gamma_half_offset.ptr, shape=(N_GRID_128 // 2,), dtype=wp.uint64)
    gamma_k_pair = wp.cast(integer_words[k >> 1], wp.vec2)
    half_k_pair = wp.cast(half_words[k >> 1], wp.vec2)
    gamma_k_0 = gamma_k_pair[0]
    gamma_k_1 = gamma_k_pair[1]
    half_i = half_flat[i]
    half_j = half_flat[j]
    half_k_0 = half_k_pair[0]
    half_k_1 = half_k_pair[1]
    ez_j_next = wp.cast(ez_words[j_next_word], wp.vec2)
    ey_k_last = ey_flat[k_next_scalar]
    hx_old = wp.cast(hx_words[packed], wp.vec2)
    a_x_0, b_x_0 = loss_coeffs_128(gamma_i + half_j + half_k_0)
    a_x_1, b_x_1 = loss_coeffs_128(gamma_i + half_j + half_k_1)
    curl_x_0 = ez_j_next[0] - ez_center[0] - (ey_center[1] - ey_center[0])
    curl_x_1 = ez_j_next[1] - ez_center[1] - (ey_k_last - ey_center[1])
    hx_next_0 = a_x_0 * hx_old[0] - b_x_0 * dt_dx * curl_x_0
    hx_candidate_1 = a_x_1 * hx_old[1] - b_x_1 * dt_dx * curl_x_1
    hx_next_1 = hx_old[1]
    if k + 1 <= last_staggered:
        hx_next_1 = hx_candidate_1
    if j <= last_staggered:
        hx_words[packed] = wp.cast(wp.vec2(hx_next_0, hx_next_1), wp.uint64)
    ez_i_next = wp.cast(ez_words[i_next_word], wp.vec2)
    ex_k_last = ex_flat[k_next_scalar]
    hy_old = wp.cast(hy_words[packed], wp.vec2)
    a_y_0, b_y_0 = loss_coeffs_128(half_i + gamma_j + half_k_0)
    a_y_1, b_y_1 = loss_coeffs_128(half_i + gamma_j + half_k_1)
    curl_y_0 = ex_center[1] - ex_center[0] - (ez_i_next[0] - ez_center[0])
    curl_y_1 = ex_k_last - ex_center[1] - (ez_i_next[1] - ez_center[1])
    hy_next_0 = a_y_0 * hy_old[0] - b_y_0 * dt_dx * curl_y_0
    hy_candidate_1 = a_y_1 * hy_old[1] - b_y_1 * dt_dx * curl_y_1
    hy_next_1 = hy_old[1]
    if k + 1 <= last_staggered:
        hy_next_1 = hy_candidate_1
    if i <= last_staggered:
        hy_words[packed] = wp.cast(wp.vec2(hy_next_0, hy_next_1), wp.uint64)
    ey_i_next = wp.cast(ey_words[i_next_word], wp.vec2)
    ex_j_next = wp.cast(ex_words[j_next_word], wp.vec2)
    hz_old = wp.cast(hz_words[packed], wp.vec2)
    a_z_0, b_z_0 = loss_coeffs_128(half_i + half_j + gamma_k_0)
    a_z_1, b_z_1 = loss_coeffs_128(half_i + half_j + gamma_k_1)
    curl_z_0 = ey_i_next[0] - ey_center[0] - (ex_j_next[0] - ex_center[0])
    curl_z_1 = ey_i_next[1] - ey_center[1] - (ex_j_next[1] - ex_center[1])
    hz_next_0 = a_z_0 * hz_old[0] - b_z_0 * dt_dx * curl_z_0
    hz_next_1 = a_z_1 * hz_old[1] - b_z_1 * dt_dx * curl_z_1
    if i <= last_staggered and j <= last_staggered:
        hz_words[packed] = wp.cast(wp.vec2(hz_next_0, hz_next_1), wp.uint64)


@wp.kernel(grid_stride=False, launch_bounds=(256, 4), enable_cuda_smem_spilling=True)
def update_e_128(
    h_x: wp.array3d[float],
    h_y: wp.array3d[float],
    h_z: wp.array3d[float],
    gamma_integer: wp.array[float],
    gamma_half_offset: wp.array[float],
    timestep_counter: wp.array[wp.int32],
    e_x: wp.array3d[float],
    e_y: wp.array3d[float],
    e_z: wp.array3d[float],
):
    alignment_bits = (
        h_x.ptr | h_y.ptr | h_z.ptr | e_x.ptr | e_y.ptr | e_z.ptr | gamma_integer.ptr | gamma_half_offset.ptr
    )
    if h_x.shape[0] == 128 and (alignment_bits & wp.uint64(15)) == wp.uint64(0):
        aligned_launch_x, aligned_launch_y, aligned_launch_z = wp.tid()
        aligned_grid_size = h_x.shape[0]
        if aligned_launch_x >= aligned_grid_size // 8:
            return

        aligned_tile_index = (
            aligned_launch_x * aligned_grid_size + aligned_launch_y
        ) * aligned_grid_size + aligned_launch_z
        aligned_index_x = aligned_tile_index >> 11
        aligned_first_y = ((aligned_tile_index >> 5) & 63) << 1
        aligned_first_z = (aligned_tile_index & 31) << 2
        aligned_last = aligned_grid_size - 2
        if aligned_index_x > aligned_last:
            return

        aligned_dx = float(0.06692913385826772)
        aligned_dt = float(0.036709475777214395)
        aligned_dt_dx = float(0.5484827756881714)
        aligned_source_scale = float(0.24804306036139942)
        aligned_source_k = int(19)
        aligned_vector_z_size = aligned_grid_size // 4
        aligned_volume = aligned_grid_size * aligned_grid_size * aligned_grid_size
        aligned_vector_volume = aligned_volume // 4
        aligned_first_vector_index = (aligned_index_x * aligned_grid_size + aligned_first_y) * aligned_vector_z_size + (
            aligned_first_z >> 2
        )
        aligned_vector_h_x = wp.array(ptr=(h_x.ptr & ~wp.uint64(15)), shape=(aligned_vector_volume,), dtype=wp.vec4)
        aligned_vector_h_y = wp.array(ptr=(h_y.ptr & ~wp.uint64(15)), shape=(aligned_vector_volume,), dtype=wp.vec4)
        aligned_vector_h_z = wp.array(ptr=(h_z.ptr & ~wp.uint64(15)), shape=(aligned_vector_volume,), dtype=wp.vec4)
        aligned_vector_e_x = wp.array(ptr=(e_x.ptr & ~wp.uint64(15)), shape=(aligned_vector_volume,), dtype=wp.vec4)
        aligned_vector_e_y = wp.array(ptr=(e_y.ptr & ~wp.uint64(15)), shape=(aligned_vector_volume,), dtype=wp.vec4)
        aligned_vector_e_z = wp.array(ptr=(e_z.ptr & ~wp.uint64(15)), shape=(aligned_vector_volume,), dtype=wp.vec4)
        aligned_flat_h_x = wp.array(ptr=h_x.ptr, shape=(aligned_volume,), dtype=float)
        aligned_flat_h_y = wp.array(ptr=h_y.ptr, shape=(aligned_volume,), dtype=float)
        aligned_node_profile = wp.array(ptr=gamma_integer.ptr, shape=(aligned_grid_size,), dtype=float)
        aligned_half_profile = wp.array(ptr=gamma_half_offset.ptr, shape=(aligned_grid_size,), dtype=float)
        aligned_vector_node_profile = wp.array(
            ptr=(gamma_integer.ptr & ~wp.uint64(15)), shape=(aligned_vector_z_size,), dtype=wp.vec4
        )
        aligned_vector_half_profile = wp.array(
            ptr=(gamma_half_offset.ptr & ~wp.uint64(15)), shape=(aligned_vector_z_size,), dtype=wp.vec4
        )

        aligned_previous_vector_y = aligned_first_vector_index - wp.where(aligned_first_y > 0, aligned_vector_z_size, 0)
        aligned_h_x_previous_y = aligned_vector_h_x[aligned_previous_vector_y]
        aligned_h_z_previous_y = aligned_vector_h_z[aligned_previous_vector_y]
        aligned_gamma_node_x = aligned_node_profile[aligned_index_x]
        aligned_gamma_half_x = aligned_half_profile[aligned_index_x]
        aligned_gamma_nodes_z = aligned_vector_node_profile[aligned_first_z >> 2]
        aligned_gamma_halves_z = aligned_vector_half_profile[aligned_first_z >> 2]
        aligned_half_dt = 0.5 * aligned_dt
        aligned_half_dx = 0.5 * aligned_dx
        aligned_x_node = float(aligned_index_x) * aligned_dx - 4.25
        aligned_x_half = aligned_x_node + aligned_half_dx
        aligned_source_ij = (aligned_grid_size - 1) // 2

        for aligned_row_offset in range(2):
            aligned_row = wp.static(aligned_row_offset)
            aligned_index_y = aligned_first_y + aligned_row
            if aligned_index_y <= aligned_last:
                aligned_vector_index = aligned_first_vector_index + aligned_row * aligned_vector_z_size
                aligned_previous_vector_x = aligned_vector_index - wp.where(
                    aligned_index_x > 0, aligned_grid_size * aligned_vector_z_size, 0
                )
                aligned_previous_scalar_z = (aligned_vector_index << 2) - wp.where(aligned_first_z > 0, 1, 0)
                aligned_h_x_values = aligned_vector_h_x[aligned_vector_index]
                aligned_h_y_values = aligned_vector_h_y[aligned_vector_index]
                aligned_h_z_values = aligned_vector_h_z[aligned_vector_index]
                aligned_h_y_previous_x = aligned_vector_h_y[aligned_previous_vector_x]
                aligned_h_z_previous_x = aligned_vector_h_z[aligned_previous_vector_x]
                aligned_h_x_previous_z = aligned_flat_h_x[aligned_previous_scalar_z]
                aligned_h_y_previous_z = aligned_flat_h_y[aligned_previous_scalar_z]
                aligned_e_x_values = aligned_vector_e_x[aligned_vector_index]
                aligned_e_y_values = aligned_vector_e_y[aligned_vector_index]
                aligned_e_z_values = aligned_vector_e_z[aligned_vector_index]

                aligned_curl_x_values = wp.vec4()
                aligned_curl_y_values = wp.vec4()
                aligned_curl_z_values = (
                    aligned_h_y_values - aligned_h_y_previous_x - (aligned_h_x_values - aligned_h_x_previous_y)
                )
                for aligned_curl_offset in range(4):
                    aligned_curl_cell = wp.static(aligned_curl_offset)
                    aligned_curl_x_values[aligned_curl_cell] = (
                        aligned_h_z_values[aligned_curl_cell]
                        - aligned_h_z_previous_y[aligned_curl_cell]
                        - (aligned_h_y_values[aligned_curl_cell] - aligned_h_y_previous_z)
                    )
                    aligned_curl_y_values[aligned_curl_cell] = (
                        aligned_h_x_values[aligned_curl_cell]
                        - aligned_h_x_previous_z
                        - (aligned_h_z_values[aligned_curl_cell] - aligned_h_z_previous_x[aligned_curl_cell])
                    )
                    aligned_h_x_previous_z = aligned_h_x_values[aligned_curl_cell]
                    aligned_h_y_previous_z = aligned_h_y_values[aligned_curl_cell]

                aligned_gamma_node_y = aligned_node_profile[aligned_index_y]
                aligned_gamma_half_y = aligned_half_profile[aligned_index_y]
                aligned_gamma_xy_x = aligned_gamma_half_x + aligned_gamma_node_y
                aligned_gamma_xy_y = aligned_gamma_node_x + aligned_gamma_half_y
                aligned_gamma_xy_z = aligned_gamma_node_x + aligned_gamma_node_y
                aligned_y_node = float(aligned_index_y) * aligned_dx - 4.25
                aligned_y_half = aligned_y_node + aligned_half_dx
                aligned_radius_xy_x = aligned_x_half * aligned_x_half + aligned_y_node * aligned_y_node
                aligned_radius_xy_y = aligned_x_node * aligned_x_node + aligned_y_half * aligned_y_half
                aligned_radius_xy_z = aligned_x_node * aligned_x_node + aligned_y_node * aligned_y_node

                aligned_is_source_vector = (
                    aligned_index_x == aligned_source_ij
                    and aligned_index_y == aligned_source_ij
                    and aligned_first_z <= aligned_source_k
                    and aligned_source_k < aligned_first_z + 4
                )
                aligned_source = float(0.0)
                if aligned_is_source_vector:
                    aligned_source_time = (float(timestep_counter[0]) + 0.5) * aligned_dt
                    aligned_source = (
                        -aligned_source_scale
                        * wp.sin(2.0 * wp.pi * aligned_source_time)
                        * (1.0 - wp.exp(-((aligned_source_time / 3.0) ** 2.0)))
                    )

                for aligned_cell_offset in range(4):
                    aligned_cell = wp.static(aligned_cell_offset)
                    aligned_index_z = aligned_first_z + aligned_cell
                    if aligned_index_z <= aligned_last:
                        aligned_gamma_dt_x = aligned_half_dt * (
                            aligned_gamma_xy_x + aligned_gamma_nodes_z[aligned_cell]
                        )
                        aligned_gamma_dt_y = aligned_half_dt * (
                            aligned_gamma_xy_y + aligned_gamma_nodes_z[aligned_cell]
                        )
                        aligned_gamma_dt_z = aligned_half_dt * (
                            aligned_gamma_xy_z + aligned_gamma_halves_z[aligned_cell]
                        )
                        aligned_b_x = wp.div_approx(1.0, 1.0 + aligned_gamma_dt_x)
                        aligned_b_y = wp.div_approx(1.0, 1.0 + aligned_gamma_dt_y)
                        aligned_b_z = wp.div_approx(1.0, 1.0 + aligned_gamma_dt_z)
                        aligned_a_x = (1.0 - aligned_gamma_dt_x) * aligned_b_x
                        aligned_a_y = (1.0 - aligned_gamma_dt_y) * aligned_b_y
                        aligned_a_z = (1.0 - aligned_gamma_dt_z) * aligned_b_z

                        aligned_z_node = float(aligned_index_z) * aligned_dx - 4.25
                        aligned_z_half = aligned_z_node + aligned_half_dx
                        aligned_radius_x = aligned_radius_xy_x + aligned_z_node * aligned_z_node
                        aligned_radius_y = aligned_radius_xy_y + aligned_z_node * aligned_z_node
                        aligned_radius_z = aligned_radius_xy_z + aligned_z_half * aligned_z_half
                        aligned_inv_eps_x = float(1.0)
                        aligned_inv_eps_y = float(1.0)
                        aligned_inv_eps_z = float(1.0)
                        if aligned_radius_x < 9.0:
                            aligned_inv_eps_x = wp.div_approx(1.0, 2.0 - aligned_radius_x / 9.0)
                        if aligned_radius_y < 9.0:
                            aligned_inv_eps_y = wp.div_approx(1.0, 2.0 - aligned_radius_y / 9.0)
                        if aligned_radius_z < 9.0:
                            aligned_inv_eps_z = wp.div_approx(1.0, 2.0 - aligned_radius_z / 9.0)

                        aligned_curl_x = aligned_curl_x_values[aligned_cell]
                        aligned_curl_y = aligned_curl_y_values[aligned_cell]
                        aligned_curl_z = aligned_curl_z_values[aligned_cell]
                        if aligned_is_source_vector and aligned_index_z == aligned_source_k:
                            aligned_curl_x -= aligned_source
                        if aligned_index_y >= 1 and aligned_index_z >= 1:
                            aligned_e_x_values[aligned_cell] = (
                                aligned_a_x * aligned_e_x_values[aligned_cell]
                                + aligned_b_x * aligned_dt_dx * aligned_inv_eps_x * aligned_curl_x
                            )
                        if aligned_index_x >= 1 and aligned_index_z >= 1:
                            aligned_e_y_values[aligned_cell] = (
                                aligned_a_y * aligned_e_y_values[aligned_cell]
                                + aligned_b_y * aligned_dt_dx * aligned_inv_eps_y * aligned_curl_y
                            )
                        if aligned_index_x >= 1 and aligned_index_y >= 1:
                            aligned_e_z_values[aligned_cell] = (
                                aligned_a_z * aligned_e_z_values[aligned_cell]
                                + aligned_b_z * aligned_dt_dx * aligned_inv_eps_z * aligned_curl_z
                            )

                if aligned_index_y >= 1:
                    aligned_vector_e_x[aligned_vector_index] = aligned_e_x_values
                if aligned_index_x >= 1:
                    aligned_vector_e_y[aligned_vector_index] = aligned_e_y_values
                if aligned_index_x >= 1 and aligned_index_y >= 1:
                    aligned_vector_e_z[aligned_vector_index] = aligned_e_z_values
                aligned_h_x_previous_y = aligned_h_x_values
                aligned_h_z_previous_y = aligned_h_z_values
    else:
        launch_x, launch_y, launch_z = wp.tid()
        grid_size = h_x.shape[0]
        if launch_x >= grid_size // 2:
            return

        vector_index = (launch_x * grid_size + launch_y) * grid_size + launch_z
        vector_row_shift = int(5)
        vector_plane_shift = int(11)
        vector_z_mask = int(31)
        grid_mask = int(63)
        dx = float(0.1349206349206349)
        dt = float(0.0740016416461306)
        dt_dx = float(0.5484827160835266)
        source_scale = float(0.06103806228373703)
        source_k = int(9)
        if grid_size == 128:
            vector_row_shift = 6
            vector_plane_shift = 13
            vector_z_mask = 63
            grid_mask = 127
            dx = 0.06692913385826772
            dt = 0.036709475777214395
            dt_dx = 0.5484827756881714
            source_scale = 0.24804306036139942
            source_k = 19

        index_x = vector_index >> vector_plane_shift
        index_y = (vector_index >> vector_row_shift) & grid_mask
        first_z = (vector_index & vector_z_mask) << 1
        last = grid_size - 2
        if index_x > last or index_y > last:
            return

        vector_z_size = grid_size // 2
        volume = grid_size * grid_size * grid_size
        vector_volume = volume // 2
        vector_h_x = wp.array(ptr=h_x.ptr, shape=(vector_volume,), dtype=wp.uint64)
        vector_h_y = wp.array(ptr=h_y.ptr, shape=(vector_volume,), dtype=wp.uint64)
        vector_h_z = wp.array(ptr=h_z.ptr, shape=(vector_volume,), dtype=wp.uint64)
        vector_e_x = wp.array(ptr=e_x.ptr, shape=(vector_volume,), dtype=wp.uint64)
        vector_e_y = wp.array(ptr=e_y.ptr, shape=(vector_volume,), dtype=wp.uint64)
        vector_e_z = wp.array(ptr=e_z.ptr, shape=(vector_volume,), dtype=wp.uint64)
        flat_h_x = wp.array(ptr=h_x.ptr, shape=(volume,), dtype=float)
        flat_h_y = wp.array(ptr=h_y.ptr, shape=(volume,), dtype=float)
        node_profile = wp.array(ptr=gamma_integer.ptr, shape=(grid_size,), dtype=float)
        half_profile = wp.array(ptr=gamma_half_offset.ptr, shape=(grid_size,), dtype=float)
        vector_node_profile = wp.array(ptr=gamma_integer.ptr, shape=(vector_z_size,), dtype=wp.uint64)
        vector_half_profile = wp.array(ptr=gamma_half_offset.ptr, shape=(vector_z_size,), dtype=wp.uint64)

        previous_vector_x = vector_index - wp.where(index_x > 0, grid_size * vector_z_size, 0)
        previous_vector_y = vector_index - wp.where(index_y > 0, vector_z_size, 0)
        previous_scalar_z = (vector_index << 1) - wp.where(first_z > 0, 1, 0)
        h_x_values = wp.cast(vector_h_x[vector_index], wp.vec2)
        h_y_values = wp.cast(vector_h_y[vector_index], wp.vec2)
        h_z_values = wp.cast(vector_h_z[vector_index], wp.vec2)
        h_x_previous_y = wp.cast(vector_h_x[previous_vector_y], wp.vec2)
        h_y_previous_x = wp.cast(vector_h_y[previous_vector_x], wp.vec2)
        h_z_previous_x = wp.cast(vector_h_z[previous_vector_x], wp.vec2)
        h_z_previous_y = wp.cast(vector_h_z[previous_vector_y], wp.vec2)
        h_x_previous_z = flat_h_x[previous_scalar_z]
        h_y_previous_z = flat_h_y[previous_scalar_z]
        e_x_values = wp.cast(vector_e_x[vector_index], wp.vec2)
        e_y_values = wp.cast(vector_e_y[vector_index], wp.vec2)
        e_z_values = wp.cast(vector_e_z[vector_index], wp.vec2)

        gamma_node_x = node_profile[index_x]
        gamma_node_y = node_profile[index_y]
        gamma_half_x = half_profile[index_x]
        gamma_half_y = half_profile[index_y]
        gamma_nodes_z = wp.cast(vector_node_profile[first_z >> 1], wp.vec2)
        gamma_halves_z = wp.cast(vector_half_profile[first_z >> 1], wp.vec2)
        gamma_xy_x = gamma_half_x + gamma_node_y
        gamma_xy_y = gamma_node_x + gamma_half_y
        gamma_xy_z = gamma_node_x + gamma_node_y
        half_dt = 0.5 * dt
        half_dx = 0.5 * dx
        x_node = float(index_x) * dx - 4.25
        y_node = float(index_y) * dx - 4.25
        x_half = x_node + half_dx
        y_half = y_node + half_dx
        radius_xy_x = x_half * x_half + y_node * y_node
        radius_xy_y = x_node * x_node + y_half * y_half
        radius_xy_z = x_node * x_node + y_node * y_node

        source_ij = (grid_size - 1) // 2
        is_source_vector = (
            index_x == source_ij and index_y == source_ij and first_z <= source_k and source_k < first_z + 2
        )
        source = float(0.0)
        if is_source_vector:
            source_time = (float(timestep_counter[0]) + 0.5) * dt
            source = -source_scale * wp.sin(2.0 * wp.pi * source_time) * (1.0 - wp.exp(-((source_time / 3.0) ** 2.0)))

        for cell_offset in range(2):
            cell = wp.static(cell_offset)
            index_z = first_z + cell
            if index_z <= last:
                gamma_dt_x = half_dt * (gamma_xy_x + gamma_nodes_z[cell])
                gamma_dt_y = half_dt * (gamma_xy_y + gamma_nodes_z[cell])
                gamma_dt_z = half_dt * (gamma_xy_z + gamma_halves_z[cell])
                b_x = wp.div_approx(1.0, 1.0 + gamma_dt_x)
                b_y = wp.div_approx(1.0, 1.0 + gamma_dt_y)
                b_z = wp.div_approx(1.0, 1.0 + gamma_dt_z)
                a_x = (1.0 - gamma_dt_x) * b_x
                a_y = (1.0 - gamma_dt_y) * b_y
                a_z = (1.0 - gamma_dt_z) * b_z

                z_node = float(index_z) * dx - 4.25
                z_half = z_node + half_dx
                radius_x = radius_xy_x + z_node * z_node
                radius_y = radius_xy_y + z_node * z_node
                radius_z = radius_xy_z + z_half * z_half
                inv_eps_x = float(1.0)
                inv_eps_y = float(1.0)
                inv_eps_z = float(1.0)
                if radius_x < 9.0:
                    inv_eps_x = wp.div_approx(1.0, 2.0 - radius_x / 9.0)
                if radius_y < 9.0:
                    inv_eps_y = wp.div_approx(1.0, 2.0 - radius_y / 9.0)
                if radius_z < 9.0:
                    inv_eps_z = wp.div_approx(1.0, 2.0 - radius_z / 9.0)

                curl_x = h_z_values[cell] - h_z_previous_y[cell] - (h_y_values[cell] - h_y_previous_z)
                curl_y = h_x_values[cell] - h_x_previous_z - (h_z_values[cell] - h_z_previous_x[cell])
                curl_z = h_y_values[cell] - h_y_previous_x[cell] - (h_x_values[cell] - h_x_previous_y[cell])
                if is_source_vector and index_z == source_k:
                    curl_x -= source
                if index_y >= 1 and index_z >= 1:
                    e_x_values[cell] = a_x * e_x_values[cell] + b_x * dt_dx * inv_eps_x * curl_x
                if index_x >= 1 and index_z >= 1:
                    e_y_values[cell] = a_y * e_y_values[cell] + b_y * dt_dx * inv_eps_y * curl_y
                if index_x >= 1 and index_y >= 1:
                    e_z_values[cell] = a_z * e_z_values[cell] + b_z * dt_dx * inv_eps_z * curl_z
            h_x_previous_z = h_x_values[cell]
            h_y_previous_z = h_y_values[cell]

        if index_y >= 1:
            vector_e_x[vector_index] = wp.cast(e_x_values, wp.uint64)
        if index_x >= 1:
            vector_e_y[vector_index] = wp.cast(e_y_values, wp.uint64)
        if index_x >= 1 and index_y >= 1:
            vector_e_z[vector_index] = wp.cast(e_z_values, wp.uint64)


@wp.kernel
def compute_em_energy(
    e_x: wp.array3d[float],
    e_y: wp.array3d[float],
    e_z: wp.array3d[float],
    h_x: wp.array3d[float],
    h_y: wp.array3d[float],
    h_z: wp.array3d[float],
    electric_energy_sum: wp.array[float],
    magnetic_energy_sum: wp.array[float],
):
    """Accumulate ``eps_r * E^2`` and ``H^2`` over the Yee grid."""
    i, j, k = wp.tid()

    x_node = float(i) * DX - HALF_WIDTH
    y_node = float(j) * DX - HALF_WIDTH
    z_node = float(k) * DX - HALF_WIDTH
    half_dx = 0.5 * DX

    # Note: DO NOT interpret `electric_energy_density` below as the energy density at a point `i, j, k` in space.
    # The ``compute_em_energy`` integrates each component over the entire domain at its own respective Yee cell locations.
    # It is valid here to do so because we are ultimately reporting a single summed electrical and magnetic energy value
    # over the entire domain.
    electric_energy_density = (
        e_x[i, j, k] * e_x[i, j, k] / inv_eps_r(x_node + half_dx, y_node, z_node)
        + e_y[i, j, k] * e_y[i, j, k] / inv_eps_r(x_node, y_node + half_dx, z_node)
        + e_z[i, j, k] * e_z[i, j, k] / inv_eps_r(x_node, y_node, z_node + half_dx)
    )
    magnetic_energy_density = h_x[i, j, k] * h_x[i, j, k] + h_y[i, j, k] * h_y[i, j, k] + h_z[i, j, k] * h_z[i, j, k]

    # `tile_sum` reduces per-thread values to one sum per block. Each
    # `wp.tile_atomic_add` then issues one global atomic per block instead
    # of one global atomic per thread (which would have been the case with wp.atomic_add()).
    electric_tile_sum = wp.tile_sum(wp.tile(electric_energy_density))
    magnetic_tile_sum = wp.tile_sum(wp.tile(magnetic_energy_density))
    wp.tile_atomic_add(electric_energy_sum, electric_tile_sum)
    wp.tile_atomic_add(magnetic_energy_sum, magnetic_tile_sum)


@wp.kernel
def increment_timestep(timestep_counter: wp.array[wp.int32]):
    timestep_counter[0] = timestep_counter[0] + 1


@wp.kernel
def extract_e_x_center_plane(e_x: wp.array3d[float], y_index: int, e_x_plane: wp.array2d[float]):
    i, k = wp.tid()
    e_x_plane[i, k] = e_x[i, y_index, k]


def build_loss_profiles() -> tuple[wp.array[float], wp.array[float]]:
    """Build the 1-D matched-loss rate profiles at grid nodes and half-cell offsets.

    The loss rate is additive per axis, ``gamma_q(d) = GAMMA_MAX * (d / L_ABSORBER)^3``,
    where ``d`` runs from zero at the inner edge of the attenuation layer to
    ``L_ABSORBER`` at the wall. The grid is cubic and uniform, so the three axes share the
    same two profiles.
    """

    def gamma_profile(coords: np.ndarray) -> np.ndarray:
        depth = np.clip((np.abs(coords) - (HALF_WIDTH - L_ABSORBER)) / L_ABSORBER, 0.0, 1.0)
        return GAMMA_MAX * depth**3

    node_coords = (np.arange(N_GRID) - (N_GRID - 1) / 2.0) * DX
    half_coords = node_coords + DX / 2.0

    gamma_integer = wp.array(gamma_profile(node_coords).astype(np.float32), dtype=float)
    gamma_half_offset = wp.array(gamma_profile(half_coords).astype(np.float32), dtype=float)
    return gamma_integer, gamma_half_offset


def _use_optimized_128(example) -> bool:
    """Select the CUDA specialization only for its fixed physical configuration."""
    expected_dx = 8.5 / 127
    expected_dt = 0.95 * expected_dx / math.sqrt(3.0)
    if not (
        N_GRID == N_GRID_128 == 128
        and HALF_WIDTH == HALF_WIDTH_128 == 4.25
        and R_LENS == 3.0
        and E0 == 1.0
        and DX_REF == 1.0 / 30.0
        and SOURCE_RAMP == 3.0
        and DX == DX_128 == expected_dx
        and DT == DT_128 == expected_dt
        and SOURCE_SCALE == ((1.0 / 30.0) / expected_dx) ** 2
        and SOURCE_IJ == 63
        and SOURCE_K == 19
    ):
        return False
    fields = (example.e_x, example.e_y, example.e_z, example.h_x, example.h_y, example.h_z)
    profiles = (example.gamma_integer, example.gamma_half_offset)
    arrays = fields + profiles
    device = fields[0].device
    if not device.is_cuda:
        return False
    for array in arrays:
        if array.device != device or array.dtype != wp.float32 or not array.is_contiguous or array.ptr % 16:
            return False
    if any(array.shape != (128, 128, 128) for array in fields):
        return False
    if any(array.shape != (128,) for array in profiles):
        return False
    # The native constructor owns separate allocations; reject substituted aliases.
    spans = sorted((array.ptr, array.ptr + array.size * 4) for array in arrays)
    return all(left[1] <= right[0] for left, right in itertools.pairwise(spans))


class Example:
    """FDTD solver for a point source collimated by a Luneburg lens.

    The six field components are stored as ``(N_GRID, N_GRID, N_GRID)`` fp32 arrays.
    Simulation parameters (``N_GRID``, ``DX``, ``DT``, etc.) are defined as
    module-level constants and captured directly by Warp kernels.
    """

    def __init__(self) -> None:
        self.steps_per_frame = 1  # maximum timesteps per rendered frame

        # use CUDA graph if GPU is available
        self.use_cuda_graph = wp.get_device().is_cuda
        self.graph = None

        field_shape = (N_GRID, N_GRID, N_GRID)
        self.e_x = wp.zeros(field_shape, dtype=float)
        self.e_y = wp.zeros(field_shape, dtype=float)
        self.e_z = wp.zeros(field_shape, dtype=float)
        self.h_x = wp.zeros(field_shape, dtype=float)
        self.h_y = wp.zeros(field_shape, dtype=float)
        self.h_z = wp.zeros(field_shape, dtype=float)
        self.e_x_center_plane_buffer = wp.empty((N_GRID, N_GRID), dtype=float)

        self.electric_energy_sum = wp.zeros(1, dtype=float)
        self.magnetic_energy_sum = wp.zeros(1, dtype=float)

        # global timestep index counter on device to make CUDA graph capture possible
        self.timestep_counter = wp.zeros(1, dtype=wp.int32)

        # the three axes index the same integer and half-offset profiles for a cubic uniform grid
        self.gamma_integer, self.gamma_half_offset = build_loss_profiles()

        optimized = _use_optimized_128(self)
        self._update_h = update_h_128 if optimized else update_h
        self._update_e = update_e_128 if optimized else update_e

        points_per_wavelength = (N_GRID - 1) / (2.0 * HALF_WIDTH)
        print(
            f"FDTD Luneburg lens: {N_GRID}^3 grid, dx = {DX:.6f} "
            f"({points_per_wavelength:.1f} points per wavelength), dt = {DT:.6f}"
        )

    def _check_em_energy(self, timestep: int) -> None:
        """Compute and report electromagnetic energy, raising if it is not finite."""
        self.electric_energy_sum.zero_()
        self.magnetic_energy_sum.zero_()
        wp.launch(
            compute_em_energy,
            dim=(N_GRID, N_GRID, N_GRID),
            inputs=[self.e_x, self.e_y, self.e_z, self.h_x, self.h_y, self.h_z],
            outputs=[self.electric_energy_sum, self.magnetic_energy_sum],
        )

        w_e = 0.5 * DX**3 * float(self.electric_energy_sum.numpy()[0])
        w_h = 0.5 * DX**3 * float(self.magnetic_energy_sum.numpy()[0])
        if not (np.isfinite(w_e) and np.isfinite(w_h)):
            raise RuntimeError(f"Non-finite electromagnetic energy at timestep {timestep}.")
        print(f"step {timestep}, t = {timestep * DT:.6f}, W_E = {w_e:.6e}, W_H = {w_h:.6e}", flush=True)

    def step(self) -> None:
        """Advance ``H`` from curl ``E``, then ``E`` from curl ``H`` in leapfrog fashion."""
        wp.launch(
            self._update_h,
            dim=(N_GRID, N_GRID, N_GRID),
            inputs=[self.e_x, self.e_y, self.e_z, self.gamma_integer, self.gamma_half_offset],
            outputs=[self.h_x, self.h_y, self.h_z],
        )

        wp.launch(
            self._update_e,
            dim=(N_GRID, N_GRID, N_GRID),
            inputs=[
                self.h_x,
                self.h_y,
                self.h_z,
                self.gamma_integer,
                self.gamma_half_offset,
                self.timestep_counter,
            ],
            outputs=[self.e_x, self.e_y, self.e_z],
        )
        # Advance the device-side timestep counter after an electric-field update.
        wp.launch(increment_timestep, dim=1, inputs=[self.timestep_counter])

    def e_x_center_plane(self) -> np.ndarray:
        """Return ``E_x`` on the x-z plane closest to y = 0."""
        j_center = (N_GRID - 1) // 2
        wp.launch(
            extract_e_x_center_plane,
            dim=(N_GRID, N_GRID),
            inputs=[self.e_x, j_center],
            outputs=[self.e_x_center_plane_buffer],
        )
        return self.e_x_center_plane_buffer.numpy()

    def step_and_render_frame(self, frame_num: int, img=None, num_steps: int | None = None) -> tuple:
        """Advance up to ``steps_per_frame`` timesteps and update the Matplotlib image."""
        first_timestep = frame_num * self.steps_per_frame
        steps_this_frame = self.steps_per_frame
        if num_steps is not None:
            steps_this_frame = min(steps_this_frame, num_steps - first_timestep)
        self.timestep_counter.fill_(first_timestep)

        for substep in range(steps_this_frame):
            timestep_index = first_timestep + substep
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.step()
            if (timestep_index + 1) % DIAG_INTERVAL == 0:
                self._check_em_energy(timestep_index + 1)

        if img is not None:
            sim_timestep = first_timestep + steps_this_frame
            img.set_array(self.e_x_center_plane())
            img.axes.set_title(f"Luneburg lens, {N_GRID}$^3$, $E_x$ at timestep {sim_timestep}")

        return (img,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="3-D FDTD simulation of a point source collimated by a Luneburg lens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the default Warp device.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=N_GRID,
        help="Grid resolution: nodes per axis across the fixed 8.5-wavelength domain. "
        "Refining the grid leaves the physical extent, lens, and attenuation-layer depth unchanged.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Total number of leapfrog timesteps. Defaults to unlimited in interactive mode and to "
        "SIM_DURATION source periods, rounded to whole steps, in headless mode.",
    )
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=20,
        help="Maximum number of leapfrog timesteps per rendered frame.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    if args.steps_per_frame < 1:
        parser.error(f"--steps-per-frame must be at least 1, got {args.steps_per_frame}")
    if args.resolution < 48:
        parser.error(
            f"--resolution {args.resolution} is too coarse: below 48 nodes per axis the absorbing "
            f"layer spans fewer than four cells and the grid falls under six points per wavelength."
        )

    # Rebind the resolution-derived module constants before the first kernel launch.
    N_GRID = args.resolution
    DX = 2.0 * HALF_WIDTH / (N_GRID - 1)
    DT = 0.95 * DX / math.sqrt(3.0)
    SOURCE_SCALE = (DX_REF / DX) ** 2
    SOURCE_IJ = (N_GRID - 1) // 2
    SOURCE_K = round((HALF_WIDTH - R_LENS) / DX)

    # dt tracks dx, so the headless default step count derives from a fixed physical duration
    num_steps = args.num_steps
    if num_steps is None and args.headless:
        num_steps = round(SIM_DURATION / DT)

    with wp.ScopedDevice(args.device):
        example = Example()
        example.steps_per_frame = args.steps_per_frame

        if example.use_cuda_graph:
            # capture one leapfrog timestep in a CUDA graph
            with wp.ScopedCapture() as capture:
                example.step()
            example.graph = capture.graph

        if args.headless:
            import time

            num_frames = (num_steps + args.steps_per_frame - 1) // args.steps_per_frame

            wp.synchronize_device()
            start_time = time.perf_counter()

            for i in range(num_frames):
                example.step_and_render_frame(frame_num=i, num_steps=num_steps)

            wp.synchronize_device()
            stop_time = time.perf_counter()

            elapsed = stop_time - start_time
            mcells_per_s = num_steps * N_GRID**3 / elapsed / 1e6
            print(f"{num_steps} timesteps on a {N_GRID}^3 grid: {elapsed:.3f} s, {mcells_per_s:.1f} Mcell-updates/s")
        else:
            import matplotlib.animation as anim
            import matplotlib.pyplot as plt

            num_frames = (
                (num_steps + args.steps_per_frame - 1) // args.steps_per_frame if num_steps is not None else None
            )

            fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
            ax.set_title(f"Luneburg lens, {N_GRID}$^3$, $E_x$ at timestep 0")
            ax.set_xlabel("z")
            ax.set_ylabel("x")

            img = ax.imshow(
                example.e_x_center_plane(),
                origin="lower",
                cmap="seismic",
                interpolation="antialiased",
                vmin=-PLOT_E_SCALE,
                vmax=PLOT_E_SCALE,
                extent=[-HALF_WIDTH, HALF_WIDTH, -HALF_WIDTH, HALF_WIDTH],
            )
            fig.colorbar(img, ax=ax, label="$E_x$")
            ax.add_patch(plt.Circle((0.0, 0.0), R_LENS, fill=False, color="0.3", lw=0.8, ls="--"))

            seq = anim.FuncAnimation(
                fig,
                example.step_and_render_frame,
                fargs=(img, num_steps),
                frames=num_frames,
                init_func=lambda: (),  # without this, Matplotlib delivers frame 0 twice and replays the source window
                blit=False,
                interval=1,
                repeat=False,
                cache_frame_data=False,
            )

            plt.show()
