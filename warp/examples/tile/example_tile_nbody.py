# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example N-Body
#
# Shows how to simulate an N-Body gravitational problem using an all-pairs
# approach with Warp tile primitives.
#
# References:
#   L. Nyland, M. Harris, and J. Prins. "Fast N-Body Simulation with
#   CUDA" in GPU Gems 3. H. Nguyen, Addison-Wesley Professional, 2007.
#   https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
#
###########################################################################

import argparse

import numpy as np

import warp as wp

wp.init()

DT = wp.constant(0.016)
SOFTENING_SQ = wp.constant(0.1**2)  # Softening factor for numerical stability
TILE_SIZE = wp.constant(64)
PARTICLE_MASS = wp.constant(1.0)


@wp.func
def body_body_interaction(p0: wp.vec3, pi: wp.vec3):
    """Return the acceleration of the particle at position `p0` due to the
    particle at position `pi`."""
    r = pi - p0

    dist_sq = wp.length_sq(r) + SOFTENING_SQ

    inv_dist = 1.0 / wp.sqrt(dist_sq)
    inv_dist_cubed = inv_dist * inv_dist * inv_dist

    acc = PARTICLE_MASS * inv_dist_cubed * r

    return acc


@wp.kernel
def integrate_bodies_tiled(
    old_position: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    new_position: wp.array[wp.vec3],
    num_bodies: int,
):
    i = wp.tid()

    p0 = old_position[i]

    accel = wp.vec3(0.0, 0.0, 0.0)

    for k in range(num_bodies / TILE_SIZE):
        k_tile = wp.tile_load(old_position, shape=TILE_SIZE, offset=k * TILE_SIZE)
        for idx in range(TILE_SIZE):
            pi = k_tile[idx]
            accel += body_body_interaction(p0, pi)

    # Advance the velocity one timestep (in-place)
    velocity[i] = velocity[i] + accel * DT

    # Advance the positions (using a second array)
    new_position[i] = old_position[i] + DT * velocity[i]


# Reuse a shared source tile across 64 targets and expose 32 independent
# interaction calculations before accumulating them in the original order.
SOURCE_TILE_SIZE = wp.constant(256)


# Keep the generated compilation options local to this kernel so that the
# original generic kernel retains its compilation and differentiation behavior.
@wp.kernel(
    launch_bounds=(64, 4),
    module="unique",
    module_options={"block_dim": 64, "max_unroll": 1, "enable_backward": False, "fast_math": False},
)
def _integrate_bodies_tiled_optimized(
    old_position: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    new_position: wp.array[wp.vec3],
    num_bodies: int,
):
    body_index = wp.tid()
    position = old_position[body_index]
    acceleration = wp.vec3(0.0, 0.0, 0.0)

    source_tiles = num_bodies / TILE_SIZE
    for tile_index in range((source_tiles + 3) / 4):
        positions = wp.tile_load(
            old_position, shape=SOURCE_TILE_SIZE, offset=tile_index * SOURCE_TILE_SIZE, storage="shared"
        )
        source_groups = wp.min(4, source_tiles - tile_index * 4) * 2
        for group_index in range(source_groups):
            source_index = group_index * 32
            displacement0 = positions[source_index] - position
            displacement1 = positions[source_index + 1] - position
            displacement2 = positions[source_index + 2] - position
            displacement3 = positions[source_index + 3] - position
            displacement4 = positions[source_index + 4] - position
            displacement5 = positions[source_index + 5] - position
            displacement6 = positions[source_index + 6] - position
            displacement7 = positions[source_index + 7] - position
            displacement8 = positions[source_index + 8] - position
            displacement9 = positions[source_index + 9] - position
            displacement10 = positions[source_index + 10] - position
            displacement11 = positions[source_index + 11] - position
            displacement12 = positions[source_index + 12] - position
            displacement13 = positions[source_index + 13] - position
            displacement14 = positions[source_index + 14] - position
            displacement15 = positions[source_index + 15] - position
            displacement16 = positions[source_index + 16] - position
            displacement17 = positions[source_index + 17] - position
            displacement18 = positions[source_index + 18] - position
            displacement19 = positions[source_index + 19] - position
            displacement20 = positions[source_index + 20] - position
            displacement21 = positions[source_index + 21] - position
            displacement22 = positions[source_index + 22] - position
            displacement23 = positions[source_index + 23] - position
            displacement24 = positions[source_index + 24] - position
            displacement25 = positions[source_index + 25] - position
            displacement26 = positions[source_index + 26] - position
            displacement27 = positions[source_index + 27] - position
            displacement28 = positions[source_index + 28] - position
            displacement29 = positions[source_index + 29] - position
            displacement30 = positions[source_index + 30] - position
            displacement31 = positions[source_index + 31] - position
            distance0 = wp.length_sq(displacement0) + SOFTENING_SQ
            distance1 = wp.length_sq(displacement1) + SOFTENING_SQ
            distance2 = wp.length_sq(displacement2) + SOFTENING_SQ
            distance3 = wp.length_sq(displacement3) + SOFTENING_SQ
            distance4 = wp.length_sq(displacement4) + SOFTENING_SQ
            distance5 = wp.length_sq(displacement5) + SOFTENING_SQ
            distance6 = wp.length_sq(displacement6) + SOFTENING_SQ
            distance7 = wp.length_sq(displacement7) + SOFTENING_SQ
            distance8 = wp.length_sq(displacement8) + SOFTENING_SQ
            distance9 = wp.length_sq(displacement9) + SOFTENING_SQ
            distance10 = wp.length_sq(displacement10) + SOFTENING_SQ
            distance11 = wp.length_sq(displacement11) + SOFTENING_SQ
            distance12 = wp.length_sq(displacement12) + SOFTENING_SQ
            distance13 = wp.length_sq(displacement13) + SOFTENING_SQ
            distance14 = wp.length_sq(displacement14) + SOFTENING_SQ
            distance15 = wp.length_sq(displacement15) + SOFTENING_SQ
            distance16 = wp.length_sq(displacement16) + SOFTENING_SQ
            distance17 = wp.length_sq(displacement17) + SOFTENING_SQ
            distance18 = wp.length_sq(displacement18) + SOFTENING_SQ
            distance19 = wp.length_sq(displacement19) + SOFTENING_SQ
            distance20 = wp.length_sq(displacement20) + SOFTENING_SQ
            distance21 = wp.length_sq(displacement21) + SOFTENING_SQ
            distance22 = wp.length_sq(displacement22) + SOFTENING_SQ
            distance23 = wp.length_sq(displacement23) + SOFTENING_SQ
            distance24 = wp.length_sq(displacement24) + SOFTENING_SQ
            distance25 = wp.length_sq(displacement25) + SOFTENING_SQ
            distance26 = wp.length_sq(displacement26) + SOFTENING_SQ
            distance27 = wp.length_sq(displacement27) + SOFTENING_SQ
            distance28 = wp.length_sq(displacement28) + SOFTENING_SQ
            distance29 = wp.length_sq(displacement29) + SOFTENING_SQ
            distance30 = wp.length_sq(displacement30) + SOFTENING_SQ
            distance31 = wp.length_sq(displacement31) + SOFTENING_SQ
            root0 = wp.sqrt(distance0)
            inverse0 = 1.0 / root0
            cube0 = inverse0 * inverse0 * inverse0
            root1 = wp.sqrt(distance1)
            inverse1 = 1.0 / root1
            cube1 = inverse1 * inverse1 * inverse1
            root2 = wp.sqrt(distance2)
            inverse2 = 1.0 / root2
            cube2 = inverse2 * inverse2 * inverse2
            root3 = wp.sqrt(distance3)
            inverse3 = 1.0 / root3
            cube3 = inverse3 * inverse3 * inverse3
            root4 = wp.sqrt(distance4)
            inverse4 = 1.0 / root4
            cube4 = inverse4 * inverse4 * inverse4
            root5 = wp.sqrt(distance5)
            inverse5 = 1.0 / root5
            cube5 = inverse5 * inverse5 * inverse5
            root6 = wp.sqrt(distance6)
            inverse6 = 1.0 / root6
            cube6 = inverse6 * inverse6 * inverse6
            root7 = wp.sqrt(distance7)
            inverse7 = 1.0 / root7
            cube7 = inverse7 * inverse7 * inverse7
            root8 = wp.sqrt(distance8)
            inverse8 = 1.0 / root8
            cube8 = inverse8 * inverse8 * inverse8
            root9 = wp.sqrt(distance9)
            inverse9 = 1.0 / root9
            cube9 = inverse9 * inverse9 * inverse9
            root10 = wp.sqrt(distance10)
            inverse10 = 1.0 / root10
            cube10 = inverse10 * inverse10 * inverse10
            root11 = wp.sqrt(distance11)
            inverse11 = 1.0 / root11
            cube11 = inverse11 * inverse11 * inverse11
            root12 = wp.sqrt(distance12)
            inverse12 = 1.0 / root12
            cube12 = inverse12 * inverse12 * inverse12
            root13 = wp.sqrt(distance13)
            inverse13 = 1.0 / root13
            cube13 = inverse13 * inverse13 * inverse13
            root14 = wp.sqrt(distance14)
            inverse14 = 1.0 / root14
            cube14 = inverse14 * inverse14 * inverse14
            root15 = wp.sqrt(distance15)
            inverse15 = 1.0 / root15
            cube15 = inverse15 * inverse15 * inverse15
            root16 = wp.sqrt(distance16)
            inverse16 = 1.0 / root16
            cube16 = inverse16 * inverse16 * inverse16
            root17 = wp.sqrt(distance17)
            inverse17 = 1.0 / root17
            cube17 = inverse17 * inverse17 * inverse17
            root18 = wp.sqrt(distance18)
            inverse18 = 1.0 / root18
            cube18 = inverse18 * inverse18 * inverse18
            root19 = wp.sqrt(distance19)
            inverse19 = 1.0 / root19
            cube19 = inverse19 * inverse19 * inverse19
            root20 = wp.sqrt(distance20)
            inverse20 = 1.0 / root20
            cube20 = inverse20 * inverse20 * inverse20
            root21 = wp.sqrt(distance21)
            inverse21 = 1.0 / root21
            cube21 = inverse21 * inverse21 * inverse21
            root22 = wp.sqrt(distance22)
            inverse22 = 1.0 / root22
            cube22 = inverse22 * inverse22 * inverse22
            root23 = wp.sqrt(distance23)
            inverse23 = 1.0 / root23
            cube23 = inverse23 * inverse23 * inverse23
            root24 = wp.sqrt(distance24)
            inverse24 = 1.0 / root24
            cube24 = inverse24 * inverse24 * inverse24
            root25 = wp.sqrt(distance25)
            inverse25 = 1.0 / root25
            cube25 = inverse25 * inverse25 * inverse25
            root26 = wp.sqrt(distance26)
            inverse26 = 1.0 / root26
            cube26 = inverse26 * inverse26 * inverse26
            root27 = wp.sqrt(distance27)
            inverse27 = 1.0 / root27
            cube27 = inverse27 * inverse27 * inverse27
            root28 = wp.sqrt(distance28)
            inverse28 = 1.0 / root28
            cube28 = inverse28 * inverse28 * inverse28
            root29 = wp.sqrt(distance29)
            inverse29 = 1.0 / root29
            cube29 = inverse29 * inverse29 * inverse29
            root30 = wp.sqrt(distance30)
            inverse30 = 1.0 / root30
            cube30 = inverse30 * inverse30 * inverse30
            root31 = wp.sqrt(distance31)
            inverse31 = 1.0 / root31
            cube31 = inverse31 * inverse31 * inverse31
            acceleration += PARTICLE_MASS * cube0 * displacement0
            acceleration += PARTICLE_MASS * cube1 * displacement1
            acceleration += PARTICLE_MASS * cube2 * displacement2
            acceleration += PARTICLE_MASS * cube3 * displacement3
            acceleration += PARTICLE_MASS * cube4 * displacement4
            acceleration += PARTICLE_MASS * cube5 * displacement5
            acceleration += PARTICLE_MASS * cube6 * displacement6
            acceleration += PARTICLE_MASS * cube7 * displacement7
            acceleration += PARTICLE_MASS * cube8 * displacement8
            acceleration += PARTICLE_MASS * cube9 * displacement9
            acceleration += PARTICLE_MASS * cube10 * displacement10
            acceleration += PARTICLE_MASS * cube11 * displacement11
            acceleration += PARTICLE_MASS * cube12 * displacement12
            acceleration += PARTICLE_MASS * cube13 * displacement13
            acceleration += PARTICLE_MASS * cube14 * displacement14
            acceleration += PARTICLE_MASS * cube15 * displacement15
            acceleration += PARTICLE_MASS * cube16 * displacement16
            acceleration += PARTICLE_MASS * cube17 * displacement17
            acceleration += PARTICLE_MASS * cube18 * displacement18
            acceleration += PARTICLE_MASS * cube19 * displacement19
            acceleration += PARTICLE_MASS * cube20 * displacement20
            acceleration += PARTICLE_MASS * cube21 * displacement21
            acceleration += PARTICLE_MASS * cube22 * displacement22
            acceleration += PARTICLE_MASS * cube23 * displacement23
            acceleration += PARTICLE_MASS * cube24 * displacement24
            acceleration += PARTICLE_MASS * cube25 * displacement25
            acceleration += PARTICLE_MASS * cube26 * displacement26
            acceleration += PARTICLE_MASS * cube27 * displacement27
            acceleration += PARTICLE_MASS * cube28 * displacement28
            acceleration += PARTICLE_MASS * cube29 * displacement29
            acceleration += PARTICLE_MASS * cube30 * displacement30
            acceleration += PARTICLE_MASS * cube31 * displacement31

    updated_velocity = velocity[body_index] + acceleration * DT
    velocity[body_index] = updated_velocity
    new_position[body_index] = position + DT * updated_velocity


def _integration_kernel(old_position, velocity, new_position, num_bodies):
    """Select the optimized kernel for its validated native configuration."""
    if num_bodies not in (4096, 16384):
        return integrate_bodies_tiled
    if DT != 0.016 or SOFTENING_SQ != 0.1**2 or TILE_SIZE != 64 or PARTICLE_MASS != 1.0:
        return integrate_bodies_tiled
    if SOURCE_TILE_SIZE != 256 or wp.get_module_options(integrate_bodies_tiled.module)["fast_math"]:
        return integrate_bodies_tiled
    arrays = (old_position, velocity, new_position)
    device = old_position.device
    if not device.is_cuda:
        return integrate_bodies_tiled
    for array in arrays:
        if (
            array.device != device
            or array.dtype != wp.vec3
            or array.shape != (num_bodies,)
            or array.strides != (12,)
            or array.capacity != num_bodies * 12
            or array.requires_grad
        ):
            return integrate_bodies_tiled
    spans = [(array.ptr, array.ptr + array.capacity) for array in arrays]
    for i, (begin, end) in enumerate(spans):
        for other_begin, other_end in spans[i + 1 :]:
            if begin < other_end and other_begin < end:
                return integrate_bodies_tiled
    return _integrate_bodies_tiled_optimized


class Example:
    def __init__(self, headless=False, num_bodies=16384):
        self.num_bodies = num_bodies

        rng = np.random.default_rng(42)

        # Sample the surface of a sphere
        phi = np.arccos(1.0 - 2.0 * rng.uniform(low=0.0, high=1.0, size=self.num_bodies))
        theta = rng.uniform(low=0.0, high=2.0 * np.pi, size=self.num_bodies)
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        init_pos_np = np.stack((x, y, z), axis=1)

        scale = (num_bodies / 1024) ** (1 / 2)  # Scale factor to maintain a constant density
        inner = 0.9625 * scale
        outer = 1.54 * scale
        radii = inner + (outer - inner) * rng.uniform(size=(self.num_bodies, 1))
        init_pos_np = init_pos_np * radii

        axis = np.array([0.0, 0.0, 1.0])
        v_scale = scale * 3.08
        init_vel_np = v_scale * np.cross(init_pos_np, axis)

        self.graph_scale = np.max(radii) * 5.0
        self.pos_array_0 = wp.array(init_pos_np, dtype=wp.vec3)
        self.pos_array_1 = wp.empty_like(self.pos_array_0)
        self.vel_array = wp.array(init_vel_np, dtype=wp.vec3)

        if headless:
            self.scatter_plot = None
        else:
            self.scatter_plot = self.create_plot()

    def create_plot(self):
        import matplotlib.pyplot as plt  # noqa: PLC0415

        # Create a figure and a 3D axis for the plot
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111, projection="3d")

        # Scatter plot of initial positions
        point_size = 0.05 * self.graph_scale
        init_pos_np = self.pos_array_0.numpy()
        scatter_plot = ax.scatter(
            init_pos_np[:, 0], init_pos_np[:, 1], init_pos_np[:, 2], s=point_size, c="#76b900", alpha=0.5
        )

        # Set axis limits
        ax.set_xlim(-self.graph_scale, self.graph_scale)
        ax.set_ylim(-self.graph_scale, self.graph_scale)
        ax.set_zlim(-self.graph_scale, self.graph_scale)

        return scatter_plot

    def step(self):
        wp.launch(
            _integration_kernel(self.pos_array_0, self.vel_array, self.pos_array_1, self.num_bodies),
            dim=self.num_bodies,
            inputs=[self.pos_array_0, self.vel_array, self.pos_array_1, self.num_bodies],
            block_dim=TILE_SIZE,
        )

        # Swap arrays
        (self.pos_array_0, self.pos_array_1) = (self.pos_array_1, self.pos_array_0)

    def render(self):
        positions_cpu = self.pos_array_0.numpy()

        # Update scatter plot positions
        self.scatter_plot._offsets3d = (
            positions_cpu[:, 0],
            positions_cpu[:, 1],
            positions_cpu[:, 2],
        )

    # Function to update the scatter plot
    def step_and_render(self, frame):
        self.step()
        self.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num-frames", type=int, default=1000, help="Total number of frames.")
    parser.add_argument("-N", help="Number of bodies. Should be a multiple of 64.", type=int, default=16384)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    if args.device == "cpu":
        print("This example only runs on CUDA devices.")
        exit()

    with wp.ScopedDevice(args.device):
        example = Example(headless=args.headless, num_bodies=args.N)

        if not args.headless:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation

            # Create the animation
            ani = FuncAnimation(example.fig, example.step_and_render, frames=args.num_frames, interval=50, repeat=False)

            # Display the animation
            plt.show()

        else:
            for _ in range(args.num_frames):
                example.step()
