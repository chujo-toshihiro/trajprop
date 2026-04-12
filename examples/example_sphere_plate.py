"""Example: trajectory propagation with a sphere-plate spacecraft."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice

from trajprop import (
    AtmosphericDrag,
    ExponentialAtmosphere,
    J2Perturbation,
    PropagationResult,
    Propagator,
    SolarRadiationPressure,
    SpherePlateSpacecraft,
    SRPAttitude,
    get_body_radius,
    init_spice,
)


def main() -> None:
    """Run the sphere-plate spacecraft trajectory propagation example."""

    kernel_dir = "spice_kernels"
    init_spice(kernel_dir)

    central_body = "EARTH"
    frame = "J2000"

    start_time_str = "2024-01-01T00:00:00"
    et0 = spice.utc2et(start_time_str)

    state0 = np.array([0.0, 0.0, 7000.0, 7.5, 0.0, 0.0])
    t_span = (0.0, 86400.0)
    t_eval = np.linspace(0, 86400, 1000)

    attitude = SRPAttitude(
        et0=et0,
        central_body=central_body,
        frame=frame,
        cone_angle=0.0,
        clock_angle=0.0,
    )

    spacecraft = SpherePlateSpacecraft(
        sphere_area_mass_ratio=0.00454,
        plate_area_mass_ratio=0.6,
        cd_sphere=2.2,
        cr_sphere=1.5,
        cd_plate=2.2,
        ca=0.053,
        cs=0.882,
        cd_srp=0.065,
        attitude=attitude,
    )

    propagator = Propagator(
        central_body=central_body,
        third_bodies=["MOON", "SUN"],
        frame=frame,
    )

    j2 = J2Perturbation(et0=et0)
    propagator.add_perturbation(j2)

    drag = AtmosphericDrag(
        atmosphere=ExponentialAtmosphere(),
        spacecraft=spacecraft,
        et0=et0,
        frame=frame,
    )
    propagator.add_perturbation(drag)

    srp = SolarRadiationPressure(
        spacecraft=spacecraft,
        et0=et0,
        central_body=central_body,
    )
    propagator.add_perturbation(srp)

    result = propagator.propagate(
        initial_state=state0,
        t_span=t_span,
        et0=et0,
        t_eval=t_eval,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(result.states[:, 0], result.states[:, 1], result.states[:, 2])

    # Earth sphere
    r_earth = get_body_radius(central_body)
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    ex = r_earth * np.outer(np.cos(u), np.sin(v))
    ey = r_earth * np.outer(np.sin(u), np.sin(v))
    ez = r_earth * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(ex, ey, ez, color="royalblue", alpha=0.4, linewidth=0)

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.grid(True)

    # Equal-aspect scaling for the 3-D orbit plot
    xyz_min = np.min(result.states[:, :3], axis=0)
    xyz_max = np.max(result.states[:, :3], axis=0)
    max_range = np.max(xyz_max - xyz_min)
    mid = (xyz_max + xyz_min) / 2.0
    ax.set_xlim(mid[0] - max_range / 2.0, mid[0] + max_range / 2.0)
    ax.set_ylim(mid[1] - max_range / 2.0, mid[1] + max_range / 2.0)
    ax.set_zlim(mid[2] - max_range / 2.0, mid[2] + max_range / 2.0)
    ax.set_box_aspect([1, 1, 1])

    plt.show()


if __name__ == "__main__":
    main()
