"""Solar radiation pressure perturbation."""

from __future__ import annotations

import numpy as np
import spiceypy as spice

from ..models.attitude import AttitudeModel
from ..models.spacecraft import FlatPlateSpacecraft, SpacecraftModel, SpherePlateSpacecraft, SphericalSpacecraft
from ..utils.constants import (
    AU,
    LAMBERTIAN_COEFFICIENT,
    SOLAR_PRESSURE_1AU,
    get_body_radius,
)


class SolarRadiationPressure:
    """Solar radiation pressure perturbation.

    Parameters
    ----------
    spacecraft : SpacecraftModel
        Spacecraft model.
    et0 : float
        Initial ephemeris time [s].
    central_body : str
        Central body for shadow calculations.
    frame : str
        Inertial reference frame.
    use_shadow : bool
        Include cylindrical shadow model.
    attitude : AttitudeModel, optional
        Required for ``FlatPlateSpacecraft``.
    """

    def __init__(
        self,
        spacecraft: SpacecraftModel,
        et0: float,
        central_body: str,
        frame: str = "J2000",
        use_shadow: bool = True,
        attitude: AttitudeModel | None = None,
    ) -> None:
        self.spacecraft = spacecraft
        self.et0 = et0
        self.central_body = central_body.upper()
        self.frame = frame
        self.use_shadow = use_shadow
        self._attitude = attitude

    def compute_acceleration(
        self,
        t: float,
        state: np.ndarray,
    ) -> np.ndarray:
        """Compute the SRP acceleration.

        Parameters
        ----------
        t : float
            Elapsed time since the initial epoch [s].
        state : np.ndarray
            State vector ``[x, y, z, vx, vy, vz]`` [km, km/s].

        Returns
        -------
        np.ndarray
            SRP acceleration vector [km/s^2].
        """

        r_sat = state[:3]
        et = self.et0 + t

        r_sun, _ = spice.spkpos("SUN", et, self.frame, "NONE", self.central_body)
        r_sat_sun = r_sun - r_sat
        d_sun = np.linalg.norm(r_sat_sun)
        sun_dir = r_sat_sun / d_sun

        # Shadow factor: 0.0 in umbra, 1.0 in full sunlight
        shadow = self._shadow_factor(r_sat, r_sun)

        if shadow < 1e-10:
            return np.zeros(3)

        # Solar pressure scaled to the current Sun distance
        pressure = SOLAR_PRESSURE_1AU * (AU / d_sun) ** 2

        if isinstance(self.spacecraft, SphericalSpacecraft):
            cr = self.spacecraft.cr
            area_mass = self.spacecraft.area_mass_ratio
            return shadow * cr * pressure * area_mass / 1000.0 * sun_dir

        if isinstance(self.spacecraft, FlatPlateSpacecraft):
            attitude = self._attitude if self._attitude is not None else self.spacecraft.attitude
            if attitude is None:
                raise ValueError("FlatPlateSpacecraft requires an attitude model.")
            normal = attitude.get_normal_vector(t, state)
            return self._plate_srp_accel(
                shadow, pressure, sun_dir, normal,
                self.spacecraft.area_mass_ratio, self.spacecraft.ca, self.spacecraft.cs, self.spacecraft.cd_srp,
            )

        # SpherePlateSpacecraft
        attitude = self._attitude if self._attitude is not None else self.spacecraft.attitude
        if attitude is None:
            raise ValueError("SpherePlateSpacecraft plate SRP requires an attitude model.")
        normal = attitude.get_normal_vector(t, state)
        a_total = shadow * self.spacecraft.cr_sphere * pressure * self.spacecraft.sphere_area_mass_ratio / 1000.0 * sun_dir
        a_total += self._plate_srp_accel(
            shadow, pressure, sun_dir, normal,
            self.spacecraft.plate_area_mass_ratio, self.spacecraft.ca, self.spacecraft.cs, self.spacecraft.cd_srp,
        )
        return a_total

    @staticmethod
    def _plate_srp_accel(
        shadow: float,
        pressure: float,
        sun_dir: np.ndarray,
        normal: np.ndarray,
        area_mass: float,
        ca: float,
        cs: float,
        cd_srp: float,
    ) -> np.ndarray:
        """Compute the SRP acceleration contribution from a flat plate."""

        s_dot_n = np.dot(sun_dir, normal)
        abs_s_dot_n = abs(s_dot_n)
        pressure_am = pressure * area_mass / 1000.0
        s_component = abs_s_dot_n * (ca + cd_srp) * sun_dir
        n_component = s_dot_n * (LAMBERTIAN_COEFFICIENT * cd_srp + 2.0 * cs * abs_s_dot_n) * normal
        return -shadow * pressure_am * (s_component + n_component)

    def _shadow_factor(
        self,
        r_sat: np.ndarray,
        r_sun: np.ndarray,
    ) -> float:
        """Compute the cylindrical shadow factor.

        Returns
        -------
        float
            0.0 in umbra, 1.0 in full sunlight.
        """

        if not self.use_shadow:
            return 1.0

        body_radius = get_body_radius(self.central_body)
        u_sun = r_sun / np.linalg.norm(r_sun)

        # Projection of satellite position onto the Earth-Sun axis
        proj = np.dot(r_sat, u_sun)

        # Satellite is on the sunlit side of the body
        if proj > 0:
            return 1.0

        r_perp = r_sat - proj * u_sun

        if np.linalg.norm(r_perp) < body_radius:
            return 0.0

        return 1.0

    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        """Evaluate the SRP acceleration (alias for ``compute_acceleration``)."""

        return self.compute_acceleration(t, state)
