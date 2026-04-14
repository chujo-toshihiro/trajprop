"""Atmospheric drag perturbation."""

from __future__ import annotations

import numpy as np
import spiceypy as spice

from ..models.atmosphere import AtmosphereModel
from ..models.spacecraft import FlatPlateSpacecraft, SpacecraftModel, SpherePlateSpacecraft, SphericalSpacecraft
from ..utils.constants import EARTH_ROTATION_RATE


class AtmosphericDrag:
    """Atmospheric drag perturbation.

    Parameters
    ----------
    atmosphere : AtmosphereModel
        Atmospheric density model.
    spacecraft : SpacecraftModel
        Spacecraft model.
    central_body : str
        Central body name (must be ``"EARTH"``).
    co_rotating : bool
        Account for Earth co-rotation when computing relative velocity.
    et0 : float, optional
        Ephemeris time at *t* = 0 [s]. Required when ``co_rotating=True``.
    frame : str, optional
        Inertial reference frame of the state vector (e.g. ``"J2000"``).
        Required when ``co_rotating=True``.
    """

    def __init__(
        self,
        atmosphere: AtmosphereModel,
        spacecraft: SpacecraftModel,
        central_body: str = "EARTH",
        co_rotating: bool = True,
        et0: float | None = None,
        frame: str | None = None,
    ) -> None:
        if central_body.upper() != "EARTH":
            raise ValueError(f"only EARTH is supported, got {central_body}.")
        if co_rotating and (et0 is None or frame is None):
            raise ValueError("co_rotating=True requires et0 and frame.")

        self.atmosphere = atmosphere
        self.spacecraft = spacecraft
        self.central_body = central_body.upper()
        self.co_rotating = co_rotating
        self.et0 = et0
        self.frame = frame

    def compute_acceleration(
        self,
        t: float,
        state: np.ndarray,
    ) -> np.ndarray:
        """Compute the aerodynamic acceleration.

        Parameters
        ----------
        t : float
            Elapsed time since the initial epoch [s].
        state : np.ndarray
            State vector ``[x, y, z, vx, vy, vz]`` [km, km/s].

        Returns
        -------
        np.ndarray
            Aerodynamic acceleration vector [km/s^2].
        """

        r, v = state[:3], state[3:]
        rho = self.atmosphere.density(t, state)

        if self.co_rotating:
            # Earth rotation vector in IAU_EARTH, transformed to the inertial frame
            et = self.et0 + t
            R = spice.pxform("IAU_EARTH", self.frame, et)
            omega = spice.mxv(R, [0.0, 0.0, EARTH_ROTATION_RATE])
            v_rel = v - np.cross(omega, r)
        else:
            v_rel = v

        v_norm = np.linalg.norm(v_rel)
        u_v = v_rel / v_norm

        if isinstance(self.spacecraft, SphericalSpacecraft):
            q_am = 0.5 * rho * v_norm**2 * self.spacecraft.area_mass_ratio * 1e3
            return -q_am * self.spacecraft.cd * u_v

        if isinstance(self.spacecraft, FlatPlateSpacecraft):
            if self.spacecraft.attitude is None:
                raise ValueError("FlatPlateSpacecraft requires an attitude model.")
            normal = self.spacecraft.attitude.get_normal_vector(t, state)
            return self._plate_drag_accel(rho, v_norm, u_v, normal, self.spacecraft.area_mass_ratio, self.spacecraft.cd)

        # SpherePlateSpacecraft
        if self.spacecraft.attitude is None:
            raise ValueError("SpherePlateSpacecraft plate drag requires an attitude model.")
        normal = self.spacecraft.attitude.get_normal_vector(t, state)
        q_am_sphere = 0.5 * rho * v_norm**2 * self.spacecraft.sphere_area_mass_ratio * 1e3
        a_total = -q_am_sphere * self.spacecraft.cd_sphere * u_v
        a_total += self._plate_drag_accel(rho, v_norm, u_v, normal, self.spacecraft.plate_area_mass_ratio, self.spacecraft.cd_plate)
        return a_total

    @staticmethod
    def _plate_drag_accel(
        rho: float,
        v_norm: float,
        u_v: np.ndarray,
        normal: np.ndarray,
        area_mass: float,
        cd: float,
    ) -> np.ndarray:
        """Compute the drag acceleration contribution from a flat plate."""

        q_am = 0.5 * rho * v_norm**2 * area_mass * 1e3
        cos_theta = abs(np.dot(normal, u_v))
        return -q_am * cd * cos_theta * u_v

    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        """Evaluate the aerodynamic acceleration (alias for ``compute_acceleration``)."""

        return self.compute_acceleration(t, state)
