"""J2 oblateness perturbation."""

from __future__ import annotations

import numpy as np
import spiceypy as spice

from ..utils.constants import J2_EARTH, get_body_radius, get_mu


class J2Perturbation:
    """J2 gravitational perturbation (Earth oblateness).

    Parameters
    ----------
    et0 : float
        Ephemeris time at *t* = 0 [s].
    frame : str
        Inertial reference frame.
    """

    def __init__(self, et0: float, frame: str = "J2000") -> None:
        self.et0 = et0
        self.frame = frame

    def acceleration(self, t: float, r: np.ndarray) -> np.ndarray:
        """Compute J2 perturbation acceleration.

        Parameters
        ----------
        t : float
            Elapsed time since the initial epoch [s].
        r : np.ndarray
            Position vector in the inertial frame [km].

        Returns
        -------
        np.ndarray
            J2 acceleration vector in the inertial frame [km/s^2].
        """

        mu = get_mu("EARTH")

        # Transform to IAU_EARTH for body-fixed coordinates
        et = self.et0 + t
        R = spice.pxform(self.frame, "IAU_EARTH", et)
        r_itrf = spice.mxv(R, r)
        x, y, z = r_itrf

        r_norm = np.linalg.norm(r_itrf)
        r2 = r_norm**2
        z2 = z**2
        earth_radius = get_body_radius("EARTH")
        # J2 acceleration in the body-fixed frame
        factor = 1.5 * J2_EARTH * mu * earth_radius**2 / r_norm**5

        a_x = factor * x * (5.0 * z2 / r2 - 1.0)
        a_y = factor * y * (5.0 * z2 / r2 - 1.0)
        a_z = factor * z * (5.0 * z2 / r2 - 3.0)

        # Rotate back to inertial frame
        Rt = spice.xpose(R)
        a = spice.mxv(Rt, [a_x, a_y, a_z])

        return a

    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        """Evaluate J2 perturbation acceleration (alias for ``acceleration``)."""

        r = state[:3]
        return self.acceleration(t, r)
