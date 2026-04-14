"""Nonspherical gravity perturbation via spherical harmonics."""

from __future__ import annotations

import numpy as np
import spiceypy as spice

from ..utils.constants import J2_EARTH, get_body_radius, get_mu


class GravityPerturbation:
    """Nonspherical gravity perturbation via spherical harmonics.

    Parameters
    ----------
    et0 : float
        Ephemeris time at *t* = 0 [s].
    frame : str, optional
        Inertial reference frame of the state vector (e.g. ``"J2000"``).
    model : str, optional
        Gravity model identifier (``"J2"`` or ``"EGM2008"``).
    lmax : int, optional
        Maximum spherical-harmonic degree/order.
    """

    def __init__(
        self,
        et0: float,
        frame: str = "J2000",
        model: str = "J2",
        lmax: int = 70,
    ) -> None:
        model_upper = model.upper()
        if model_upper not in ("J2", "EGM2008"):
            raise ValueError(
                f"unknown gravity model '{model}', supported models are 'J2' and 'EGM2008'."
            )

        self.et0 = et0
        self.frame = frame
        self.model = model_upper
        self.lmax = lmax
        self._clm = None

        if self.model == "EGM2008":
            self._clm = self._load_egm2008(lmax)

    @staticmethod
    def _load_egm2008(lmax: int):
        """Load EGM2008 coefficients from ``pyshtools.datasets.Earth``."""

        try:
            import pyshtools as pysh
        except ImportError as exc:
            raise ImportError(
                "pyshtools is required for model='EGM2008'."
                " install it with: pip install pyshtools"
            ) from exc

        return pysh.datasets.Earth.EGM2008(lmax=lmax)

    def acceleration(self, t: float, r: np.ndarray) -> np.ndarray:
        """Compute the nonspherical gravity perturbation acceleration.

        Parameters
        ----------
        t : float
            Elapsed time since the initial epoch [s].
        r : np.ndarray
            Position vector in the inertial frame [km].

        Returns
        -------
        np.ndarray
            Perturbation acceleration in the inertial frame [km/s^2].
        """

        et = self.et0 + t
        R = spice.pxform(self.frame, "IAU_EARTH", et)
        r_itrf = np.array(spice.mxv(R, r))

        if self.model == "J2":
            a_itrf = self._accel_j2(r_itrf)
        else:
            a_itrf = self._accel_sh(r_itrf)

        Rt = spice.xpose(R)
        return np.array(spice.mxv(Rt, a_itrf))

    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        """Evaluate the perturbation acceleration (alias for ``acceleration``)."""

        return self.acceleration(t, state[:3])

    def _accel_j2(self, r_itrf: np.ndarray) -> np.ndarray:
        """Compute the analytic J2 perturbation acceleration in the body-fixed frame."""

        mu = get_mu("EARTH")
        x, y, z = r_itrf
        r_norm = np.linalg.norm(r_itrf)
        r2 = r_norm**2
        z2 = z**2
        re = get_body_radius("EARTH")
        factor = 1.5 * J2_EARTH * mu * re**2 / r_norm**5

        return np.array([
            factor * x * (5.0 * z2 / r2 - 1.0),
            factor * y * (5.0 * z2 / r2 - 1.0),
            factor * z * (5.0 * z2 / r2 - 3.0),
        ])

    def _accel_sh(self, r_itrf: np.ndarray) -> np.ndarray:
        """Compute the spherical-harmonic perturbation acceleration in the body-fixed frame."""

        from pyshtools.legendre import PlmBar_d1

        gm = self._clm.gm
        r0 = self._clm.r0
        lmax = self.lmax

        r_m = r_itrf * 1e3
        r = np.linalg.norm(r_m)
        x, y, z = r_m

        lat = np.arcsin(np.clip(z / r, -1.0, 1.0))
        lon = np.arctan2(y, x)
        cos_lat = np.cos(lat)
        sin_lat = np.sin(lat)

        p_flat, dp_flat = PlmBar_d1(lmax, sin_lat)

        size = lmax + 1
        m_idx, n_idx = np.triu_indices(size)   # all (m, n) with m <= n
        flat_idx = n_idx * (n_idx + 1) // 2 + m_idx
        Pbar = np.zeros((size, size))
        dPbar_dz = np.zeros((size, size))
        Pbar[m_idx, n_idx] = p_flat[flat_idx]
        dPbar_dz[m_idx, n_idx] = dp_flat[flat_idx]

        dPbar_dlat = dPbar_dz * cos_lat

        C = self._clm.coeffs[0].T
        S = self._clm.coeffs[1].T

        M = np.arange(lmax + 1)[:, np.newaxis]
        cos_ml = np.cos(M * lon)
        sin_ml = np.sin(M * lon)

        CS = C * cos_ml + S * sin_ml
        dCS_dlon = M * (-C * sin_ml + S * cos_ml)

        rn2d = ((r0 / r) ** np.arange(lmax + 1))[np.newaxis, :]
        n_plus_1 = (np.arange(lmax + 1) + 1)[np.newaxis, :]

        n_arr = np.arange(lmax + 1)[np.newaxis, :]
        m_arr = np.arange(lmax + 1)[:, np.newaxis]
        mask = (n_arr >= 2) & (m_arr <= n_arr)

        def _msum(arr: np.ndarray) -> float:
            return float(np.sum(np.where(mask, arr, 0.0)))

        coeff = gm / r**2
        a_r = -coeff * _msum(n_plus_1 * rn2d * Pbar * CS)
        a_lat = coeff * _msum(rn2d * dPbar_dlat * CS)
        a_lon = coeff * _msum(rn2d * Pbar * dCS_dlon)

        cos_lat_safe = cos_lat if abs(cos_lat) > 1e-10 else np.sign(cos_lat + 1e-30) * 1e-10
        cos_lon = np.cos(lon)
        sin_lon = np.sin(lon)

        a_x = a_r * cos_lat * cos_lon - a_lat * sin_lat * cos_lon - a_lon / cos_lat_safe * sin_lon
        a_y = a_r * cos_lat * sin_lon - a_lat * sin_lat * sin_lon + a_lon / cos_lat_safe * cos_lon
        a_z = a_r * sin_lat + a_lat * cos_lat

        return np.array([a_x, a_y, a_z]) * 1e-3
