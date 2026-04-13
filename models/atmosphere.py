"""Atmosphere density models."""

from __future__ import annotations

import bisect
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pymsis
import spiceypy as spice

from ..utils.constants import EARTH_FLATTENING, get_body_radius


class AtmosphereModel(ABC):
    """Abstract base class for atmosphere density models."""

    @abstractmethod
    def density(self, t: float, state: np.ndarray) -> float:
        """Compute atmospheric density.

        Parameters
        ----------
        t : float
            Elapsed time since the initial epoch [s].
        state : np.ndarray
            State vector ``[x, y, z, vx, vy, vz]`` [km, km/s].

        Returns
        -------
        float
            Atmospheric density [kg/m^3].
        """


class ZeroAtmosphere(AtmosphereModel):
    """Zero-density atmosphere (vacuum)."""

    def density(self, t: float, state: np.ndarray) -> float:
        """Return zero density."""

        return 0.0


class ExponentialAtmosphere(AtmosphereModel):
    """Exponential atmosphere model.

    Parameters
    ----------
    rho0 : float
        Reference density at ``href`` [kg/m^3].
    href : float
        Reference altitude [km].
    scale_height : float
        Scale height [km].
    """

    def __init__(
        self,
        rho0: float = 4e-12,
        href: float = 400.0,
        scale_height: float = 55.0,
    ) -> None:
        self.rho0 = rho0
        self.href = href
        self.scale_height = scale_height

    def density(self, t: float, state: np.ndarray) -> float:
        """Return density from an exponential profile."""

        r = state[:3]
        r_norm = np.linalg.norm(r)
        h = r_norm - get_body_radius("EARTH")
        return self.rho0 * np.exp(-(h - self.href) / self.scale_height)


class SimpleTableAtmosphere(AtmosphereModel):
    """Table-based atmosphere with log-linear interpolation.

    Parameters
    ----------
    altitudes : np.ndarray
        Altitude grid [km].
    densities : np.ndarray
        Density values [kg/m^3].
    """

    def __init__(
        self,
        altitudes: np.ndarray,
        densities: np.ndarray,
    ) -> None:
        self.altitudes = altitudes
        self.densities = densities

    def density(self, t: float, state: np.ndarray) -> float:
        """Return density via log-linear interpolation in the altitude table."""

        r = state[:3]
        r_norm = np.linalg.norm(r)
        h = r_norm - get_body_radius("EARTH")

        i = bisect.bisect_left(self.altitudes, h)

        if i <= 0:
            return self.densities[0]
        if i >= len(self.altitudes):
            return self.densities[-1]

        # Log-linear interpolation between bracketing altitude points
        h0, h1 = self.altitudes[i - 1], self.altitudes[i]
        rho0, rho1 = self.densities[i - 1], self.densities[i]
        lr0, lr1 = np.log(rho0), np.log(rho1)
        w = (h - h0) / (h1 - h0)

        return np.exp(lr0 * (1.0 - w) + lr1 * w)


class NRLMSISEAtmosphere(AtmosphereModel):
    """NRLMSISE-00 atmosphere model via ``pymsis``.

    Parameters
    ----------
    et0 : float
        Ephemeris time at *t* = 0 [s].
    frame : str
        Inertial reference frame (e.g., ``"J2000"``).
    """

    def __init__(self, et0: float, frame: str) -> None:
        self.et0 = et0
        self.frame = frame

    def density(self, t: float, state: np.ndarray) -> float:
        """Return density from the NRLMSISE-00 model."""

        r = state[:3]
        et = self.et0 + t

        # Convert to IAU_EARTH for geodetic coordinates
        R = spice.pxform(self.frame, "IAU_EARTH", et)
        r_itrf = spice.mxv(R, r)
        re = get_body_radius("EARTH")
        lon_rad, lat_rad, alt = spice.recgeo(r_itrf, re, EARTH_FLATTENING)
        lon = np.degrees(lon_rad)
        lat = np.degrees(lat_rad)

        # Solar and geomagnetic activity defaults
        f107a = 150.0
        f107 = 150.0
        ap = 4.0

        output = pymsis.calculate(
            dates=datetime.fromisoformat(
                spice.et2utc(et, "ISOC", 3).replace("Z", "+00:00")
            ),
            lons=lon,
            lats=lat,
            alts=alt,
            f107s=f107,
            f107as=f107a,
            aps=ap,
        )

        return float(output[0, 0])
