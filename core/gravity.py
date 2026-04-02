"""Gravity acceleration models."""

from __future__ import annotations

import numpy as np
import spiceypy as spice

from ..utils.constants import get_mu


def gravity_acceleration(r: np.ndarray, central_body: str) -> np.ndarray:
    """Compute gravitational acceleration from a central body.

    Parameters
    ----------
    r : np.ndarray
        Position vector in the inertial frame [km].
    central_body : str
        Central body name (e.g., ``"EARTH"``).

    Returns
    -------
    np.ndarray
        Gravitational acceleration vector [km/s^2].
    """

    mu = get_mu(central_body)
    r_mag = np.linalg.norm(r)
    return -mu * r / (r_mag**3)

def third_body_acceleration(
    r: np.ndarray,
    et: float,
    third_bodies: list[str],
    central_body: str,
    frame: str,
) -> np.ndarray:
    """Compute third-body gravitational perturbation accelerations.

    Uses SPICE ephemeris to obtain third-body positions.

    Parameters
    ----------
    r : np.ndarray
        Position vector in the inertial frame [km].
    et : float
        Ephemeris time [s].
    third_bodies : list[str]
        Third-body names (e.g., ``["MOON", "SUN"]``).
    central_body : str
        Central body name (e.g., ``"EARTH"``).
    frame : str
        Reference frame for SPICE positions (e.g., ``"J2000"``).

    Returns
    -------
    np.ndarray
        Third-body acceleration vector [km/s^2].
    """

    a_total = np.zeros(3)

    for body in third_bodies:
        body_name = body.upper()
        mu = get_mu(body_name)
        r_body, _ = spice.spkpos(body_name, et, frame, "NONE", central_body)
        r_rel = r_body - r

        r_rel_norm = np.linalg.norm(r_rel)
        r_body_norm = np.linalg.norm(r_body)

        if r_rel_norm > 1e-10 and r_body_norm > 1e-10:
            a_total += mu * (r_rel / r_rel_norm**3 - r_body / r_body_norm**3)

    return a_total
