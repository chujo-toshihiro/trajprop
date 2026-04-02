"""Spacecraft attitude models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import spiceypy as spice


def _eval_angle_spec(
    spec: float | Callable,
    t: float,
    state: np.ndarray,
) -> float:
    """Evaluate an angle specification (constant or callable).

    Parameters
    ----------
    spec : float or Callable
        A constant angle value, or a callable ``f(t, state) -> float``
        or ``f(t) -> float``.
    t : float
        Elapsed time [s].
    state : np.ndarray
        State vector.

    Returns
    -------
    float
        Evaluated angle [rad].
    """

    if callable(spec):
        try:
            return float(spec(t, state))
        except TypeError:
            return float(spec(t))
    return float(spec)


class AttitudeModel(ABC):
    """Abstract base class for spacecraft attitude models."""

    @abstractmethod
    def get_normal_vector(
        self,
        t: float,
        state: np.ndarray,
    ) -> np.ndarray:
        """Return the spacecraft normal vector in the inertial frame.

        Parameters
        ----------
        t : float
            Elapsed time since the initial epoch [s].
        state : np.ndarray
            State vector ``[x, y, z, vx, vy, vz]`` [km, km/s].

        Returns
        -------
        np.ndarray
            Unit normal vector in the inertial frame.
        """


class VelocityAlignedAttitude(AttitudeModel):
    """Attitude defined by cone/clock angles relative to the velocity vector.

    Parameters
    ----------
    cone_angle : float or Callable
        Cone angle from the velocity direction [rad].
    clock_angle : float or Callable
        Clock angle about velocity, measured from orbit normal [rad].
    """

    def __init__(
        self,
        cone_angle: float | Callable,
        clock_angle: float | Callable,
    ) -> None:
        self.cone_angle = cone_angle
        self.clock_angle = clock_angle

    def get_normal_vector(
        self,
        t: float,
        state: np.ndarray,
    ) -> np.ndarray:
        """Return the normal vector in a velocity-aligned reference frame."""

        r, v = state[:3], state[3:]

        # Build the velocity-aligned orthonormal basis
        u_v = v / np.linalg.norm(v)
        u_h = np.cross(r, v)
        u_h = u_h / np.linalg.norm(u_h)
        u_perp = np.cross(u_h, u_v)

        cone = _eval_angle_spec(self.cone_angle, t, state)
        clock = _eval_angle_spec(self.clock_angle, t, state)

        normal = np.cos(cone) * u_v + np.sin(cone) * (
            np.cos(clock) * u_h + np.sin(clock) * u_perp
        )

        return normal / np.linalg.norm(normal)


class SRPAttitude(AttitudeModel):
    """Attitude defined by cone/clock angles relative to the Sun direction.

    Parameters
    ----------
    et0 : float
        Initial ephemeris time [s].
    central_body : str
        Central body name (e.g., ``"EARTH"``).
    frame : str
        Reference frame (e.g., ``"J2000"``).
    cone_angle : float or Callable
        Cone angle from the Sun direction [rad].
    clock_angle : float or Callable
        Clock angle about the Sun direction [rad].
    """

    def __init__(
        self,
        et0: float,
        central_body: str,
        frame: str,
        cone_angle: float | Callable,
        clock_angle: float | Callable,
    ) -> None:
        self.et0 = et0
        self.central_body = central_body.upper()
        self.frame = frame
        self.cone_angle = cone_angle
        self.clock_angle = clock_angle

    def get_normal_vector(
        self,
        t: float,
        state: np.ndarray,
    ) -> np.ndarray:
        """Return the normal vector in a Sun-direction-aligned reference frame."""

        et = self.et0 + t
        r_sun, _ = spice.spkpos("SUN", et, self.frame, "NONE", self.central_body)
        sun_dir = r_sun - state[:3]
        sun_dir = sun_dir / np.linalg.norm(sun_dir)

        # Build an orthonormal basis around the Sun direction
        z_axis = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(sun_dir, z_axis)) > 0.999:
            u_perp1 = np.cross(sun_dir, np.array([1.0, 0.0, 0.0]))
        else:
            u_perp1 = np.cross(sun_dir, z_axis)

        u_perp1 = u_perp1 / np.linalg.norm(u_perp1)
        u_perp2 = np.cross(sun_dir, u_perp1)

        cone = _eval_angle_spec(self.cone_angle, t, state)
        clock = _eval_angle_spec(self.clock_angle, t, state)

        normal = np.cos(cone) * sun_dir + np.sin(cone) * (
            np.cos(clock) * u_perp1 + np.sin(clock) * u_perp2
        )

        return normal / np.linalg.norm(normal)


class NormalVectorAttitude(AttitudeModel):
    """Attitude defined by a fixed or time-varying normal vector.

    Parameters
    ----------
    normal_vector : np.ndarray or Callable
        Normal vector in the inertial frame, or a callable
        ``f(t, state) -> np.ndarray`` or ``f(t) -> np.ndarray``.
        The vector is normalised before use.
    """

    def __init__(
        self,
        normal_vector: np.ndarray | Callable,
    ) -> None:
        self.normal_vector = normal_vector

    def get_normal_vector(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return the normalised normal vector."""

        return self._evaluate_normal_vector(self.normal_vector, t, state)

    def _evaluate_normal_vector(
        self,
        spec: np.ndarray | Callable,
        t: float,
        state: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a vector specification (constant or callable)."""

        if callable(spec):
            try:
                vec = spec(t, state)
            except TypeError:
                vec = spec(t)
        else:
            vec = spec

        vec = np.array(vec, dtype=float)

        return vec / np.linalg.norm(vec)
