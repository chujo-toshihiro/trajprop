"""Orbital trajectory propagator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import spiceypy as spice
from scipy.integrate import solve_ivp

from .gravity import gravity_acceleration, third_body_acceleration
from ..utils.constants import get_body_radius


@dataclass
class PropagationResult:
    """Result of a trajectory propagation.

    Attributes
    ----------
    times : np.ndarray
        Time points [s].
    states : np.ndarray
        Trajectory states at each time point ``[km, km/s]``.
    collided : bool
        ``True`` if integration was stopped by a collision event.
    """

    times: np.ndarray
    states: np.ndarray
    collided: bool = False


def _make_collision_event(
    body_name: str,
    central_body: str,
    frame: str,
    et0: float,
    radius: float,
) -> Callable[[float, np.ndarray], float]:
    """Return a terminal event function that triggers on body surface contact."""

    body_upper = body_name.upper()
    central_upper = central_body.upper()
    is_central = body_upper == central_upper

    def event(t: float, state: np.ndarray) -> float:
        r = state[:3]
        if is_central:
            r_body = np.zeros(3)
        else:
            r_body, _ = spice.spkpos(body_upper, et0 + t, frame, "NONE", central_upper)
        return float(np.linalg.norm(r - r_body) - radius)

    event.terminal = True
    event.direction = -1
    return event


class Perturbation(Protocol):
    """Protocol for perturbation models."""

    def __call__(self, t: float, state: np.ndarray) -> np.ndarray: ...


class Propagator:
    """Trajectory propagator for spacecraft simulation.

    Parameters
    ----------
    central_body : str
        Central body name (e.g., ``"EARTH"``, ``"SUN"``).
    third_bodies : list[str]
        Third-body names for gravitational perturbations.
    frame : str
        Inertial reference frame (e.g., ``"J2000"``).
    """

    def __init__(
        self,
        central_body: str,
        third_bodies: list[str],
        frame: str,
    ) -> None:
        self.central_body = central_body
        self.third_bodies = third_bodies
        self.frame = frame
        self.perturbations: list[Perturbation] = []

    def add_perturbation(self, perturbation: Perturbation) -> None:
        """Add a perturbation model.

        Parameters
        ----------
        perturbation : Perturbation
            Perturbation model implementing the ``Perturbation`` protocol.

        Raises
        ------
        ValueError
            If the perturbation is incompatible with the central body.
        """

        if type(perturbation).__name__ in ("GravityPerturbation", "AtmosphericDrag"):
            if self.central_body.upper() != "EARTH":
                raise ValueError(
                    "GravityPerturbation and AtmosphericDrag are only supported for central_body='EARTH'."
                )
        self.perturbations.append(perturbation)

    def remove_perturbation(self, perturbation: Perturbation) -> None:
        """Remove a perturbation model.

        Parameters
        ----------
        perturbation : Perturbation
            Perturbation model to remove.
        """

        if perturbation in self.perturbations:
            self.perturbations.remove(perturbation)

    def clear_perturbations(self) -> None:
        """Remove all perturbation models."""

        self.perturbations.clear()

    def dynamics(
        self,
        t: float,
        state: np.ndarray,
        et0: float,
    ) -> np.ndarray:
        """Compute the state time-derivative for ODE integration.

        Parameters
        ----------
        t : float
            Elapsed time since the initial epoch [s].
        state : np.ndarray
            State vector ``[x, y, z, vx, vy, vz]`` [km, km/s].
        et0 : float
            Initial epoch (SPICE ephemeris time) [s].

        Returns
        -------
        np.ndarray
            State derivative ``[vx, vy, vz, ax, ay, az]`` [km/s, km/s^2].
        """

        r = state[:3]
        v = state[3:]

        # Central-body gravity
        a_gravity = gravity_acceleration(r, self.central_body)

        # Third-body perturbations
        a_third_body = np.zeros(3)
        if self.third_bodies:
            et = et0 + t
            a_third_body = third_body_acceleration(
                r,
                et,
                self.third_bodies,
                self.central_body,
                self.frame,
            )

        # User-defined perturbations
        a_perturbations = np.zeros(3)
        for perturbation in self.perturbations:
            a_perturbations += perturbation(t, state)

        a_total = a_gravity + a_third_body + a_perturbations

        return np.concatenate([v, a_total])

    def propagate(
        self,
        initial_state: np.ndarray,
        t_span: tuple[float, float],
        et0: float,
        t_eval: np.ndarray | None = None,
        detect_collision: bool = False,
        ivp_method: str = "RK45",
        rtol: float = 1e-8,
        atol: float = 1e-10,
        **integrator_options: object,
    ) -> PropagationResult:
        """Propagate a spacecraft trajectory.

        Parameters
        ----------
        initial_state : np.ndarray
            Initial state ``[x, y, z, vx, vy, vz]`` [km, km/s].
        t_span : tuple[float, float]
            Integration interval ``(t0, tf)`` [s].
        et0 : float
            Initial epoch (SPICE ephemeris time) [s].
        t_eval : np.ndarray or None, optional
            Specific time points at which to store the solution [s].
        detect_collision : bool, optional
            When ``True``, stop integration if the spacecraft intersects
            the surface of the central body or any third body.
        ivp_method : str, optional
            Integration method for ``solve_ivp`` (default: ``"RK45"``).
        rtol : float, optional
            Relative tolerance (default: ``1e-8``).
        atol : float, optional
            Absolute tolerance (default: ``1e-10``).
        **integrator_options : object
            Additional keyword arguments passed to ``solve_ivp``.

        Returns
        -------
        PropagationResult
            Result containing ``times`` [s], ``states`` [km, km/s], and
            ``collided`` flag.
        """

        events: list[Callable] | None = None
        if detect_collision:
            bodies = [self.central_body] + [
                b for b in self.third_bodies
                if b.upper() != self.central_body.upper()
            ]
            events = [
                _make_collision_event(b, self.central_body, self.frame, et0, get_body_radius(b))
                for b in bodies
            ]

        solution = solve_ivp(
            lambda t, state: self.dynamics(t, state, et0),
            t_span,
            initial_state,
            t_eval=t_eval,
            method=ivp_method,
            rtol=rtol,
            atol=atol,
            events=events,
            **integrator_options,
        )
        return PropagationResult(
            times=solution.t,
            states=solution.y.T,
            collided=solution.status == 1,
        )
