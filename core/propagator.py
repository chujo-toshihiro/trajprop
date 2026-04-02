"""Orbital trajectory propagator."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.integrate import solve_ivp

from .gravity import gravity_acceleration, third_body_acceleration


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

        if type(perturbation).__name__ in ("J2Perturbation", "AtmosphericDrag"):
            if self.central_body.upper() != "EARTH":
                raise ValueError(
                    "J2Perturbation and AtmosphericDrag are only supported for central_body='EARTH'."
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
        t_eval: np.ndarray,
        et0: float,
        ivp_method: str = "RK45",
        rtol: float = 1e-8,
        atol: float = 1e-10,
        **integrator_options: object,
    ) -> np.ndarray:
        """Propagate a spacecraft trajectory.

        Parameters
        ----------
        initial_state : np.ndarray
            Initial state ``[x, y, z, vx, vy, vz]`` [km, km/s].
        t_eval : np.ndarray
            Evaluation time points [s].
        et0 : float
            Initial epoch (SPICE ephemeris time) [s].
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
        np.ndarray
            Trajectory states at each evaluation time [km, km/s].
        """

        t_min = t_eval[0]
        t_max = t_eval[-1]
        solution = solve_ivp(
            self.dynamics,
            (t_min, t_max),
            initial_state,
            t_eval=t_eval,
            method=ivp_method,
            rtol=rtol,
            atol=atol,
            args=(et0,),
            **integrator_options,
        )
        return solution.y.T
