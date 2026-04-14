"""Spacecraft trajectory propagation with gravity and perturbations."""

from .core.gravity import gravity_acceleration, third_body_acceleration
from .core.propagator import PropagationResult, Propagator
from .models.atmosphere import (
    AtmosphereModel,
    ExponentialAtmosphere,
    NRLMSISEAtmosphere,
    SimpleTableAtmosphere,
    ZeroAtmosphere,
)
from .models.attitude import (
    AttitudeModel,
    NormalVectorAttitude,
    SRPAttitude,
    VelocityAlignedAttitude,
)
from .models.spacecraft import FlatPlateSpacecraft, SpacecraftModel, SphericalSpacecraft, SpherePlateSpacecraft
from .perturbations.atmospheric import AtmosphericDrag
from .perturbations.gravity import GravityPerturbation
from .perturbations.solar_radiation import SolarRadiationPressure
from .utils import get_body_radius, get_mu, init_spice

__all__ = [
    "AtmosphereModel",
    "AtmosphericDrag",
    "AttitudeModel",
    "ExponentialAtmosphere",
    "FlatPlateSpacecraft",
    "get_body_radius",
    "get_mu",
    "GravityPerturbation",
    "gravity_acceleration",
    "init_spice",
    "NormalVectorAttitude",
    "NRLMSISEAtmosphere",
    "PropagationResult",
    "Propagator",
    "SimpleTableAtmosphere",
    "SolarRadiationPressure",
    "SpacecraftModel",
    "SphericalSpacecraft",
    "SpherePlateSpacecraft",
    "SRPAttitude",
    "third_body_acceleration",
    "VelocityAlignedAttitude",
    "ZeroAtmosphere",
]
