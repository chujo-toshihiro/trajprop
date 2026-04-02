"""Spacecraft geometry models."""

from __future__ import annotations


class SpacecraftModel:
    """Abstract base class for spacecraft geometry models."""


class SphericalSpacecraft(SpacecraftModel):
    """Spherical (cannonball) spacecraft model.

    Parameters
    ----------
    area_mass_ratio : float
        Area-to-mass ratio [m^2/kg].
    cd : float, optional
        Drag coefficient (required for atmospheric drag).
    cr : float, optional
        Radiation pressure coefficient (required for SRP).
    """

    def __init__(
        self,
        area_mass_ratio: float,
        cd: float | None = None,
        cr: float | None = None,
    ) -> None:
        self.area_mass_ratio = area_mass_ratio
        self.cd = cd
        self.cr = cr


class FlatPlateSpacecraft(SpacecraftModel):
    """Flat-plate spacecraft model.

    Parameters
    ----------
    area_mass_ratio : float
        Area-to-mass ratio [m^2/kg].
    cd : float, optional
        Drag coefficient (required for atmospheric drag).
    ca : float, optional
        SRP absorption coefficient (required for SRP).
    cs : float, optional
        SRP specular reflection coefficient (required for SRP).
    cd_srp : float, optional
        SRP diffuse reflection coefficient (required for SRP).
    attitude : AttitudeModel, optional
        Attitude model.
    """

    def __init__(
        self,
        area_mass_ratio: float,
        cd: float | None = None,
        ca: float | None = None,
        cs: float | None = None,
        cd_srp: float | None = None,
        attitude: object | None = None,
    ) -> None:
        self.area_mass_ratio = area_mass_ratio
        self.cd = cd
        self.ca = ca
        self.cs = cs
        self.cd_srp = cd_srp
        self.attitude = attitude
