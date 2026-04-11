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


class SpherePlateSpacecraft(SpacecraftModel):
    """Spacecraft model combining a spherical body and a single flat plate.

    The spherical body contributes drag and SRP independently of attitude.
    The flat plate contributes drag and SRP that depend on the panel normal
    vector, which is provided through an :class:`~trajprop.models.attitude.AttitudeModel`.
    Because the entire spacecraft attitude is encoded in the single normal
    vector of the plate, any existing attitude model can be used as-is.

    Parameters
    ----------
    sphere_area_mass_ratio : float
        Area-to-mass ratio of the spherical body [m^2/kg].
    plate_area_mass_ratio : float
        Area-to-mass ratio of the flat plate [m^2/kg].
    cd_sphere : float, optional
        Drag coefficient of the sphere (required for atmospheric drag).
    cr_sphere : float, optional
        Radiation pressure coefficient of the sphere (required for SRP).
    cd_plate : float, optional
        Drag coefficient of the flat plate (required for atmospheric drag).
    ca : float, optional
        SRP absorption coefficient of the plate (required for SRP).
    cs : float, optional
        SRP specular reflection coefficient of the plate (required for SRP).
    cd_srp : float, optional
        SRP diffuse reflection coefficient of the plate (required for SRP).
    attitude : AttitudeModel, optional
        Attitude model that returns the plate normal vector.
        Required when computing drag or SRP for the flat-plate component.
    """

    def __init__(
        self,
        sphere_area_mass_ratio: float,
        plate_area_mass_ratio: float,
        cd_sphere: float | None = None,
        cr_sphere: float | None = None,
        cd_plate: float | None = None,
        ca: float | None = None,
        cs: float | None = None,
        cd_srp: float | None = None,
        attitude: object | None = None,
    ) -> None:
        self.sphere_area_mass_ratio = sphere_area_mass_ratio
        self.plate_area_mass_ratio = plate_area_mass_ratio
        self.cd_sphere = cd_sphere
        self.cr_sphere = cr_sphere
        self.cd_plate = cd_plate
        self.ca = ca
        self.cs = cs
        self.cd_srp = cd_srp
        self.attitude = attitude
