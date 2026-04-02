"""Physical constants and look-up utilities."""

from __future__ import annotations

# Astronomical unit [km]
AU: float = 149597870.7

# Earth J2 zonal harmonic coefficient [-]
J2_EARTH: float = 0.001082626925639

# Earth sidereal rotation rate [rad/s]
EARTH_ROTATION_RATE: float = 7.2921150e-5

# WGS-84 Earth flattening [-]
EARTH_FLATTENING: float = 1.0 / 298.257223563

# Solar radiation pressure at 1 AU [N/m^2]
SOLAR_PRESSURE_1AU: float = 4.56e-6

# Lambertian diffuse reflection coefficient [-]
LAMBERTIAN_COEFFICIENT: float = 2.0 / 3.0


def get_mu(body_name: str) -> float:
    """Return the gravitational parameter for a celestial body.

    Parameters
    ----------
    body_name : str
        Celestial body name (case-insensitive).

    Returns
    -------
    float
        Gravitational parameter [km^3/s^2].

    Raises
    ------
    ValueError
        If the body name is not recognised.
    """

    mu_table = {
        "MERCURY": 22031.868551,
        "VENUS": 324858.592000,
        "EARTH": 398600.435507,
        "MARS": 42828.375816,
        "JUPITER": 126712764.100000,
        "SATURN": 37940584.841800,
        "URANUS": 5794556.400000,
        "NEPTUNE": 6836527.100580,
        "SUN": 132712440041.279419,
        "MOON": 4902.800118,
    }

    key = body_name.upper()
    if key in mu_table:
        return mu_table[key]

    raise ValueError(f"unknown body name: {body_name}.")


def get_body_radius(body_name: str) -> float:
    """Return the mean radius of a celestial body.

    Parameters
    ----------
    body_name : str
        Celestial body name (case-insensitive).

    Returns
    -------
    float
        Body radius [km].

    Raises
    ------
    ValueError
        If the body name is not recognised.
    """

    radius_table = {
        "MERCURY": 2439.7,
        "VENUS": 6051.8,
        "EARTH": 6378.137,
        "MARS": 3396.2,
        "JUPITER": 71492.0,
        "SATURN": 60268.0,
        "URANUS": 25559.0,
        "NEPTUNE": 24764.0,
        "SUN": 696000.0,
        "MOON": 1737.4,
    }

    key = body_name.upper()
    if key in radius_table:
        return radius_table[key]

    raise ValueError(f"unknown body name: {body_name}.")
