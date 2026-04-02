# Trajectory Propagator

Spacecraft trajectory propagation with gravitational and perturbation modeling.

## Overview

A Python package for computing spacecraft trajectories in multi-body
gravitational environments with environmental perturbations. Integrates
position-dependent forces (gravity, atmospheric drag, solar radiation pressure)
with attitude-dependent effects and SPICE ephemeris data.

## Features

- **Gravity models** — Central body and third-body gravitational effects
- **Perturbations** — J2 oblateness, atmospheric drag, solar radiation
  pressure
- **Attitude models** — Velocity-aligned, Sun-aligned, fixed normal vector
- **Spacecraft models** — Spherical (cannonball) and flat plate geometries
- **Atmosphere models** — Exponential, table-based, NRLMSISE-00
- **SPICE integration** — Ephemeris data via spiceypy

## Requirements

- Python 3.9+
- NumPy >= 1.20
- SciPy >= 1.7
- spiceypy
- pymsis (optional, for NRLMSISE-00)

## Installation

```bash
conda create -n trajectory python=3.9
conda activate trajectory
conda install numpy scipy matplotlib
pip install spiceypy pymsis
```

### SPICE Kernels

Download required kernels from [NAIF](https://naif.jpl.nasa.gov/naif/data.html)
and place in `spice_kernels/`:

- `de440s.bsp` — Solar system ephemeris
- `naif0012.tls` — Leap seconds kernel
- `pck00010.tpc` — Planetary constants

## Usage

### Basic Trajectory Propagation

```python
import numpy as np
import spiceypy as spice
from trajprop import (
    Propagator,
    SphericalSpacecraft,
    ExponentialAtmosphere,
    AtmosphericDrag,
    J2Perturbation,
    init_spice,
)

# Initialize SPICE kernels
init_spice("spice_kernels")

# Setup
et0 = spice.utc2et("2024-01-01T00:00:00")
state0 = np.array([0.0, 0.0, 7000.0, 7.5, 0.0, 0.0])  # [km, km/s]
t_eval = np.linspace(0, 86400, 1000)  # [s]

# Create spacecraft and propagator
spacecraft = SphericalSpacecraft(area_mass_ratio=0.00454, cd=2.2, cr=1.5)
propagator = Propagator(
    central_body="EARTH",
    third_bodies=["MOON", "SUN"],
    frame="J2000",
)

# Add perturbations
j2 = J2Perturbation(et0=et0)
propagator.add_perturbation(j2)

drag = AtmosphericDrag(
    atmosphere=ExponentialAtmosphere(rho0=4e-12, href=400.0, scale_height=55.0),
    spacecraft=spacecraft,
    et0=et0,
    frame="J2000",
)
propagator.add_perturbation(drag)

# Propagate
solution = propagator.propagate(
    initial_state=state0,
    t_eval=t_eval,
    et0=et0,
)
```

### Solar Radiation Pressure with Attitude

```python
from trajprop import (
    FlatPlateSpacecraft,
    VelocityAlignedAttitude,
    SolarRadiationPressure,
)

# Define attitude (cone/clock angles from velocity direction)
attitude = VelocityAlignedAttitude(cone_angle=0.0, clock_angle=0.0)

# Create flat plate spacecraft with optical properties
spacecraft = FlatPlateSpacecraft(
    area_mass_ratio=0.6,
    cd=2.2,
    ca=0.053,      # Absorption coefficient
    cs=0.882,      # Specular reflection coefficient
    cd_srp=0.065,  # Diffuse reflection coefficient
    attitude=attitude,
)

# Add solar radiation pressure
srp = SolarRadiationPressure(
    spacecraft=spacecraft,
    et0=et0,
    central_body="EARTH",
    attitude=attitude,
)
propagator.add_perturbation(srp)
```

### Examples

```bash
python examples/example_spherical.py
python examples/example_flatplate.py
```

## API Reference

### Core

- `Propagator` — Main trajectory propagation class
- `SphericalSpacecraft` — Spherical spacecraft model
- `FlatPlateSpacecraft` — Flat plate spacecraft model

### Perturbations

- `J2Perturbation` — Earth oblateness perturbation
- `AtmosphericDrag` — Aerodynamic drag
- `SolarRadiationPressure` — Solar radiation pressure

### Attitude Models

- `VelocityAlignedAttitude` — Orientation relative to velocity vector
- `SRPAttitude` — Orientation relative to Sun direction
- `NormalVectorAttitude` — Fixed normal vector in inertial frame

### Atmosphere Models

- `ExponentialAtmosphere` — Simple exponential density model
- `SimpleTableAtmosphere` — Table-based with log-linear interpolation
- `NRLMSISEAtmosphere` — NRLMSISE-00 model (requires pymsis)

## Project Structure

```
trajprop/
├── core/
│   ├── gravity.py           # Gravitational acceleration
│   └── propagator.py        # Main propagator class
├── perturbations/
│   ├── j2.py                # J2 perturbation
│   ├── atmospheric.py       # Atmospheric drag/lift
│   └── solar_radiation.py   # Solar radiation pressure
├── models/
│   ├── atmosphere.py        # Atmosphere density models
│   ├── attitude.py          # Spacecraft attitude models
│   └── spacecraft.py        # Spacecraft geometry models
├── utils/
│   ├── constants.py         # Physical constants
│   └── spice.py             # SPICE kernel utilities
└── examples/
    ├── example_spherical.py
    └── example_flatplate.py
```

## References

- [SPICE Toolkit](https://naif.jpl.nasa.gov/naif/toolkit.html)
- [spiceypy](https://spiceypy.readthedocs.io/)
- [pymsis](https://github.com/SWxTREC/pymsis)

## License

MIT License
