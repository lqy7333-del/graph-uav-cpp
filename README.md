# Multi-field UAV coverage environment

This repository implements the geometry foundation, field discretization, and
the first Depot/UAV simulation model. It converts WGS84 farm polygons to local
metric coordinates, derives field properties and boundary segments, creates
regular-grid coverage points with 8-neighborhood links, and plots the result.
Each retained point represents a full grid-cell area, so total estimated area
can differ from polygon area near the boundary. The error depends on resolution
and grid alignment and is not guaranteed to decrease monotonically.

The UAV model supports waypoint path execution and a configurable linear energy
model, `alpha * distance + beta * covered_area`. Its default coefficients are
`alpha=0.2 Wh/m` and `beta=0.02 Wh/m^2`; replace them with measured values for a
real vehicle. Coverage configuration, region generation, coverage path
planning, and RL remain future phases.

## Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Run the demo

```powershell
python -m coverage_env.main
```

## Run tests

```powershell
python -m unittest discover -s coverage_env/tests -v
```

Coordinate sequences passed to `Field` are `(latitude, longitude)` pairs.
Shapely polygons use the standard `(longitude, latitude)` X/Y ordering.
Projected geometry and lengths are in meters.

`Environment(fields, depot, uavs).reset()` resets each UAV to the depot and
restores its energy and cumulative state. `UAV.execute_path(path, area=0.0)`
returns path length, energy consumption, remaining energy, and final position.
