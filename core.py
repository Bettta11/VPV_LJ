"""Shared OpenMM helpers for the Lennard-Jones project."""

from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import openmm as mm
    from openmm import app, unit
except ImportError as exc:  # pragma: no cover - exercised in runtime checks.
    mm = None
    app = None
    unit = None
    OPENMM_IMPORT_ERROR = exc
else:
    OPENMM_IMPORT_ERROR = None


def require_openmm() -> None:
    if OPENMM_IMPORT_ERROR is not None:
        raise RuntimeError(
            "OpenMM is required for MD runs. Install it with: python3 -m pip install --user openmm"
        ) from OPENMM_IMPORT_ERROR


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return data


def prepare_output_dir(config_path: str | Path, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out / "config.yaml")
    return out


def cubic_box_length(n_particles: int, density: float) -> float:
    if n_particles <= 0:
        raise ValueError("N must be positive.")
    if density <= 0:
        raise ValueError("rho must be positive.")
    return float((n_particles / density) ** (1.0 / 3.0))


def lattice_positions(n_particles: int, box_length: float) -> np.ndarray:
    n_side = math.ceil(n_particles ** (1.0 / 3.0))
    spacing = box_length / n_side
    coords: list[list[float]] = []
    offset = 0.5 * spacing
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                if len(coords) == n_particles:
                    return np.asarray(coords, dtype=float)
                coords.append([offset + ix * spacing, offset + iy * spacing, offset + iz * spacing])
    return np.asarray(coords, dtype=float)


def make_topology(n_particles: int, box_length: float) -> Any:
    require_openmm()
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("LJ", chain)
    element = app.Element.getByAtomicNumber(18)
    for _ in range(n_particles):
        topology.addAtom("Ar", element, residue)
    return topology


def create_lj_system(config: dict[str, Any], box_length: float) -> Any:
    require_openmm()
    n_particles = int(config["system"]["N"])
    mass = float(config["units"].get("mass", 1.0))
    sigma = float(config["units"].get("sigma", 1.0))
    epsilon = float(config["units"].get("epsilon", 1.0))
    rcut = float(config["potential"].get("rcut", 2.5))

    system = mm.System()
    for _ in range(n_particles):
        system.addParticle(mass * unit.dalton)

    system.setDefaultPeriodicBoxVectors(
        unit.Quantity(mm.Vec3(box_length, 0, 0), unit.nanometer),
        unit.Quantity(mm.Vec3(0, box_length, 0), unit.nanometer),
        unit.Quantity(mm.Vec3(0, 0, box_length), unit.nanometer),
    )

    expr = (
        "4*epsilon*((sigma/r)^12-(sigma/r)^6)"
        "-4*epsilon*((sigma/rcut)^12-(sigma/rcut)^6)"
    )
    force = mm.CustomNonbondedForce(expr)
    force.addGlobalParameter("sigma", sigma * unit.nanometer)
    force.addGlobalParameter("epsilon", epsilon * unit.kilojoule_per_mole)
    force.addGlobalParameter("rcut", rcut * unit.nanometer)
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(rcut * unit.nanometer)
    force.setUseLongRangeCorrection(False)
    for _ in range(n_particles):
        force.addParticle([])
    system.addForce(force)

    external = config.get("external_field", {})
    if external.get("type", "none") not in ("none", None):
        raise NotImplementedError("External fields are not implemented in stage 1.")

    return system


def create_integrator(config: dict[str, Any]) -> Any:
    require_openmm()
    epsilon = float(config["units"].get("epsilon", 1.0))
    gas_r = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
    temperature = (float(config["system"]["T"]) * epsilon / gas_r) * unit.kelvin
    friction = float(config["thermostat"].get("friction", 0.2)) / unit.picosecond
    dt = float(config["run"].get("dt", 0.002)) * unit.picoseconds
    return mm.LangevinMiddleIntegrator(temperature, friction, dt)


def create_simulation(config: dict[str, Any]) -> tuple[Any, Any, np.ndarray, float]:
    require_openmm()
    n_particles = int(config["system"]["N"])
    density = float(config["system"]["rho"])
    box_length = cubic_box_length(n_particles, density)
    positions = lattice_positions(n_particles, box_length)
    topology = make_topology(n_particles, box_length)
    system = create_lj_system(config, box_length)
    integrator = create_integrator(config)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions * unit.nanometer)
    epsilon = float(config["units"].get("epsilon", 1.0))
    gas_r = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
    temperature = (float(config["system"]["T"]) * epsilon / gas_r) * unit.kelvin
    simulation.context.setVelocitiesToTemperature(temperature)
    return simulation, topology, positions, box_length


def observables(
    simulation: Any,
    n_particles: int,
    box_length: float,
    step: int,
    epsilon: float = 1.0,
) -> dict[str, float]:
    require_openmm()
    state = simulation.context.getState(getEnergy=True)
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    dof = max(1, 3 * n_particles - 3)
    temperature = 2.0 * kinetic / (dof * epsilon)
    rho = n_particles / (box_length**3)
    pressure_ideal = rho * temperature
    return {
        "step": int(step),
        "T": float(temperature),
        "P": float(pressure_ideal),
        "U": float(potential),
        "K": float(kinetic),
        "E": float(potential + kinetic),
    }


def write_trace(path: str | Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise ValueError("No trace rows to write.")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "T", "P", "U", "K", "E"])
        writer.writeheader()
        writer.writerows(rows)
