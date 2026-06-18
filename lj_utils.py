"""Small helper functions for the LJ -> EOS -> VdW notebook."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

try:
    import openmm as mm
    from openmm import app, unit
except ImportError as exc:  # pragma: no cover
    mm = None
    app = None
    unit = None
    OPENMM_IMPORT_ERROR = exc
else:
    OPENMM_IMPORT_ERROR = None


EOS_FIELDS = [
    "run_id",
    "T_target",
    "T_mean",
    "rho_target",
    "rho_mean",
    "N",
    "L",
    "dt",
    "equil_steps",
    "prod_steps",
    "seed",
    "P_mean",
    "P_std",
    "P_sem",
    "U_mean",
    "U_std",
    "U_sem",
    "K_mean",
    "E_mean",
    "status",
    "notes",
]

PROFILE_FIELDS = ["run_id", "bin", "z_min", "z_max", "z_center", "count"]


def require_openmm() -> None:
    if OPENMM_IMPORT_ERROR is not None:
        raise RuntimeError("OpenMM is required. Install dependencies from requirements.txt.") from OPENMM_IMPORT_ERROR


def cubic_box_length(n_particles: int, density: float) -> float:
    return float((n_particles / density) ** (1.0 / 3.0))


def lattice_positions(n_particles: int, box_length: float) -> np.ndarray:
    n_side = math.ceil(n_particles ** (1.0 / 3.0))
    spacing = box_length / n_side
    coords = []
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                if len(coords) == n_particles:
                    return np.asarray(coords, dtype=float)
                coords.append([(ix + 0.5) * spacing, (iy + 0.5) * spacing, (iz + 0.5) * spacing])
    return np.asarray(coords, dtype=float)


def make_topology(n_particles: int) -> Any:
    require_openmm()
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("LJ", chain)
    element = app.Element.getByAtomicNumber(18)
    for _ in range(n_particles):
        topology.addAtom("Ar", element, residue)
    return topology


def create_lj_simulation(params: dict[str, Any], temperature: float, density: float, seed: int) -> tuple[Any, float]:
    """Create a periodic NVT LJ simulation in reduced units."""
    require_openmm()
    n_particles = int(params["N"])
    mass = float(params.get("mass", 1.0))
    sigma = float(params.get("sigma", 1.0))
    epsilon = float(params.get("epsilon", 1.0))
    rcut = float(params.get("rcut", 2.5))
    friction = float(params.get("friction", 0.2))
    dt = float(params.get("dt", 0.002))
    box_length = cubic_box_length(n_particles, density)

    system = mm.System()
    for _ in range(n_particles):
        system.addParticle(mass * unit.dalton)
    system.setDefaultPeriodicBoxVectors(
        unit.Quantity(mm.Vec3(box_length, 0, 0), unit.nanometer),
        unit.Quantity(mm.Vec3(0, box_length, 0), unit.nanometer),
        unit.Quantity(mm.Vec3(0, 0, box_length), unit.nanometer),
    )

    expression = "4*epsilon*((sigma/r)^12-(sigma/r)^6)-4*epsilon*((sigma/rcut)^12-(sigma/rcut)^6)"
    force = mm.CustomNonbondedForce(expression)
    force.addGlobalParameter("sigma", sigma * unit.nanometer)
    force.addGlobalParameter("epsilon", epsilon * unit.kilojoule_per_mole)
    force.addGlobalParameter("rcut", rcut * unit.nanometer)
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(rcut * unit.nanometer)
    force.setUseLongRangeCorrection(False)
    for _ in range(n_particles):
        force.addParticle([])
    system.addForce(force)

    gas_r = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
    openmm_temperature = (temperature * epsilon / gas_r) * unit.kelvin
    integrator = mm.LangevinMiddleIntegrator(openmm_temperature, friction / unit.picosecond, dt * unit.picoseconds)
    integrator.setRandomNumberSeed(int(seed))

    topology = make_topology(n_particles)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(lattice_positions(n_particles, box_length) * unit.nanometer)
    simulation.context.setVelocitiesToTemperature(openmm_temperature, int(seed))
    return simulation, box_length


def get_positions(simulation: Any) -> np.ndarray:
    require_openmm()
    state = simulation.context.getState(getPositions=True)
    return np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=float)


def observables(simulation: Any, n_particles: int, box_length: float, epsilon: float = 1.0) -> dict[str, float]:
    require_openmm()
    state = simulation.context.getState(getEnergy=True)
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    dof = max(1, 3 * n_particles - 3)
    temperature = 2.0 * kinetic / (dof * epsilon)
    return {"T": float(temperature), "U": float(potential), "K": float(kinetic), "E": float(potential + kinetic)}


def lj_virial_pressure(
    positions: np.ndarray,
    box_length: float,
    temperature: float,
    rcut: float = 2.5,
    sigma: float = 1.0,
    epsilon: float = 1.0,
) -> float:
    volume = box_length**3
    rho = len(positions) / volume
    virial = 0.0
    rcut2 = rcut * rcut
    for i in range(len(positions) - 1):
        delta = positions[i + 1 :] - positions[i]
        delta -= box_length * np.rint(delta / box_length)
        r2 = np.sum(delta * delta, axis=1)
        mask = (r2 > 0.0) & (r2 < rcut2)
        if not np.any(mask):
            continue
        inv_r2 = (sigma * sigma) / r2[mask]
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        virial += float(np.sum(24.0 * epsilon * (2.0 * inv_r12 - inv_r6)))
    return float(rho * temperature + virial / (3.0 * volume))


def profile_counts(positions: np.ndarray, box_length: float, bins_z: int, run_id: str) -> list[dict[str, Any]]:
    counts, edges = np.histogram(positions[:, 2] % box_length, bins=bins_z, range=(0.0, box_length))
    rows = []
    for index, count in enumerate(counts):
        z_min = float(edges[index])
        z_max = float(edges[index + 1])
        rows.append(
            {
                "run_id": run_id,
                "bin": index,
                "z_min": z_min,
                "z_max": z_max,
                "z_center": 0.5 * (z_min + z_max),
                "count": int(count),
            }
        )
    return rows


def sample_stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), 0.0, 0.0
    sigma = stdev(values)
    return float(mean(values)), float(sigma), float(sigma / math.sqrt(len(values)))


def run_eos_point(params: dict[str, Any], temperature: float, density: float, seed: int, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one EOS point and return one table row plus final z-profile counts."""
    n_particles = int(params["N"])
    equil_steps = int(params.get("equil_steps", 1000))
    prod_steps = int(params.get("prod_steps", 2000))
    sample_interval = int(params.get("sample_interval", 500))
    dt = float(params.get("dt", 0.002))
    epsilon = float(params.get("epsilon", 1.0))
    sigma = float(params.get("sigma", 1.0))
    rcut = float(params.get("rcut", 2.5))

    simulation, box_length = create_lj_simulation(params, temperature, density, seed)
    if equil_steps:
        simulation.step(equil_steps)

    samples = []
    completed = 0
    status = "ok"
    notes = ""
    while completed < prod_steps:
        chunk = min(sample_interval, prod_steps - completed)
        simulation.step(chunk)
        completed += chunk
        row = observables(simulation, n_particles, box_length, epsilon)
        positions = get_positions(simulation)
        row["P"] = lj_virial_pressure(positions, box_length, row["T"], rcut, sigma, epsilon)
        samples.append(row)
        if not math.isfinite(row["T"]) or row["T"] > 10.0 * temperature:
            status = "unstable_temperature"
            notes = "temperature became non-finite or too high"
            break

    t_mean, _, _ = sample_stats([row["T"] for row in samples])
    p_mean, p_std, p_sem = sample_stats([row["P"] for row in samples])
    u_mean, u_std, u_sem = sample_stats([row["U"] for row in samples])
    k_mean, _, _ = sample_stats([row["K"] for row in samples])
    e_mean, _, _ = sample_stats([row["E"] for row in samples])
    final_positions = get_positions(simulation)

    point = {
        "run_id": run_id,
        "T_target": float(temperature),
        "T_mean": t_mean,
        "rho_target": float(density),
        "rho_mean": n_particles / (box_length**3),
        "N": n_particles,
        "L": box_length,
        "dt": dt,
        "equil_steps": equil_steps,
        "prod_steps": completed,
        "seed": int(seed),
        "P_mean": p_mean,
        "P_std": p_std,
        "P_sem": p_sem,
        "U_mean": u_mean,
        "U_std": u_std,
        "U_sem": u_sem,
        "K_mean": k_mean,
        "E_mean": e_mean,
        "status": status,
        "notes": notes,
    }
    profiles = profile_counts(final_positions, box_length, int(params.get("profile_bins", 40)), run_id)
    return point, profiles


def write_csv(path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def vdw_pressure(rho: float | np.ndarray, temperature: float, a: float, b: float) -> float | np.ndarray:
    return rho * temperature / (1.0 - b * rho) - a * rho * rho


def _fit_a_for_b(data: list[tuple[float, float, float]], b: float) -> float:
    numerator = 0.0
    denominator = 0.0
    for rho, temperature, pressure in data:
        base = rho * temperature / (1.0 - b * rho)
        x = rho * rho
        numerator += x * (base - pressure)
        denominator += x * x
    return max(0.0, numerator / denominator) if denominator else 0.0


def fit_vdw(points: list[dict[str, Any]], rho_max: float | None = None) -> dict[str, Any]:
    data = []
    for row in points:
        if row.get("status") != "ok":
            continue
        rho = float(row["rho_mean"])
        if rho_max is not None and rho > rho_max:
            continue
        values = (rho, float(row["T_mean"]), float(row["P_mean"]))
        if all(math.isfinite(value) for value in values):
            data.append(values)
    if len(data) < 3:
        raise ValueError("Need at least 3 good EOS points for VdW fit.")

    upper = 0.95 / max(rho for rho, _, _ in data)
    lower = 0.0
    phi = (math.sqrt(5.0) - 1.0) / 2.0

    def score(b: float) -> tuple[float, float]:
        a = _fit_a_for_b(data, b)
        sse = sum((pressure - vdw_pressure(rho, temperature, a, b)) ** 2 for rho, temperature, pressure in data)
        return float(sse), float(a)

    x1 = upper - phi * (upper - lower)
    x2 = lower + phi * (upper - lower)
    f1, _ = score(x1)
    f2, _ = score(x2)
    for _ in range(160):
        if f1 > f2:
            lower = x1
            x1 = x2
            f1 = f2
            x2 = lower + phi * (upper - lower)
            f2, _ = score(x2)
        else:
            upper = x2
            x2 = x1
            f2 = f1
            x1 = upper - phi * (upper - lower)
            f1, _ = score(x1)
        if abs(upper - lower) < 1.0e-10:
            break
    b = 0.5 * (lower + upper)
    _, a = score(b)
    residuals = [
        {
            "rho": rho,
            "T": temperature,
            "P_md": pressure,
            "P_vdw": float(vdw_pressure(rho, temperature, a, b)),
            "residual": float(pressure - vdw_pressure(rho, temperature, a, b)),
        }
        for rho, temperature, pressure in data
    ]
    return {
        "a": float(a),
        "b": float(b),
        "n_fit_points": len(data),
        "fit_region": {"status": ["ok"], "rho_max": rho_max, "source": "eos_points in notebook"},
        "residuals": residuals,
    }


def plot_eos_isotherms(points: list[dict[str, Any]], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in points if row.get("status") == "ok"]
    temperatures = sorted({float(row["T_target"]) for row in rows})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for temperature in temperatures:
        subset = sorted([row for row in rows if float(row["T_target"]) == temperature], key=lambda row: float(row["rho_mean"]))
        plt.plot([row["rho_mean"] for row in subset], [row["P_mean"] for row in subset], marker="o", label=f"T={temperature:g}")
    plt.xlabel("rho")
    plt.ylabel("P")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_vdw_fit(points: list[dict[str, Any]], fit: dict[str, Any], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in points if row.get("status") == "ok"]
    temperatures = sorted({float(row["T_target"]) for row in rows})
    rho_values = [float(row["rho_mean"]) for row in rows]
    rho_grid = np.linspace(max(1.0e-9, 0.9 * min(rho_values)), 1.05 * max(rho_values), 200)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for temperature in temperatures:
        subset = sorted([row for row in rows if float(row["T_target"]) == temperature], key=lambda row: float(row["rho_mean"]))
        t_curve = mean(float(row["T_mean"]) for row in subset)
        plt.plot([row["rho_mean"] for row in subset], [row["P_mean"] for row in subset], marker="o", linestyle="none", label=f"MD T={temperature:g}")
        plt.plot(rho_grid, vdw_pressure(rho_grid, t_curve, fit["a"], fit["b"]), label=f"VdW T={temperature:g}")
    plt.xlabel("rho")
    plt.ylabel("P")
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
