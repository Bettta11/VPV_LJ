"""EOS sweep and stage-2 tabular outputs for the LJ project."""

from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from core import (
    create_simulation,
    get_positions_array,
    lj_virial_pressure,
    load_config,
    observables,
    prepare_output_dir,
    z_profile_counts,
)


EOS_POINT_FIELDS = [
    "run_id",
    "T_target",
    "T_mean",
    "rho_target",
    "rho_mean",
    "N",
    "Lx",
    "Ly",
    "Lz",
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


def _sample_stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), 0.0, 0.0
    sigma = stdev(values)
    return float(mean(values)), float(sigma), float(sigma / math.sqrt(len(values)))


def _point_config(base: dict[str, Any], temperature: float, density: float, seed: int) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["seed"] = int(seed)
    config.setdefault("system", {})
    config["system"]["T"] = float(temperature)
    config["system"]["rho"] = float(density)
    _validate_eos_config(config)
    return config


def _validate_eos_config(config: dict[str, Any]) -> None:
    config.setdefault("external_field", {})
    if config["external_field"].get("type", "none") != "none":
        raise ValueError("EOS must use external_field.type: none.")
    if float(config["external_field"].get("g", 0.0)) != 0.0:
        raise ValueError("EOS must use external_field.g: 0.0.")
    if config.get("save", {}).get("trajectory", {}).get("enabled", False):
        raise ValueError("EOS must not save trajectories.")


def _run_one_point(base_config: dict[str, Any], run_id: str, temperature: float, density: float, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _point_config(base_config, temperature, density, seed)
    simulation, _topology, _positions, box_length = create_simulation(config)

    n_particles = int(config["system"]["N"])
    dt = float(config["run"].get("dt", 0.002))
    equil_steps = int(config["run"].get("equil_steps", 0))
    prod_steps = int(config["run"].get("prod_steps", 0))
    sample_interval = int(config.get("save", {}).get("trace", {}).get("interval", 1000))
    sample_interval = max(1, sample_interval)
    epsilon = float(config["units"].get("epsilon", 1.0))
    sigma = float(config["units"].get("sigma", 1.0))
    rcut = float(config["potential"].get("rcut", 2.5))

    if equil_steps:
        simulation.step(equil_steps)

    samples: list[dict[str, float]] = []
    completed = 0
    status = "ok"
    notes = ""
    while completed < prod_steps:
        chunk = min(sample_interval, prod_steps - completed)
        simulation.step(chunk)
        completed += chunk
        sample = observables(simulation, n_particles, box_length, equil_steps + completed, epsilon)
        positions = get_positions_array(simulation)
        sample["P"] = lj_virial_pressure(positions, box_length, sample["T"], rcut, sigma, epsilon)
        samples.append(sample)
        if not math.isfinite(sample["T"]) or sample["T"] > 10.0 * temperature:
            status = "unstable_temperature"
            notes = "temperature became non-finite or too high"
            break

    final_positions = get_positions_array(simulation)
    profile_cfg = config.get("save", {}).get("final_profile", {})
    bins_z = int(profile_cfg.get("bins_z", 40))
    profile_rows = []
    if profile_cfg.get("enabled", True):
        for row in z_profile_counts(final_positions, box_length, bins_z):
            profile_rows.append({"run_id": run_id, **row})

    t_mean, _t_std, _t_sem = _sample_stats([row["T"] for row in samples])
    p_mean, p_std, p_sem = _sample_stats([row["P"] for row in samples])
    u_mean, u_std, u_sem = _sample_stats([row["U"] for row in samples])
    k_mean, _k_std, _k_sem = _sample_stats([row["K"] for row in samples])
    e_mean, _e_std, _e_sem = _sample_stats([row["E"] for row in samples])

    if not samples:
        status = "failed"
        notes = "no production samples"

    point = {
        "run_id": run_id,
        "T_target": float(temperature),
        "T_mean": t_mean,
        "rho_target": float(density),
        "rho_mean": n_particles / (box_length**3),
        "N": n_particles,
        "Lx": box_length,
        "Ly": box_length,
        "Lz": box_length,
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
    return point, profile_rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_eos(config_path: str = "configs/eos.yaml") -> str:
    config = load_config(config_path)
    if config.get("project") != "eos":
        raise ValueError("EOS runner expects project: eos.")
    _validate_eos_config(config)
    out_dir = prepare_output_dir(config_path, config["output_dir"])
    (out_dir / "figures").mkdir(exist_ok=True)

    points: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    sweep = config["sweep"]
    run_index = 0

    with (out_dir / "log.txt").open("w", encoding="utf-8") as log:
        log.write("EOS LJ sweep\n")
        log.write("trajectory_enabled=false\n")
        log.write("external_field=none\n")
        for temperature in sweep["temperatures"]:
            for density in sweep["densities"]:
                for seed in sweep.get("seeds", [1]):
                    run_index += 1
                    run_id = f"eos_{run_index:04d}"
                    log.write(f"start {run_id} T={temperature} rho={density} seed={seed}\n")
                    try:
                        point, profile_rows = _run_one_point(config, run_id, temperature, density, seed)
                    except Exception as exc:  # Keep the table complete for failed points.
                        point = {
                            "run_id": run_id,
                            "T_target": float(temperature),
                            "T_mean": math.nan,
                            "rho_target": float(density),
                            "rho_mean": math.nan,
                            "N": int(config["system"]["N"]),
                            "Lx": math.nan,
                            "Ly": math.nan,
                            "Lz": math.nan,
                            "dt": float(config["run"].get("dt", 0.002)),
                            "equil_steps": int(config["run"].get("equil_steps", 0)),
                            "prod_steps": 0,
                            "seed": int(seed),
                            "P_mean": math.nan,
                            "P_std": math.nan,
                            "P_sem": math.nan,
                            "U_mean": math.nan,
                            "U_std": math.nan,
                            "U_sem": math.nan,
                            "K_mean": math.nan,
                            "E_mean": math.nan,
                            "status": "failed",
                            "notes": str(exc),
                        }
                        profile_rows = []
                    points.append(point)
                    profiles.extend(profile_rows)
                    log.write(f"finish {run_id} status={point['status']}\n")

    _write_csv(out_dir / "eos_points.csv", EOS_POINT_FIELDS, points)
    _write_csv(out_dir / "eos_final_profiles.csv", PROFILE_FIELDS, profiles)
    plot_eos_results(str(out_dir))
    return str(out_dir)


def _read_points(eos_dir: str | Path) -> list[dict[str, str]]:
    with (Path(eos_dir) / "eos_points.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_profiles(eos_dir: str | Path) -> list[dict[str, str]]:
    with (Path(eos_dir) / "eos_final_profiles.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_eos_results(eos_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir = Path(eos_dir)
    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    rows = [row for row in _read_points(out_dir) if row["status"] == "ok"]
    if not rows:
        return

    temperatures = sorted({float(row["T_target"]) for row in rows})

    plt.figure(figsize=(6, 4))
    for temperature in temperatures:
        subset = sorted(
            [row for row in rows if float(row["T_target"]) == temperature],
            key=lambda row: float(row["rho_target"]),
        )
        plt.plot(
            [float(row["rho_target"]) for row in subset],
            [float(row["P_mean"]) for row in subset],
            marker="o",
            label=f"T={temperature:g}",
        )
    plt.xlabel("rho")
    plt.ylabel("P")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "eos_isotherms.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    for temperature in temperatures:
        subset = sorted(
            [row for row in rows if float(row["T_target"]) == temperature],
            key=lambda row: float(row["rho_target"]),
        )
        plt.plot(
            [float(row["rho_target"]) for row in subset],
            [float(row["U_mean"]) for row in subset],
            marker="o",
            label=f"T={temperature:g}",
        )
    plt.xlabel("rho")
    plt.ylabel("U")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "eos_energy.png", dpi=160)
    plt.close()

    profiles = _read_profiles(out_dir)
    if profiles:
        selected_runs = sorted({row["run_id"] for row in profiles})[:6]
        plt.figure(figsize=(6, 4))
        for run_id in selected_runs:
            subset = [row for row in profiles if row["run_id"] == run_id]
            plt.plot(
                [float(row["z_center"]) for row in subset],
                [float(row["count"]) for row in subset],
                label=run_id,
            )
        plt.xlabel("z")
        plt.ylabel("count")
        plt.legend(fontsize="small")
        plt.tight_layout()
        plt.savefig(figures / "profile_overview.png", dpi=160)
        plt.close()


def fit_vdw(eos_dir: str) -> dict:
    raise NotImplementedError("van der Waals fitting is scheduled for stage 3.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/eos.yaml")
    args = parser.parse_args()
    print(run_eos(args.config))


if __name__ == "__main__":
    main()
