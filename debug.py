"""Short diagnostic runs for the Lennard-Jones OpenMM core."""

from __future__ import annotations

import argparse
import csv
import math

from openmm import app, unit

from core import (
    create_simulation,
    get_positions_array,
    lj_virial_pressure,
    load_config,
    observables,
    prepare_output_dir,
    write_trace,
    z_profile_counts,
)


def _write_profiles(path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bin", "z_min", "z_max", "z_center", "count"])
        writer.writeheader()
        writer.writerows(rows)


def _plot_debug_outputs(out_dir, trace_rows, profile_rows, final_positions, box_length) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    steps = [row["step"] for row in trace_rows]

    plt.figure(figsize=(6, 4))
    plt.plot(steps, [row["T"] for row in trace_rows])
    plt.xlabel("step")
    plt.ylabel("T")
    plt.tight_layout()
    plt.savefig(figures / "temperature_trace.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(steps, [row["P"] for row in trace_rows])
    plt.xlabel("step")
    plt.ylabel("P")
    plt.tight_layout()
    plt.savefig(figures / "pressure_trace.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(steps, [row["E"] for row in trace_rows], label="E")
    plt.plot(steps, [row["U"] for row in trace_rows], label="U")
    plt.plot(steps, [row["K"] for row in trace_rows], label="K")
    plt.xlabel("step")
    plt.ylabel("energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "energy_trace.png", dpi=160)
    plt.close()

    if profile_rows:
        plt.figure(figsize=(6, 4))
        plt.plot([row["z_center"] for row in profile_rows], [row["count"] for row in profile_rows])
        plt.xlabel("z")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(figures / "final_profile.png", dpi=160)
        plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(final_positions[:, 0], final_positions[:, 1], s=10, alpha=0.8)
    plt.xlim(0.0, box_length)
    plt.ylim(0.0, box_length)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(figures / "snapshot.png", dpi=160)
    plt.close()


def run_debug(config_path: str = "configs/debug.yaml") -> str:
    config = load_config(config_path)
    out_dir = prepare_output_dir(config_path, config["output_dir"])
    log_path = out_dir / "log.txt"

    simulation, topology, positions, box_length = create_simulation(config)
    n_particles = int(config["system"]["N"])
    epsilon = float(config["units"].get("epsilon", 1.0))
    trace_interval = int(config["save"]["trace"].get("interval", 100))
    profile_cfg = config["save"].get("profiles", {})
    profile_bins = int(profile_cfg.get("bins_z", 40))
    trajectory_cfg = config["save"].get("trajectory", {})
    trajectory_interval = int(trajectory_cfg.get("interval", 200))
    max_frames = int(trajectory_cfg.get("max_frames", 200))
    equil_steps = int(config["run"].get("equil_steps", 0))
    prod_steps = int(config["run"].get("prod_steps", 0))

    with (out_dir / "topology.pdb").open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(topology, positions * unit.nanometer, handle)

    if trajectory_cfg.get("enabled", True):
        simulation.reporters.append(
            app.DCDReporter(str(out_dir / "trajectory.dcd"), trajectory_interval)
        )

    rows = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write("debug LJ run\n")
        log.write(f"N={n_particles} box_length={box_length:.6f}\n")
        if equil_steps:
            log.write(f"equil_steps={equil_steps}\n")
            simulation.step(equil_steps)

        completed = 0
        while completed < prod_steps:
            chunk = min(trace_interval, prod_steps - completed)
            simulation.step(chunk)
            completed += chunk
            row = observables(simulation, n_particles, box_length, equil_steps + completed, epsilon)
            positions_now = get_positions_array(simulation)
            row["P"] = lj_virial_pressure(
                positions_now,
                box_length,
                row["T"],
                float(config["potential"].get("rcut", 2.5)),
                float(config["units"].get("sigma", 1.0)),
                epsilon,
            )
            rows.append(row)
            if not math.isfinite(row["T"]):
                raise RuntimeError("Temperature became non-finite.")
            if row["T"] > 1000.0:
                raise RuntimeError(f"Temperature is unstable: {row['T']:.3f} K")
            if completed // trajectory_interval >= max_frames:
                break

        log.write(f"prod_steps_completed={completed}\n")
        log.write(f"trace_rows={len(rows)}\n")

    final_positions = get_positions_array(simulation)
    profile_rows = []
    if profile_cfg.get("enabled", True):
        profile_rows = z_profile_counts(final_positions, box_length, profile_bins)
        _write_profiles(out_dir / "profiles.csv", profile_rows)
    write_trace(out_dir / "state_trace.csv", rows)
    _plot_debug_outputs(out_dir, rows, profile_rows, final_positions, box_length)
    return str(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/debug.yaml")
    args = parser.parse_args()
    out_dir = run_debug(args.config)
    print(out_dir)


if __name__ == "__main__":
    main()
