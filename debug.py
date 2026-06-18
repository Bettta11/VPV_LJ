"""Short debug run for the stage 1 Lennard-Jones OpenMM core."""

from __future__ import annotations

import argparse
import math

from openmm import app, unit

from core import create_simulation, load_config, observables, prepare_output_dir, write_trace


def run_debug(config_path: str = "configs/debug.yaml") -> str:
    config = load_config(config_path)
    out_dir = prepare_output_dir(config_path, config["output_dir"])
    log_path = out_dir / "log.txt"

    simulation, topology, positions, box_length = create_simulation(config)
    n_particles = int(config["system"]["N"])
    epsilon = float(config["units"].get("epsilon", 1.0))
    trace_interval = int(config["save"]["trace"].get("interval", 100))
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
            rows.append(row)
            if not math.isfinite(row["T"]):
                raise RuntimeError("Temperature became non-finite.")
            if row["T"] > 1000.0:
                raise RuntimeError(f"Temperature is unstable: {row['T']:.3f} K")
            if completed // trajectory_interval >= max_frames:
                break

        log.write(f"prod_steps_completed={completed}\n")
        log.write(f"trace_rows={len(rows)}\n")

    write_trace(out_dir / "state_trace.csv", rows)
    return str(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/debug.yaml")
    args = parser.parse_args()
    out_dir = run_debug(args.config)
    print(out_dir)


if __name__ == "__main__":
    main()
