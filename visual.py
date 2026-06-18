"""Visual LJ workflow for short demonstrative trajectories and previews."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

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


def _write_profiles_time(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "step", "bin", "z_min", "z_max", "z_center", "count"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_frame(path: Path, positions, box_length: float, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axis = plt.subplots(figsize=(5, 5))
    colors = positions[:, 2]
    scatter = axis.scatter(positions[:, 0], positions[:, 1], c=colors, s=8, cmap="viridis", alpha=0.85)
    axis.set_xlim(0.0, box_length)
    axis.set_ylim(0.0, box_length)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(title)
    fig.colorbar(scatter, ax=axis, label="z")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_visual_trace(out_dir: Path, rows: list[dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    steps = [row["step"] for row in rows]

    plt.figure(figsize=(6, 4))
    plt.plot(steps, [row["T"] for row in rows], label="T")
    plt.xlabel("step")
    plt.ylabel("T")
    plt.tight_layout()
    plt.savefig(figures / "temperature_trace.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(steps, [row["P"] for row in rows], label="P")
    plt.xlabel("step")
    plt.ylabel("P")
    plt.tight_layout()
    plt.savefig(figures / "pressure_trace.png", dpi=160)
    plt.close()


def run_visual(config_path: str = "configs/visual.yaml") -> str:
    config = load_config(config_path)
    if config.get("project") != "visual":
        raise ValueError("Visual runner expects project: visual.")
    if config.get("external_field", {}).get("type", "none") != "none":
        raise ValueError("This compact visual workflow expects external_field.type: none.")

    out_dir = prepare_output_dir(config_path, config["output_dir"])
    (out_dir / "figures").mkdir(exist_ok=True)
    frames_dir = out_dir / "preview_frames"
    frames_dir.mkdir(exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)

    simulation, topology, positions, box_length = create_simulation(config)
    n_particles = int(config["system"]["N"])
    epsilon = float(config["units"].get("epsilon", 1.0))
    rcut = float(config["potential"].get("rcut", 2.5))
    sigma = float(config["units"].get("sigma", 1.0))
    equil_steps = int(config["run"].get("equil_steps", 0))
    prod_steps = int(config["run"].get("prod_steps", 0))
    trace_interval = int(config["save"].get("trace", {}).get("interval", 100))
    trajectory_cfg = config["save"].get("trajectory", {})
    trajectory_interval = int(trajectory_cfg.get("interval", 100))
    max_frames = int(trajectory_cfg.get("max_frames", 40))
    profile_cfg = config["save"].get("profiles_time", {})
    profile_bins = int(profile_cfg.get("bins_z", 40))

    with (out_dir / "topology.pdb").open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(topology, positions * unit.nanometer, handle)

    if trajectory_cfg.get("enabled", True):
        simulation.reporters.append(app.DCDReporter(str(out_dir / "trajectory.dcd"), trajectory_interval))

    if equil_steps:
        simulation.step(equil_steps)

    trace_rows: list[dict[str, float]] = []
    profile_rows: list[dict[str, float]] = []
    frame_index = 0
    completed = 0
    next_frame_at = trajectory_interval
    with (out_dir / "log.txt").open("w", encoding="utf-8") as log:
        log.write("visual LJ run\n")
        log.write("visual_data_used_for_fit=false\n")
        log.write(f"N={n_particles} box_length={box_length:.6f}\n")
        while completed < prod_steps and frame_index < max_frames:
            next_trace_at = ((completed // trace_interval) + 1) * trace_interval
            target = min(prod_steps, next_trace_at, next_frame_at)
            chunk = max(1, target - completed)
            simulation.step(chunk)
            completed += chunk

            if completed == next_trace_at or completed == prod_steps:
                row = observables(simulation, n_particles, box_length, equil_steps + completed, epsilon)
                current_positions = get_positions_array(simulation)
                row["P"] = lj_virial_pressure(current_positions, box_length, row["T"], rcut, sigma, epsilon)
                trace_rows.append(row)

            if completed == next_frame_at or completed == prod_steps:
                current_positions = get_positions_array(simulation)
                frame_path = frames_dir / f"frame_{frame_index:04d}.png"
                _plot_frame(frame_path, current_positions, box_length, f"step {equil_steps + completed}")
                if profile_cfg.get("enabled", True):
                    for profile in z_profile_counts(current_positions, box_length, profile_bins):
                        profile_rows.append({"frame": frame_index, "step": equil_steps + completed, **profile})
                frame_index += 1
                next_frame_at += trajectory_interval

        log.write(f"prod_steps_completed={completed}\n")
        log.write(f"trace_rows={len(trace_rows)}\n")
        log.write(f"preview_frames={frame_index}\n")

    write_trace(out_dir / "state_trace.csv", trace_rows)
    if profile_rows:
        _write_profiles_time(out_dir / "profiles_time.csv", profile_rows)
    _plot_visual_trace(out_dir, trace_rows)
    render_video(str(out_dir))
    return str(out_dir)


def render_video(visual_dir: str) -> str | None:
    """Create a short mp4 from preview PNG frames when imageio or ffmpeg is available."""
    out_dir = Path(visual_dir)
    frames = sorted((out_dir / "preview_frames").glob("frame_*.png"))
    if not frames:
        return None
    video_path = out_dir / "videos" / "preview.mp4"
    try:
        import imageio.v3 as iio
    except ImportError:
        iio = None
    if iio is not None:
        images = [iio.imread(frame) for frame in frames]
        try:
            iio.imwrite(video_path, images, fps=8)
        except Exception:
            pass
        else:
            return str(video_path)

    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        "8",
        "-i",
        str(out_dir / "preview_frames" / "frame_%04d.png"),
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return str(video_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/visual.yaml")
    args = parser.parse_args()
    print(run_visual(args.config))


if __name__ == "__main__":
    main()
