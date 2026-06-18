"""Visual workflow placeholder.

Stage 1 only verifies the shared OpenMM core through the debug workflow.
Visual trajectories and video rendering are scheduled for stage 4.
"""


def run_visual(config_path: str = "configs/visual.yaml") -> str:
    raise NotImplementedError("Visual workflow is scheduled for stage 4.")


def render_video(visual_dir: str) -> None:
    raise NotImplementedError("Video rendering is scheduled for stage 4.")

