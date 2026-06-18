"""EOS workflow placeholder.

Stage 1 intentionally does not implement the EOS sweep. The implementation
belongs to stage 2, after the shared OpenMM core and debug run are verified.
"""


def run_eos(config_path: str = "configs/eos.yaml") -> str:
    raise NotImplementedError("EOS sweep is scheduled for stage 2.")


def fit_vdw(eos_dir: str) -> dict:
    raise NotImplementedError("van der Waals fitting is scheduled for stage 3.")


def plot_eos_results(eos_dir: str) -> None:
    raise NotImplementedError("EOS plotting is scheduled for stage 2/3.")

