# VPV_LJ

Compact OpenMM project for a small Lennard-Jones molecular dynamics study:
building numerical EOS data and comparing it with the van der Waals equation.

Current implementation status:

- Stage 1: project scaffold, shared OpenMM core, and a short debug run.
- Stage 2: EOS sweep and tabular outputs are not implemented yet.
- Stage 3: van der Waals fitting and plots are not implemented yet.
- Stage 4: full visual workflows and notebook orchestration are not implemented yet.

## Layout

    cloud_runner.ipynb
    core.py
    debug.py
    eos.py
    visual.py
    configs/
    data/
    report_assets/

## Stage 1 debug check

    python3 debug.py configs/debug.yaml

Expected generated files:

- data/debug/debug_001/config.yaml
- data/debug/debug_001/state_trace.csv
- data/debug/debug_001/trajectory.dcd
- data/debug/debug_001/topology.pdb

Generated data and trajectories are ignored by Git.

