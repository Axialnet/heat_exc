# Version 2: physics-aware and realism-enhanced dataset

This folder keeps the immutable V1 dataset in `Data/version 1` and creates a V2 reconstruction that is physically consistent and more realistic for ML or analysis workflows.

## What is included

- `generate_v2.py` — transforms the V1 CSV into a V2 dataset with explicit true-vs-measured states.
- `v2_physics_corrected.csv` — generated V2 output.
- `v2_metadata.json` — provenance and metadata about the transformation.

## V2 design principles

- Use original V1 scenarios as the boundary conditions.
- Reconstruct actual operating states (`T_true_C`, `m_dot_true_kg_s`).
- Recalculate hydraulics using coupled fluid-state equations.
- Simulate dynamic fouling with deposition and removal effects.
- Apply event-driven CIP cleaning with varying effectiveness.
- Add a temperature-dependent corrosion layer.
- Recompute overall heat transfer and efficiency.
- Add measured values (`T_in_measured_C`, `m_dot_measured_kg_s`) with sensor realism.

## Regenerate the output

Run:

```bash
python Data/version\ 2/generate_v2.py
```

or from the project root:

```bash
python "Data/version 2/generate_v2.py"
```
