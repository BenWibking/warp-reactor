# Primordial chemistry collapse

This repository compares CPU and GPU implementations of a primordial chemistry
collapse calculation while preserving an explicit numerical policy.

## Language

**Collapse state**:
The thermodynamic, species, time, density-driver, completion, and integrator data for one simulated cell.
_Avoid_: Cell state, burn state

**Grid-wide timestep**:
The stable minimum physical timestep selected across all participating cells for one global collapse step.
_Avoid_: Global timestep, local timestep

**Preparation kernel**:
The GPU pass that mutates scheduled perturbations, validates cells, and publishes each tile's grid-wide timestep candidate.
_Avoid_: Prepass, setup kernel

**Chemistry adapter**:
Either GPU kernel implementation that advances participating collapse states over the selected grid-wide timestep.
_Avoid_: Backend, driver

**Grid-wide runner**:
The host module that owns allocation, preparation, reduction, chemistry-adapter launch, copy-back, reporting, and final-state comparison.
_Avoid_: GPU driver, harness

## Relationships

- A **Grid-wide runner** selects exactly one **Chemistry adapter**.
- A **Preparation kernel** publishes candidates for one **Grid-wide timestep**.
- A **Chemistry adapter** advances zero or more **Collapse states** over that timestep.

## Example dialogue

> **Developer:** "May the structured chemistry adapter choose a different timestep?"
> **Domain expert:** "No. The grid-wide runner selects the grid-wide timestep before either chemistry adapter runs."

## Flagged ambiguities

- "Driver" previously meant both host orchestration and timestep policy; use **Grid-wide runner** for host orchestration and **Grid-wide timestep** for the numerical policy.
