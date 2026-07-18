# Repository Agent Instructions

## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues. See
`docs/agents/issue-tracker.md`.

### Triage labels

Use the default five-role triage vocabulary. See
`docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository. See `docs/agents/domain.md`.

## GPU Access

- Always run GPU discovery, GPU executables, and GPU profiling outside the
  sandbox. Request escalated execution instead of interpreting sandbox driver
  failures as host or application failures.
- Load the newest available CUDA module and verify `nvidia-smi` before running
  CUDA tests. Record the actual module, driver, GPU model, and compute
  capability; do not assume a requested module version is installed.
- Before each timing or profiling batch, inspect utilization, memory use, and
  active compute processes on every GPU, then run on the least-loaded device.
  Record the selected GPU with the result.
