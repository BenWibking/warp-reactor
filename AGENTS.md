# Repository Agent Instructions

## GPU Access

- Always run GPU discovery, GPU executables, and GPU profiling outside the
  sandbox. Request escalated execution instead of interpreting sandbox driver
  failures as host or application failures.
- Load the newest available CUDA module and verify `nvidia-smi` before running
  CUDA tests. Record the actual module, driver, GPU model, and compute
  capability; do not assume a requested module version is installed.
