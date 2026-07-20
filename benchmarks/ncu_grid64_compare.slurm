#!/usr/bin/env bash
#SBATCH --job-name=chem-ncu-grid64
#SBATCH --account=general
#SBATCH --partition=general-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=benchmark_runs/ncu-grid64-%j.out
#SBATCH --error=benchmark_runs/ncu-grid64-%j.err

set -euo pipefail

repo=/mnt/ffs24/home/wibkingb/mojo-chemistry
run_dir="${repo}/benchmark_runs/ncu-grid64-${SLURM_JOB_ID}"
structured_bin="${repo}/benchmark_runs/fused-12752255/bin/reproducer_structured"
cuda_bin="${repo}/benchmark_runs/fused-12752255/bin/reproducer_cuda"
mkdir -p "${run_dir}"
cd "${run_dir}"

module purge
available_cuda="$({ module -t avail CUDA 2>&1 || true; } | sed -n 's/^[[:space:]]*\(CUDA\/[0-9][^[:space:]]*\).*$/\1/p' | sort -V)"
cuda_module="$(printf '%s\n' "${available_cuda}" | tail -n 1)"
if [[ -z "${cuda_module}" ]]; then
    echo "No CUDA module found" >&2
    exit 1
fi
module load "${cuda_module}"

{
    date --iso-8601=seconds
    hostname
    echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
    echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "SELECTED_GPU=${CUDA_VISIBLE_DEVICES:-0} (only allocated H200)"
    echo "AVAILABLE_CUDA_MODULES_BEGIN"
    printf '%s\n' "${available_cuda}"
    echo "AVAILABLE_CUDA_MODULES_END"
    echo "LOADED_CUDA_MODULE=${cuda_module}"
    module -t list
    nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap --format=csv,noheader
    ncu --version
    sha256sum "${structured_bin}" "${cuda_bin}"
    git -C "${repo}" rev-parse HEAD
    git -C "${repo}" status --short
} >"${run_dir}/environment.txt" 2>&1

snapshot_gpu() {
    local label=$1
    nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,utilization.gpu,memory.used,memory.total --format=csv,noheader \
        >"${run_dir}/gpu-before-${label}.csv"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
        >"${run_dir}/gpu-processes-before-${label}.csv"
}

profile_one() {
    local implementation=$1
    local phase=$2
    local skip=$3
    local kernel_regex=$4
    shift 4
    local label="${implementation}-${phase}"

    snapshot_gpu "${label}"
    date --iso-8601=seconds >"${run_dir}/${label}-started.txt"
    /usr/bin/time -p -o "${run_dir}/${label}.time" \
        ncu --force-overwrite \
        --target-processes all \
        --set full \
        --replay-mode application \
        --app-replay-match grid \
        --app-replay-mode strict \
        --kernel-name "regex:${kernel_regex}" \
        --launch-skip "${skip}" \
        --launch-count 1 \
        -o "${run_dir}/${label}" \
        "$@" >"${run_dir}/${label}.log" 2>&1
    ncu --import "${run_dir}/${label}.ncu-rep" --page details \
        >"${run_dir}/${label}-details.txt" 2>&1
    ncu --import "${run_dir}/${label}.ncu-rep" --csv --page raw \
        >"${run_dir}/${label}-raw.csv" 2>&1
    date --iso-8601=seconds >"${run_dir}/${label}-completed.txt"
}

# The chemistry kernel is launched exactly 1,000 times for the retained
# grid-64 workload. Sample the first, middle, and final launch. Each selected
# launch has the production grid size: 262,144 cells / 32 cells per CTA =
# 8,192 structured CTAs, or 262,144 cells / 128 threads = 2,048 CUDA CTAs.
for phase_skip in early:0 middle:499 late:999; do
    phase="${phase_skip%%:*}"
    skip="${phase_skip##*:}"
    profile_one \
        structured "${phase}" "${skip}" \
        'structured_trial_structured_che.*' \
        "${structured_bin}" --grid 64 \
        --driver-policy gridwide-cta32 --no-compare-final-state
    profile_one \
        cuda "${phase}" "${skip}" \
        '.*advance_collapse_gridwide_kernel.*' \
        "${cuda_bin}" --grid 64 --no-compare-final-state
done

snapshot_gpu after
date --iso-8601=seconds >"${run_dir}/completed.txt"
