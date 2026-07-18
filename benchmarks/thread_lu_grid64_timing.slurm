#!/usr/bin/env bash
#SBATCH --job-name=chem-thread-lu
#SBATCH --account=general
#SBATCH --partition=general-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --array=0-2
#SBATCH --output=benchmark_runs/thread-lu-grid64-%A_%a.out
#SBATCH --error=benchmark_runs/thread-lu-grid64-%A_%a.err

set -euo pipefail

repo=/mnt/ffs24/home/wibkingb/mojo-chemistry
cooperative_bin="${repo}/benchmark_runs/fused-12752255/bin/reproducer_structured"
per_thread_bin="${repo}/benchmark_runs/thread-lu-grid64-submission/bin/reproducer_structured_thread_lu"
run_dir="${repo}/benchmark_runs/thread-lu-grid64-${SLURM_ARRAY_JOB_ID}/task-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${run_dir}"
cd "${repo}"

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
    echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "SELECTED_GPU=${CUDA_VISIBLE_DEVICES:-0} (the single Slurm-allocated H200)"
    echo "AVAILABLE_CUDA_MODULES_BEGIN"
    printf '%s\n' "${available_cuda}"
    echo "AVAILABLE_CUDA_MODULES_END"
    echo "LOADED_CUDA_MODULE=${cuda_module}"
    module -t list
    nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap --format=csv,noheader
    sha256sum "${cooperative_bin}" "${per_thread_bin}"
    git -C "${repo}" rev-parse HEAD
    sha256sum \
        "${repo}/src/gpu/structured_trial.mojo" \
        "${repo}/src/gpu/structured_ops.mojo" \
        "${repo}/reproducer.mojo" \
        "${repo}/apps/reproducer_structured_gpu.mojo"
} >"${run_dir}/environment.txt" 2>&1

snapshot_gpu() {
    local label=$1
    nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,utilization.gpu,memory.used,memory.total --format=csv,noheader \
        >"${run_dir}/gpu-${label}.csv"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
        >"${run_dir}/gpu-processes-${label}.csv"
}

run_one() {
    local phase=$1
    local implementation=$2
    local binary=$3
    local label="${phase}-${implementation}"
    date --iso-8601=seconds >"${run_dir}/${label}-started.txt"
    /usr/bin/time -p -o "${run_dir}/${label}.time" \
        "${binary}" --grid 64 --driver-policy gridwide-cta32 \
        --no-compare-final-state >"${run_dir}/${label}.log" 2>&1
    grep -q '^completed global collapse steps: 1000$' "${run_dir}/${label}.log"
    date --iso-8601=seconds >"${run_dir}/${label}-completed.txt"
}

if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then
    first_name=cooperative
    first_bin="${cooperative_bin}"
    second_name=per-thread
    second_bin="${per_thread_bin}"
else
    first_name=per-thread
    first_bin="${per_thread_bin}"
    second_name=cooperative
    second_bin="${cooperative_bin}"
fi

snapshot_gpu before-warmup
run_one warmup "${first_name}" "${first_bin}"
run_one warmup "${second_name}" "${second_bin}"

snapshot_gpu before-measurement
run_one measured "${first_name}" "${first_bin}"
run_one measured "${second_name}" "${second_bin}"
snapshot_gpu after-measurement

printf 'implementation\tapp_wall_s\treal_s\n' >"${run_dir}/summary.tsv"
for implementation in cooperative per-thread; do
    app_wall="$(awk '/^wall time:/ {print $3}' "${run_dir}/measured-${implementation}.log")"
    real_wall="$(awk '/^real / {print $2}' "${run_dir}/measured-${implementation}.time")"
    printf '%s\t%s\t%s\n' "${implementation}" "${app_wall}" "${real_wall}" \
        >>"${run_dir}/summary.tsv"
done
date --iso-8601=seconds >"${run_dir}/completed.txt"
sed -n '1,3p' "${run_dir}/summary.tsv"
