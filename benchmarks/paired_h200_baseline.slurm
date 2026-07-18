#!/usr/bin/env bash
#SBATCH --job-name=chem-paired-base
#SBATCH --account=general
#SBATCH --partition=general-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=benchmark_runs/slurm-%j.out
#SBATCH --error=benchmark_runs/slurm-%j.err

set -euo pipefail

repo=/mnt/ffs24/home/wibkingb/mojo-chemistry
variant="${BENCHMARK_VARIANT:-baseline}"
structured_policy="${STRUCTURED_POLICY:-gridwide-cta32}"
skip_grid64_comparison="${SKIP_GRID64_COMPARISON:-0}"
run_dir="${repo}/benchmark_runs/${variant}-${SLURM_JOB_ID}"
bin_dir="${PREBUILT_BIN_DIR:-${run_dir}/bin}"
mkdir -p "${run_dir}"
cd "${repo}"

module purge
module load CUDA/12.9.1

{
    date --iso-8601=seconds
    hostname
    echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
    echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "BENCHMARK_VARIANT=${variant}"
    echo "PREBUILT_BIN_DIR=${PREBUILT_BIN_DIR:-unset}"
    echo "BASELINE_REFERENCE=${BASELINE_REFERENCE:-unset}"
    echo "STRUCTURED_POLICY=${structured_policy}"
    echo "SKIP_GRID64_COMPARISON=${skip_grid64_comparison}"
    module -t list
    pixi run mojo --version
    nvcc --version
    nsys --version
    ncu --version
    git rev-parse HEAD
    git status --short
} >"${run_dir}/environment.txt" 2>&1

nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,utilization.gpu,memory.used,memory.total --format=csv,noheader \
    >"${run_dir}/gpu-before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
    >"${run_dir}/gpu-processes-before.csv"

if [[ -z "${PREBUILT_BIN_DIR:-}" ]]; then
    mkdir -p "${bin_dir}"
    pixi run mojo build -I . reproducer_structured_gpu.mojo \
        -o "${bin_dir}/reproducer_structured" \
        >"${run_dir}/build-structured.log" 2>&1

    nvcc -x cu -std=c++20 -O3 -arch=sm_90 --expt-relaxed-constexpr \
        --fmad=false --maxrregcount=255 -Xptxas=-v \
        -DPRIMORDIAL_ROS2S_ENABLE_CUDA \
        -DPRIMORDIAL_ROS2S_CUDA_THREADS_PER_BLOCK=128 -I. \
        reproducer.cpp -o "${bin_dir}/reproducer_cuda" \
        >"${run_dir}/build-cuda.log" 2>&1
fi

export MODULAR_NVPTX_COMPILER_PATH="${repo}/benchmarks/ptxas_verbose.sh"

pixi run mojo build reproducer.mojo -o "${run_dir}/reproducer_cpu" \
    -Xlinker -lm >"${run_dir}/build-cpu.log" 2>&1

run_timed() {
    local label=$1
    shift
    /usr/bin/time -p -o "${run_dir}/${label}.time" \
        "$@" >"${run_dir}/${label}.log" 2>&1
}

# Grid-1 numerical comparison uses the controller-compatible CPU oracle.
run_timed grid1-cpu \
    "${run_dir}/reproducer_cpu" --grid 1 \
    --driver-policy gridwide-cta32 --no-compare-final-state \
    --write-final-state "${run_dir}/grid1-cpu.bin"
run_timed grid1-structured \
    "${bin_dir}/reproducer_structured" --grid 1 \
    --driver-policy "${structured_policy}" \
    --compare-final-state "${run_dir}/grid1-cpu.bin"
run_timed grid1-cuda \
    "${bin_dir}/reproducer_cuda" --grid 1 --no-compare-final-state

# Warm both implementations with the target problem before timing.
run_timed warm-structured \
    "${bin_dir}/reproducer_structured" --grid 64 \
    --driver-policy "${structured_policy}" --no-compare-final-state
run_timed warm-cuda \
    "${bin_dir}/reproducer_cuda" --grid 64 --no-compare-final-state

# Three warmed, interleaved target runs. The retained baseline publishes the
# grid-64 reference; an optimized variant compares every run against it.
reference="${BASELINE_REFERENCE:-${run_dir}/grid64-structured.bin}"
comparison_args=(--compare-final-state "${reference}")
if [[ "${skip_grid64_comparison}" == "1" ]]; then
    comparison_args=(--no-compare-final-state)
fi
if [[ -n "${BASELINE_REFERENCE:-}" && "${skip_grid64_comparison}" != "1" ]]; then
    run_timed run1-structured \
        "${bin_dir}/reproducer_structured" --grid 64 \
        --driver-policy "${structured_policy}" \
        --compare-final-state "${reference}" \
        --write-final-state "${run_dir}/grid64-structured.bin"
else
    run_timed run1-structured \
        "${bin_dir}/reproducer_structured" --grid 64 \
        --driver-policy "${structured_policy}" --no-compare-final-state \
        --write-final-state "${run_dir}/grid64-structured.bin"
fi
run_timed run1-cuda \
    "${bin_dir}/reproducer_cuda" --grid 64 --no-compare-final-state
run_timed run2-structured \
    "${bin_dir}/reproducer_structured" --grid 64 \
    --driver-policy "${structured_policy}" \
    "${comparison_args[@]}"
run_timed run2-cuda \
    "${bin_dir}/reproducer_cuda" --grid 64 --no-compare-final-state
run_timed run3-structured \
    "${bin_dir}/reproducer_structured" --grid 64 \
    --driver-policy "${structured_policy}" \
    "${comparison_args[@]}"
run_timed run3-cuda \
    "${bin_dir}/reproducer_cuda" --grid 64 --no-compare-final-state

nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,utilization.gpu,memory.used,memory.total --format=csv,noheader \
    >"${run_dir}/gpu-before-nsys.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
    >"${run_dir}/gpu-processes-before-nsys.csv"

nsys profile --force-overwrite=true --sample=none --cpuctxsw=none \
    --trace=cuda,osrt -o "${run_dir}/nsys-structured" \
    "${bin_dir}/reproducer_structured" --grid 64 \
    --driver-policy "${structured_policy}" --no-compare-final-state \
    >"${run_dir}/nsys-structured.log" 2>&1
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --timeunit msec \
    "${run_dir}/nsys-structured.nsys-rep" \
    >"${run_dir}/nsys-structured-stats.txt" 2>&1

nsys profile --force-overwrite=true --sample=none --cpuctxsw=none \
    --trace=cuda,osrt -o "${run_dir}/nsys-cuda" \
    "${bin_dir}/reproducer_cuda" --grid 64 --no-compare-final-state \
    >"${run_dir}/nsys-cuda.log" 2>&1
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --timeunit msec \
    "${run_dir}/nsys-cuda.nsys-rep" \
    >"${run_dir}/nsys-cuda-stats.txt" 2>&1

nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,utilization.gpu,memory.used,memory.total --format=csv,noheader \
    >"${run_dir}/gpu-before-ncu.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
    >"${run_dir}/gpu-processes-before-ncu.csv"

# Profile one fully populated chemistry CTA. Restricting the kernel and launch
# count keeps replay overhead bounded while collecting occupancy and stall data.
ncu --force-overwrite --target-processes all --set full \
    --kernel-name 'regex:structured_trial_structured_che.*' \
    --launch-count 1 -o "${run_dir}/ncu-structured" \
    "${bin_dir}/reproducer_structured" --grid 4 \
    --driver-policy "${structured_policy}" --no-compare-final-state \
    >"${run_dir}/ncu-structured.log" 2>&1
ncu --import "${run_dir}/ncu-structured.ncu-rep" --page details \
    >"${run_dir}/ncu-structured-details.txt" 2>&1

if [[ -n "${PROFILE_BASELINE_BIN:-}" ]]; then
    ncu --force-overwrite --target-processes all --set full \
        --kernel-name 'regex:structured_trial_structured_che.*' \
        --launch-count 1 -o "${run_dir}/ncu-baseline" \
        "${PROFILE_BASELINE_BIN}" --grid 4 \
        --driver-policy gridwide-cta32 --no-compare-final-state \
        >"${run_dir}/ncu-baseline.log" 2>&1
    ncu --import "${run_dir}/ncu-baseline.ncu-rep" --page details \
        >"${run_dir}/ncu-baseline-details.txt" 2>&1
fi

nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader \
    >"${run_dir}/gpu-after.csv"
date --iso-8601=seconds >"${run_dir}/completed.txt"
