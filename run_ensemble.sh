#!/bin/bash
MAX_JOBS=10
AVAILABLE_GPUS="0 1"
MAX_RETRIES=1

get_gpu_allocation() {
    local job_number=$1
    local gpus=($AVAILABLE_GPUS)
    local num_gpus=${#gpus[@]}
    local gpu_id=$((job_number % num_gpus))
    echo ${gpus[gpu_id]}
}

check_jobs() {
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

run_with_retry() {
    local script=$1
    local gpu_allocation=$2
    local attempt=0
    echo $gpu_allocation
    while [ $attempt -le $MAX_RETRIES ]; do
        # Run the Python script
        CUDA_VISIBLE_DEVICES=$gpu_allocation python $script
        status=$?
        if [ $status -eq 0 ]; then
            echo "Script $script succeeded."
            break
        else
            echo "Script $script failed on attempt $attempt. Retrying..."
            ((attempt++))
        fi
    done
    if [ $attempt -gt $MAX_RETRIES ]; then
        echo "Script $script failed after $MAX_RETRIES attempts."
    fi
}

for seed in 1000 2000 3000 4000 5000 6000 7000 8000 9000; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "ensemble.py \
    --seed ${seed}" \
    "$gpu_allocation" & 
done

wait