#!/bin/bash

set -x
echo "Starting worker script at $(hostname)"
num_gpus=8
num_nodes="$((SLURM_JOB_NUM_NODES))"
WORKDIR="/home/ifajcik/data_scratch_new/lm-evaluation-harness"
cd $WORKDIR

#export VLLM_LOGGING_LEVEL=DEBUG
#export CUDA_LAUNCH_BLOCKING=1
#export NCCL_DEBUG=TRACE
#export VLLM_TRACE_FUNCTION=1

export NCCL_IB_GID_INDEX=3
#export TORCH_NCCL_USE_COMM_NONBLOCKING=1
#export NCCL_SOCKET_IFNAME=eth0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create temp directory for ray
hostname=$(hostname)
target_dir="${WORKDIR}/ray_${hostname}"
rm -rf /tmp/ray
rm -rf $target_dir
mkdir -p $target_dir
ln -s "$target_dir" /tmp/ray

source ~/.bashrc
micromamba activate harness

LOCAL_ADDR=$(hostname)
job_array_id=$SLURM_ARRAY_JOB_ID
task_id=$SLURM_ARRAY_TASK_ID
full_job_id="${job_array_id}_${task_id}"

# Directory for job-specific locks and signals
LOCKDIR="${WORKDIR}/locks_${full_job_id}"
mkdir -p $LOCKDIR

if [ "$SUMLOGP" = "no" ]; then
  NORMALIZE_LOG_PROBS="True"
else
  NORMALIZE_LOG_PROBS="False"
fi

HEAD_STARTED_FILE="${LOCKDIR}/ray_head_started_${full_job_id}.signal"
WORKER_CONNECTED_FILE="${LOCKDIR}/worker_connected_${LOCAL_ADDR}_${full_job_id}.signal"

TOTAL_GPUS=$((num_gpus * num_nodes))

# Truncate strategy argument based on truncate_strategy argument
TRUNCATE_STRATEGY_ARG=""
if [ "$TRUNCATE_STRATEGY" != "none" ]; then
  TRUNCATE_STRATEGY_ARG=",truncate_strategy=$TRUNCATE_STRATEGY"
fi

export NUMEXPR_MAX_THREADS=$(nproc --all)

if [ "$MASTER_ADDR" == "$LOCAL_ADDR" ]; then
  # Start the Ray head
  echo "Starting Ray head at $MASTER_ADDR:$MASTER_PORT"
  ray start --include-dashboard=True --head --port $MASTER_PORT

  # Signal that the Ray head has started
  touch $HEAD_STARTED_FILE

  echo "Executing in $(pwd)"

  # Wait for all worker nodes to connect
  echo "Waiting for all worker nodes to connect..."
  for ((i=1; i<num_nodes; i++)); do
    while [ ! -f "${LOCKDIR}/worker_connected_node${i}_${full_job_id}.signal" ]; do
      sleep 2  # Check every 2 seconds
    done
  done

  echo "All workers connected. Proceeding with task execution."
  for ((i=1; i<num_nodes; i++)); do
    rm -f "${LOCKDIR}/worker_connected_node${i}_${full_job_id}.signal"
  done

  # Print the job array task ID
  echo "Running job array task: $full_job_id"

  $PYTHON ./print_ray_nodes_once.py
  $PYTHON -m lm_eval --model vllm \
    --model_args pretrained=$MODEL_NAME,tensor_parallel_size=$TOTAL_GPUS,enforce_eager=True,distributed_executor_backend=ray,dtype=bfloat16,gpu_memory_utilization=0.8,max_length=2048,normalize_log_probs=$NORMALIZE_LOG_PROBS,trust_remote_code=True$TRUNCATE_STRATEGY_ARG \
    --tasks "$TASK" \
    --batch_size 1 \
    --output_path "$OUTPUT_PATH" \
    --log_samples \
    --verbosity DEBUG \
    --num_fewshot $NUM_FEWSHOT $CHAT_TEMPLATE_ARGS

  # Release the lock after completion
  echo "Completed job array task: $full_job_id. Running cancel command."
  rm -f $HEAD_STARTED_FILE
  scancel $full_job_id
else
  set +x
  echo "Waiting for Ray head to start"

  # Wait for the signal file from the master node
  while [ ! -f "$HEAD_STARTED_FILE" ]; do
    sleep 2  # Check every 2 seconds
  done

  echo "Connecting from $LOCAL_ADDR to Ray head at $MASTER_ADDR:$MASTER_PORT"
  ray start --address $MASTER_ADDR:$MASTER_PORT

  # Signal that this worker has connected
  touch "${LOCKDIR}/worker_connected_node${SLURM_NODEID}_${full_job_id}.signal"

  echo "Waiting for master node to finish"
  while [ ! -f "$OUTPUT_PATH" ]; do
    sleep 10  # Check every 10 seconds
  done

  echo "Master node finished, exiting"
fi
