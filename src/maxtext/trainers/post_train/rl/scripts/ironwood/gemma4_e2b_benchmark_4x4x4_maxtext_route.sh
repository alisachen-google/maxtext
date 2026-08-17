#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# --- Graceful Shutdown Logic ---
_sigterm() {
  echo "Caught SIGTERM signal, sending to child process..."
  kill -SIGTERM "$PID" 2>/dev/null
}
trap _sigterm SIGTERM

MAXTEXT_CONFIG_PATH="maxtext/configs/post_train/rl.yml"
# Ironwood bring-up fixes (from smoke ladder):
export JAX_PLATFORMS="proxy,cpu"            # newer tpu_inference stages on cpu backend
export VLLM_ENABLE_V1_MULTIPROCESSING=0     # required under Pathways
XLA_FLAGS=""

# Fallback check to support different image structures
if [ ! -f "${MAXTEXT_CONFIG_PATH}" ] && [ -f "src/${MAXTEXT_CONFIG_PATH}" ]; then
  echo "MAXTEXT_CONFIG_PATH not found at ${MAXTEXT_CONFIG_PATH}, falling back to src/${MAXTEXT_CONFIG_PATH}"
  MAXTEXT_CONFIG_PATH="src/${MAXTEXT_CONFIG_PATH}"
fi

# --- TPU Worker Hostnames & ID ---
if [ ! -z "${NNODES}" ] && [ ! -z "${JOBSET_NAME}" ] && [ ! -z "${SUBDOMAIN}" ]; then
  HOSTS=""
  for i in $(seq 0 $((NNODES - 1))); do
    # JobSet pod naming: ${JOBSET_NAME}-workload-0-${i}.${SUBDOMAIN}
    HOSTS="${HOSTS}${JOBSET_NAME}-workload-0-${i}.${SUBDOMAIN},"
  done
  export TPU_WORKER_HOSTNAMES=${HOSTS%,}
  echo "Computed TPU_WORKER_HOSTNAMES: ${TPU_WORKER_HOSTNAMES}"
fi

# --- Log for Debugging ---
echo "MaxText config overrides from command line:"
echo "  model_name='gemma4-e2b' data_template_path=maxtext/examples/chat_templates/openmathinstruct2_rl.json chat_template_path=maxtext/examples/chat_templates/gemma-3-27b-chat_template.json tokenizer_path='google/gemma-4-E2B-it' dataset_name='nvidia/OpenMathInstruct-2' hf_train_files='hf://datasets/nvidia/OpenMathInstruct-2/data/train_1M-*.parquet' train_split='train_1M' rl.num_generations=8 rl.grpo_beta=0.0 rl.epsilon_high=0.28 rl.grpo_epsilon=0.2 enable_checkpointing=True checkpoint_storage_use_ocdbt=False checkpoint_storage_use_zarr3=False async_scheduling=True chips_per_vm=4 num_batches=30 num_test_batches=0 gradient_clipping_threshold=1.0 decode_sampling_temperature=0.8 decode_sampling_top_k=50 decode_sampling_nucleus_p=0.95 max_target_length=8192 max_prefill_predict_length=4096 learning_rate=1e-06 batch_size=256 train_micro_batch_size=1 checkpoint_period=10 rollout_micro_batch_size=256 rollout_data_parallelism=-1 rollout_tensor_parallelism=1 enable_dp_attention=False hbm_utilization_vllm=0.6 max_num_seqs=64 max_num_batched_tokens=8448 scan_layers=False vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' vllm_additional_config='{"maxtext_config": {"model_name": "gemma4-e2b", "model_call_mode": "inference", "log_config": false, "weight_dtype": "bfloat16", "allow_split_physical_axes": true}}' allow_split_physical_axes=True enable_tunix_perf_metrics=True load_parameters_path='gs://ubench-logs/alisachen-tpu7x/ckpt/gemma4-e2b-raw/0/items' skip_jax_distributed_system=True log_period=1 base_output_directory=${ARTIFACT_DIR}"

# --- RL Sampler Patches & Env Setup ---
if [ "maxtext.trainers.post_train.rl.train_rl" = "maxtext.trainers.post_train.rl.train_rl" ]; then
  export HF_HUB_ENABLE_HF_TRANSFER=0
  export HF_HUB_DISABLE_XET=1
  export PYTHONPATH=/app/src:$PYTHONPATH
  export NUM_PRECOMPILE_WORKERS=1
  export NEW_MODEL_DESIGN=1
  export JAX_BACKEND_TARGET=grpc://127.0.0.1:29000
fi

# --- Main Execution ---
# The python process is run in the background (&)
(
  # Set up the environment
  export ENABLE_PATHWAYS_PERSISTENCE='1'
  export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu,cpu}"
  export ENABLE_PJRT_COMPATIBILITY='true'
  if [ ! -z "${XLA_FLAGS}" ]; then
    export LIBTPU_INIT_ARGS="${XLA_FLAGS}"
  fi

  # Execute the training script, passing the config file AND the command-line overrides
  python3 -m maxtext.trainers.post_train.rl.train_rl "${MAXTEXT_CONFIG_PATH}" model_name='gemma4-e2b' data_template_path=maxtext/examples/chat_templates/openmathinstruct2_rl.json chat_template_path=maxtext/examples/chat_templates/gemma-3-27b-chat_template.json tokenizer_path='google/gemma-4-E2B-it' dataset_name='nvidia/OpenMathInstruct-2' hf_train_files='hf://datasets/nvidia/OpenMathInstruct-2/data/train_1M-*.parquet' train_split='train_1M' rl.num_generations=8 rl.grpo_beta=0.0 rl.epsilon_high=0.28 rl.grpo_epsilon=0.2 enable_checkpointing=True checkpoint_storage_use_ocdbt=False checkpoint_storage_use_zarr3=False async_scheduling=True chips_per_vm=4 num_batches=30 num_test_batches=0 gradient_clipping_threshold=1.0 decode_sampling_temperature=0.8 decode_sampling_top_k=50 decode_sampling_nucleus_p=0.95 max_target_length=8192 max_prefill_predict_length=4096 learning_rate=1e-06 batch_size=256 train_micro_batch_size=1 checkpoint_period=10 rollout_micro_batch_size=256 rollout_data_parallelism=-1 rollout_tensor_parallelism=1 enable_dp_attention=False hbm_utilization_vllm=0.6 max_num_seqs=64 max_num_batched_tokens=8448 scan_layers=False vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' vllm_additional_config='{"maxtext_config": {"model_name": "gemma4-e2b", "model_call_mode": "inference", "log_config": false, "weight_dtype": "bfloat16", "allow_split_physical_axes": true}}' allow_split_physical_axes=True enable_tunix_perf_metrics=True load_parameters_path='gs://ubench-logs/alisachen-tpu7x/ckpt/gemma4-e2b-raw/0/items' skip_jax_distributed_system=True log_period=1 base_output_directory=${ARTIFACT_DIR} run_name=${JOB_IDENTIFIER}
) & PID=$!

# Wait for the python process to finish
EXIT_CODE=0
wait $PID || EXIT_CODE=$?   # tolerate failure so the debug-hold below can run (set -e)

echo "MaxText Job Ended at: $(date) with exit code $EXIT_CODE"
if [ "$EXIT_CODE" -ne 0 ]; then
  echo "DEBUG-HOLD: failure detected, keeping pod alive 20 min for log capture"
  sleep 1200
fi
exit $EXIT_CODE