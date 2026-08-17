# Gemma4-E2B GRPO RL launchers for TPU v7x (Ironwood, Pathways)

Helm/Pathways workload launchers (pathways_tpu_jobset_workload chart), validated
2026-08-17 on bodaborg-tpu7x-nap (project cloud-tpu-shared-capacity).

| Script | Sampler route | Shape | Status |
|---|---|---|---|
| [`gemma4_e2b_smoke_maxtext_route.sh`](gemma4_e2b_smoke_maxtext_route.sh) | MaxTextForCausalLM (direct MaxText→MaxText weight sync) | smoke: 2 steps, batch 8, 2x2x1 | **VALIDATED** — "RL Training Completed Successfully!", exit 0, 2/2 steps, ckpts @ 1+2, 39/39 completions math-verified, ~44 min |
| [`gemma4_e2b_smoke_native_route.sh`](gemma4_e2b_smoke_native_route.sh) | native tpu_inference `Gemma4ForCausalLM` (uses the standalone gemma4 vLLM weight mapping added in this branch) | smoke: 2 steps, batch 8, 2x2x1 | pending validation — see below |
| [`gemma4_e2b_benchmark_4x4x4_maxtext_route.sh`](gemma4_e2b_benchmark_4x4x4_maxtext_route.sh) | MaxTextForCausalLM | benchmark: 30 steps, batch 256, 4x4x4 | ready; needs admin-provisioned multi-host pool (NAP only creates 2x2x1) |

## Two sampler routes

The trainer is always the MaxText model; the routes differ in what vLLM runs as
the rollout/sampler model and how weights are pushed to it each step.

- **MaxText route**: `vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}'`
  + `vllm_additional_config.maxtext_config` → sampler is the MaxText model under
  vLLM ([`integration/vllm/maxtext_vllm_adapter/adapter.py`](../../../../../integration/vllm/maxtext_vllm_adapter/adapter.py)).
  The presence of `maxtext_config` sets `use_no_op_mappings`
  ([`utils/model_creation_utils.py`](../../../../../utils/model_creation_utils.py),
  search `use_no_op_mappings`), which switches the tunix weight push to DIRECT
  structural sync — identity copy, no mapping file
  ([tunix `generate/vllm_sampler.py`](https://github.com/google/tunix/blob/main/tunix/generate/vllm_sampler.py),
  `update_params` else-branch "Direct Weight Sync"). Required for gemma4 because
  the HF config registers as multimodal and tpu_inference's runner rejects
  GRPO's prompt logprobs for multimodal models
  ([tpu_inference `runner/tpu_runner.py`](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/runner/tpu_runner.py),
  search "prompt_logprobs is not supported"). Mirrors the team recipe
  [`run_gemma4_e4b_rl.sh`](../run_gemma4_e4b_rl.sh).
- **Native route**: `vllm_hf_overrides='{architectures: ["Gemma4ForCausalLM"]}'`
  and NO `vllm_additional_config` → sampler is tpu_inference's serving-optimized
  JAX model ([`models/jax/gemma4.py`](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/models/jax/gemma4.py),
  registered in [`models/common/model_loader.py`](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/models/common/model_loader.py)).
  The weight push goes through the standalone mapping
  [`integration/tunix/weight_mapping/gemma4.py`](../../../../../integration/tunix/weight_mapping/gemma4.py)
  (added in this branch; dispatched from
  [`weight_mapping/__init__.py`](../../../../../integration/tunix/weight_mapping/__init__.py)
  via [`integration/tunix/tunix_adapter.py`](../../../../../integration/tunix/tunix_adapter.py))
  with an embedding un-scale hook — the MaxText checkpoint stores
  embedding × sqrt(hidden) (see `pad_and_scale_embedding` in
  [`checkpoint_conversion/utils/param_mapping.py`](../../../../../checkpoint_conversion/utils/param_mapping.py))
  while the native model scales at runtime.

## Native route — progress and blocker (2026-08-17)

Progress:
1. Weight mapping implemented and offline-validated: 22 entries cover all 23
   checkpoint param groups (23rd is the `step` counter); every destination name
   verified against the HF ground-truth shape table
   ([`checkpoint_conversion/utils/hf_shape.py`](../../../../../checkpoint_conversion/utils/hf_shape.py));
   sharding ranks verified against real checkpoint array shapes;
   ÷sqrt(hidden) embedding hook verified numerically. Registry dispatch
   (`gemma4*` → factory in
   [`weight_mapping/__init__.py`](../../../../../integration/tunix/weight_mapping/__init__.py))
   proven in-run: the original "gemma4-e2b vLLM weight mapping not found" error
   is gone.
2. Without the architecture override, vLLM auto-selects
   `Gemma4ForConditionalGeneration` (multimodal,
   [`models/jax/gemma4_mm.py`](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/models/jax/gemma4_mm.py))
   from the HF config and dies at "prompt_logprobs is not supported for
   multimodal models" — solved by overriding to the text-only
   `Gemma4ForCausalLM` class.
3. First full e2e attempt launched 2026-08-17 (2 steps, 2x2x1).

Current blocker (infra, not code): the attempt is stuck Pending — the cpu-np
nodepool is at capacity and cluster-autoscaler scale-up fails with "GCE out of
resources" in us-central1-c (FailedScaleUp; max node group / backoff). Retry
when CPU-VM capacity frees up.

Remaining validation gate once it runs: coherent generations and
sampler-trainer pearson > 0.8 — that is the end-to-end proof of the weight
mapping. Residual risk is confined to the weight push (name/ending conventions
and sharding tuples); everything else in the pipeline is identical to the
validated MaxText-route run.

## Other load-bearing gemma4 settings (both routes)

- Non-agentic GRPO (no `rl.use_agentic_rollout`): the agentic env mis-renders
  gemma chat prompts (stringifies raw message dicts into the prompt) — every
  rollout hits MAX_CONTEXT_LIMIT_REACHED. Prompt construction in the
  non-agentic path: `process_data` in
  [`trainers/post_train/rl/utils_rl.py`](../../utils_rl.py).
- [`data_template_path`](../../../../../examples/chat_templates/openmathinstruct2_rl.json)
  + [`chat_template_path`](../../../../../examples/chat_templates/gemma-3-27b-chat_template.json)
  (gemma-3-27b): gemma4-e2b-it needs proper chat turn structure to terminate
  generations.
- `scan_layers=False`: gemma4 per-layer KV sharing is incompatible with nn.scan
  (see `GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_MAPPING` in
  [`checkpoint_conversion/utils/param_mapping.py`](../../../../../checkpoint_conversion/utils/param_mapping.py)).
- `train_micro_batch_size=1`: 262k-vocab logits are not batch-sharded on the
  trainer; tmbs=4 at seq 8192 needs ~130G HLO temporaries vs ~95G usable
  HBM per Ironwood core-device (matches the GB200 "pd frozen at 2" constraint).
