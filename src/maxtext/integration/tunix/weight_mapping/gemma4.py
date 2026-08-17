# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Weight mapping from MaxText's Gemma4 models to tpu_inference's native vLLM models.

Gemma4 differs from the other standalone mappings (e.g. qwen3.py) in three ways:

1. Gemma4 MUST run unscanned (per-layer KV sharing is incompatible with
   nn.scan), so the trainer's source state has fused per-layer keys
   (`base.decoder.layers_3.mlp.wo.kernel`). The tunix transfer machinery only
   wildcard-expands scanned sources, so this mapping enumerates every layer
   explicitly instead of using `layers.*` templates.

2. Destinations are the LIVE nnx attribute paths of tpu_inference's
   `Gemma4ForCausalLM` (`language_model.layers.3.mlp.down_proj.weight`) — not
   the HF checkpoint names (`model.language_model...`); tunix matches mapping
   destinations against `nnx.state(vllm_model)`, and Gemma4ForCausalLM roots
   its tree at the `language_model` attribute. All tpu_inference layer classes
   alias their parameter to `.weight` (HF-style leaf).

3. tpu_inference's Gemma4MLP fuses gate+up into one `gate_up_proj` parameter,
   while MaxText keeps separate `wi_0` / `wi_1`. The transfer machinery cannot
   concatenate two sources into one target, so this module supplies a
   `preprocess_src_state_fn` that fuses `wi_0`+`wi_1` into a synthetic
   `mlp.gate_up.kernel` source entry before the transfer runs.

The per-layer structure (which layers have k/v — KV-shared layers have none —
and the E2B per-layer embedder) is derived from the checkpoint-conversion
source of truth, `PARAM_MAPPING[model_name](..., scan_layers=False)`.

Hook: the MaxText checkpoint stores embedding × sqrt(hidden) (MaxText
compensates on the output side via normalize_embedding_logits), while
tpu_inference keeps the raw table and applies embedding_scale at runtime — so
the push divides the embedding by sqrt(hidden).
"""

import dataclasses
import re

import numpy as np


def _maxtext_cfg_shim(model_name):
  """Minimal stand-in for the MaxText config fields PARAM_MAPPING reads."""
  # The Gemma4 PARAM_MAPPING consults exactly two maxtext_config attributes:
  # v_norm_with_scale (per configs/models/gemma4-*.yml) and use_multimodal
  # (False here — the RL weight-push path is text-only).
  values = {
      "gemma4-e2b": {"v_norm_with_scale": False, "use_multimodal": False},
      "gemma4-e4b": {"v_norm_with_scale": False, "use_multimodal": False},
      "gemma4-26b": {"v_norm_with_scale": True, "use_multimodal": False},
      "gemma4-31b": {"v_norm_with_scale": True, "use_multimodal": False},
  }[model_name]
  return type("MaxTextCfgShim", (), values)()


def _hf_config_dict(model_name):
  from maxtext.checkpoint_conversion.utils import hf_model_configs as hmc

  return {
      "gemma4-e2b": hmc.gemma4_e2b_dict,
      "gemma4-e4b": hmc.gemma4_e4b_dict,
      "gemma4-26b": hmc.gemma4_26b_dict,
      "gemma4-31b": hmc.gemma4_31b_dict,
  }[model_name]


def _src_key(raw_key):
  """'params-decoder-layers_0-mlp-wo-kernel' -> 'base.decoder.layers_0.mlp.wo.kernel'."""
  key = raw_key.replace("-", ".")
  return re.sub(r"^params\.", "base.", key)


def _dst_path(hf_value):
  """HF checkpoint name -> live Gemma4ForCausalLM nnx path (strip 'model.')."""
  first = hf_value[0] if isinstance(hf_value, (list, tuple)) else hf_value
  return re.sub(r"^model\.", "", first)


def _sharding_for(src):
  """Sharding axes matching the unscanned MaxText source array ranks."""
  if src.endswith("token_embedder.embedding"):
    return ("model", None)  # (vocab, emb)
  if src.endswith("per_layer_embedder.embed_tokens_per_layer"):
    return ("model", None)  # (vocab, layers*per_layer_emb)
  if ".self_attention.out.kernel" in src:
    return ("model", None, None)  # (heads, head_dim, emb)
  if ".self_attention." in src and src.endswith(".kernel"):
    return (None, "model", None)  # q/k/v: (emb, heads, head_dim)
  if src.endswith(".mlp.gate_up.kernel"):
    return (None, "model")  # fused (emb, 2*mlp)
  if src.endswith(".mlp.wo.kernel"):
    return ("model", None)  # (mlp, emb)
  if src.endswith("layer_scalar"):
    return (None,)
  if src.endswith(".kernel"):
    return (None, None)  # small per-layer projections — replicate
  return (None,)  # norms / scales


def _build_mapping(model_name):
  from maxtext.checkpoint_conversion.utils.param_mapping import PARAM_MAPPING

  raw = PARAM_MAPPING[model_name](
      _hf_config_dict(model_name),
      maxtext_config=_maxtext_cfg_shim(model_name),
      scan_layers=False,
  )
  mapping = {}
  mlp_layers = set()
  attn_layers = set()
  for raw_key, hf_value in raw.items():
    src = _src_key(raw_key)
    m = re.search(r"layers_(\d+)\.mlp\.wi_[01]\.kernel$", src)
    if m:
      # wi_0/wi_1 are pushed fused (see preprocess_src_state_fn); the synthetic
      # source key is emitted below instead of the raw ones.
      mlp_layers.add(int(m.group(1)))
      continue
    m = re.search(r"layers_(\d+)\.self_attention\.(query|key|value)\.kernel$", src)
    if m:
      # tpu_inference's Gemma4 uses a fused qkv_proj on every layer
      # (attention_k_eq_v=False); q/k/v are pushed fused too. KV-shared layers
      # have no MaxText k/v — the preprocessor zero-fills those sections (the
      # layer reads its KV cache from the share target, so they are never
      # consumed).
      attn_layers.add(int(m.group(1)))
      continue
    mapping[src] = (_dst_path(hf_value), _sharding_for(src))
  for i in sorted(mlp_layers):
    src = f"base.decoder.layers_{i}.mlp.gate_up.kernel"
    mapping[src] = (
        f"language_model.layers.{i}.mlp.gate_up_proj.weight",
        _sharding_for(src),
    )
  for i in sorted(attn_layers):
    src = f"base.decoder.layers_{i}.self_attention.qkv.kernel"
    mapping[src] = (
        f"language_model.layers.{i}.self_attn.qkv_proj.weight",
        (None, "model"),
    )
  return mapping


def _fuse_for_native(state):
  """Fuses mlp wi_0/wi_1 -> gate_up and attention q/k/v -> qkv for the native model.

  The fused qkv kernel is 2D (hidden, q_size + k_size + v_size) in plain
  [Q|K|V] feature order — JaxQKVParallelLinear's layout at tensor-parallel
  size 1 (our rollout_tensor_parallelism=1; the merged layer interleaves
  shards at TP>1, which this fusion does NOT model). KV-shared layers have no
  MaxText k/v weights; their sections are zero-filled (computed but never
  consumed — the layer reads the share target's KV cache).
  """
  import jax.numpy as jnp
  from flax import nnx

  flat = dict(state.flat_state())
  # kv head count is uniform across layers that have k/v (gemma4: 1).
  kv_heads = 1
  for path, var in flat.items():
    if len(path) >= 2 and str(path[-2]) == "key" and str(path[-1]) == "kernel":
      kv_heads = var.value.shape[-2]
      break

  out = []
  for path, var in flat.items():
    parts = tuple(str(p) for p in path)
    if len(parts) >= 3 and parts[-3] == "mlp" and parts[-2] in ("wi_0", "wi_1"):
      if parts[-2] == "wi_0":
        wi_1 = flat[(*path[:-2], "wi_1", path[-1])]
        fused = jnp.concatenate([var.value, wi_1.value], axis=-1)
        out.append(((*path[:-2], "gate_up", path[-1]), var.replace(value=fused)))
      continue  # drop wi_0/wi_1 from the pushed state
    if len(parts) >= 3 and parts[-3] == "self_attention" and parts[-2] in ("query", "key", "value"):
      if parts[-2] == "query":
        q = var.value  # (hidden, heads, head_dim)
        hidden, _, head_dim = q.shape
        q2 = q.reshape(hidden, -1)
        kv_shape = (hidden, kv_heads * head_dim)
        k_path = (*path[:-2], "key", path[-1])
        v_path = (*path[:-2], "value", path[-1])
        k2 = flat[k_path].value.reshape(hidden, -1) if k_path in flat else jnp.zeros(kv_shape, q.dtype)
        v2 = flat[v_path].value.reshape(hidden, -1) if v_path in flat else jnp.zeros(kv_shape, q.dtype)
        fused = jnp.concatenate([q2, k2, v2], axis=-1)
        out.append(((*path[:-2], "qkv", path[-1]), var.replace(value=fused)))
      continue  # drop query/key/value from the pushed state
    out.append((path, var))
  return nnx.State.from_flat_path(out)


def build_gemma4_vllm_mapping(model_name):
  """Factory: mapping object for one Gemma4 variant (registry entry point)."""

  @dataclasses.dataclass
  class GEMMA4_VLLM_MAPPING:  # pylint: disable=invalid-name
    """Mapping MaxText Gemma4 weights to tpu_inference's native Gemma4 weights."""

    @staticmethod
    def to_hf_mapping():
      return _build_mapping(model_name)

    @staticmethod
    def to_hf_hook_fns():
      # MaxText stores embedding × sqrt(hidden); the native model keeps the raw
      # table and scales at runtime (Gemma4Model.embedding_scale) — divide on
      # push, as gemma3 does.
      def scale_embedding(arr):
        hidden_size = arr.shape[1]
        normalizer = np.dtype(arr.dtype).type(hidden_size**0.5)
        return arr / normalizer

      return {"base.token_embedder.embedding": scale_embedding}

    @staticmethod
    def to_hf_transpose_keys():
      return {}

    @staticmethod
    def lora_to_hf_mappings():
      return None

    @staticmethod
    def preprocess_src_state_fn():
      return _fuse_for_native

  return GEMMA4_VLLM_MAPPING
