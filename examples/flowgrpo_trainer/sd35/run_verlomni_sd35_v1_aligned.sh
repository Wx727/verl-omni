#!/usr/bin/env bash
# Aligned SD3.5 + FlowGRPO benchmark for the diffusion V1 trainer.
#
# This keeps the original UniRL comparison workload while running current
# verl-omni main with TransferQueue, ReplayBuffer, and request-level rollout
# batching. Run from any directory inside an environment installed from this
# checkout:
#
#   SD35=<hf-id-or-local-dir> \
#   DATA=$HOME/data/pickscore_sd3 \
#   STEPS=4 \
#   bash benchmarks/speed_benchmarks/verl_omni/run_verlomni_sd35_v1_aligned.sh
#
# Summarize the opt-in JSONL output with:
#
#   python scripts/parse_v1_benchmark_timing.py --skip 1 "$BENCH_TIMING_JSONL"
set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT_DIR"

SD35=${SD35:-stabilityai/stable-diffusion-3.5-medium}
DATA=${DATA:-$HOME/data/pickscore_sd3}
STEPS=${STEPS:-4}
ATTN=${ATTN:-sdpa}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
REQUEST_BATCH_MAX_WAIT_MS=${REQUEST_BATCH_MAX_WAIT_MS:-10}

if [ ! -f "$DATA/train.parquet" ] || [ ! -f "$DATA/test.parquet" ]; then
    echo "Missing PickScore parquet data under $DATA" >&2
    exit 2
fi

if [ -z "${BENCH_TIMING_JSONL:-}" ]; then
    export BENCH_TIMING_JSONL=$SCRIPT_DIR/verlomni_v1_timing.jsonl
fi
if [ -z "${BENCH_TIMING_RUN_ID:-}" ]; then
    export BENCH_TIMING_RUN_ID=v1-aligned-$(date +%Y%m%d-%H%M%S)
fi

if [ "$ATTN" = "fa3" ]; then
    ACTOR_ATTN=_flash_3_varlen_hub
    ROLLOUT_ATTN=FLASH_ATTN_3_HUB
else
    ACTOR_ATTN=native
    ROLLOUT_ATTN=TORCH_SDPA
fi

custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

python3 -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files=$DATA/train.parquet \
    data.val_files=$DATA/test.parquet \
    data.train_batch_size=48 \
    data.val_max_samples=8 \
    data.max_prompt_length=512 \
    data.truncation=error \
    data.seed=42 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.model.path=$SD35 \
    actor_rollout_ref.model.custom_chat_template="\"$custom_chat_template\"" \
    'actor_rollout_ref.model.extra_tokenizers={clip: {path: tokenizer, max_length: 77}, t5: {path: tokenizer_3, max_length: 256}}' \
    actor_rollout_ref.model.attn_backend=$ACTOR_ATTN \
    actor_rollout_ref.model.enable_gradient_checkpointing=false \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=false \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT_ATTN \
    actor_rollout_ref.rollout.step_execution=false \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.pipeline.height=384 \
    actor_rollout_ref.rollout.pipeline.width=384 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.max_prompt_embed_length=333 \
    actor_rollout_ref.rollout.algo.noise_level=0.8 \
    actor_rollout_ref.rollout.algo.sde_type=cps \
    actor_rollout_ref.rollout.algo.sde_window_size=3 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=$MAX_NUM_SEQS \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.request_batch_max_wait_ms=$REQUEST_BATCH_MAX_WAIT_MS \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=28 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.rollout_correction.bypass_mode=true \
    reward.num_workers=1 \
    reward.reward_model.enable=false \
    reward.reward_model.enable_resource_pool=false \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/pickscore_reward.py \
    reward.custom_reward_function.name=compute_score_pickscore \
    trainer.logger='["console"]' \
    trainer.project_name=speed_benchmarks \
    trainer.experiment_name=sd35_flowgrpo_pickscore_v1_aligned \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=$STEPS \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync \
    "$@"
