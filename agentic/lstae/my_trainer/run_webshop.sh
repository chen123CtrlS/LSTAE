set -x
ENGINE=${1:-vllm}
ulimit -u 65536

SPO_ENABLE=${SPO_ENABLE:-True}
SPO_RHO_CLIP_LOWER=${SPO_RHO_CLIP_LOWER:-0.875}

export JAVA_HOME=/opt/conda/lib/jvm
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so


export VLLM_ATTENTION_BACKEND=XFORMERS

export CUDA_VISIBLE_DEVICES="2,3"

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=64
val_data_size=128
group_size=1
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"

HOME1=/ossfs/workspace/verl-agent-master
model_path=/mnt/augdata3/xhy/ckpts/models/qw2.5-1.5b-ins

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $((val_data_size * 2)) # evaluate 2 × val_data_size tasks during each iteration

python3 -m verl.trainer.main_ppo \
    +trainer.spo.enable=$SPO_ENABLE \
    +trainer.spo.history=True \
    +trainer.spo.offline_values=$SPO_OFFLINE_VALUES \
    +trainer.spo.rho.type=kl \
    +trainer.spo.rho.clip_lower=$SPO_RHO_CLIP_LOWER \
    +trainer.rollout_data_dir=/ossfs/workspace/verl-agent-master/logs/steps \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME1/data/verl-agent/text/train.parquet \
    data.val_files=$HOME1/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name='gigpo_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=300 \
    trainer.default_local_dir="/mnt/antllm/xhy/ckpts1/qwen-1.5b-alfworld-spo-12-23-webshop" \
    trainer.val_before_train=False 2>&1 | tee /ossfs/workspace/verl-agent-master/logs/spo_webshop.log