.PHONY: setup smoke debug sweep train6 eval6 render6 uniform6 lqr6 swingup-debug swingup6 swingup6-gradient-low-momentum swingup6-uniform-low-momentum eval-swingup6 eval-swingup6-low-momentum render-swingup6 probe-swingup-trajectory search-swingup-trajectory search-swingup-capture search-swingup-chain export-swingup-states export-policy-handoff-states export-chain-states eval-expert-chain eval-expert-chain-learned capture-handoff6 eval-capture-handoff6 capture-state-list6 eval-capture-state-list6 capture-state-list-shaped6 eval-capture-state-list-shaped6 capture-policy-handoff-shaped6 capture-policy-handoff-stage eval-capture-policy-handoff-shaped6 chain-state-list-shaped6 eval-chain-state-list-shaped6

PYTHON ?= .venv/bin/python
SWING_HANDOFF_RUN ?= runs/swingup6_gradient_low_momentum
SWING_HANDOFF_CONFIG ?= $(SWING_HANDOFF_RUN)/config.resolved.yaml
SWING_HANDOFF_CHECKPOINT ?= $(SWING_HANDOFF_RUN)/checkpoints/frontier.safetensors
SWING_HANDOFF_OUT ?= runs/swingup6_policy_handoff/swing_handoff_states.json
SWING_HANDOFF_EPISODES ?= 128
SWING_HANDOFF_MAX_STATES ?= 512
CAPTURE_STAGE_PROGRESS ?= 1.0
CAPTURE_STAGE_OUT ?= runs/swingup6_policy_handoff_capture_stage
CAPTURE_STAGE_UPDATES ?= 5000
CAPTURE_STAGE_ENVS ?= 48
CAPTURE_STAGE_ROLLOUT_STEPS ?= 256
CAPTURE_STAGE_EVAL_EVERY ?= 50
CAPTURE_STAGE_EVAL_EPISODES ?= 8
CAPTURE_STAGE_ENTROPY ?= 0.006
CAPTURE_STAGE_RAIL_LIMIT ?= 3.0
CAPTURE_STAGE_CENTERED_MAX_CART_ABS ?= 1.25
CAPTURE_STAGE_CART_POS_COST ?= 8.0
CAPTURE_STAGE_RAIL_MARGIN ?= 220.0
CAPTURE_STAGE_RAIL_MARGIN_START ?= 0.55
CAPTURE_STAGE_CENTERED_STREAK ?= 80.0
CAPTURE_STAGE_LOW_MOMENTUM_STREAK ?= 120.0

setup:
	bash scripts/setup_mac.sh

smoke:
	$(PYTHON) scripts/smoke_test.py

debug:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/debug6_fast.yaml

sweep:
	$(PYTHON) scripts/linear_sweep.py --config configs/sweep_n.yaml

train6:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/gradient6_curriculum.yaml

uniform6:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/uniform6_finetune.yaml --init-checkpoint runs/gradient6_curriculum/checkpoints/best.safetensors

lqr6:
	$(PYTHON) scripts/make_lqr_checkpoint.py --config configs/uniform6_near_upright_lqr.yaml

eval6:
	$(PYTHON) scripts/evaluate.py --config configs/uniform6_near_upright_lqr.yaml --checkpoint runs/uniform6_finetune/checkpoints/best.safetensors --episodes 20 --progress 1.0 --out runs/uniform6_finetune/eval_uniform6.json

render6:
	$(PYTHON) scripts/render_video.py --config configs/uniform6_near_upright_lqr.yaml --checkpoint runs/uniform6_finetune/checkpoints/best.safetensors --out runs/uniform6_finetune/six_link_uniform_success.mp4 --seconds 30 --progress 1.0 --metadata-out runs/uniform6_finetune/six_link_uniform_success.video.json --fail-on-reset

swingup-debug:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_debug.yaml

swingup6:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_uniform.yaml

swingup6-gradient-low-momentum:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_gradient_low_momentum.yaml

swingup6-uniform-low-momentum:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_uniform_low_momentum_finetune.yaml --init-checkpoint $(SWING_HANDOFF_CHECKPOINT)

capture-handoff6:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_capture_handoff.yaml

eval-swingup6:
	$(PYTHON) scripts/evaluate.py --config configs/swingup6_uniform.yaml --checkpoint runs/swingup6_uniform/checkpoints/best.safetensors --episodes 20 --progress 1.0 --out runs/swingup6_uniform/eval_swingup6.json

eval-swingup6-low-momentum:
	$(PYTHON) scripts/evaluate.py --config configs/swingup6_uniform_low_momentum_finetune.yaml --checkpoint runs/swingup6_uniform_low_momentum_finetune/checkpoints/best.safetensors --episodes 20 --progress 1.0 --out runs/swingup6_uniform_low_momentum_finetune/eval_swingup6_low_momentum.json

eval-capture-handoff6:
	$(PYTHON) scripts/evaluate.py --config configs/swingup6_capture_handoff.yaml --checkpoint runs/swingup6_capture_handoff/checkpoints/best.safetensors --episodes 20 --progress 1.0 --out runs/swingup6_capture_handoff/eval_capture_handoff6.json

render-swingup6:
	$(PYTHON) scripts/render_video.py --config configs/swingup6_uniform.yaml --checkpoint runs/swingup6_uniform/checkpoints/best.safetensors --out runs/swingup6_uniform/six_link_swingup_success.mp4 --seconds 30 --progress 1.0 --metadata-out runs/swingup6_uniform/six_link_swingup_success.video.json --fail-on-reset

probe-swingup-trajectory:
	$(PYTHON) scripts/probe_swingup_trajectory.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --out runs/swingup6_trajectory_probe/probe.json

search-swingup-trajectory:
	$(PYTHON) scripts/search_swingup_trajectory.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --out runs/swingup6_trajectory_search/search.json

search-swingup-capture:
	$(PYTHON) scripts/search_swingup_capture.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --out runs/swingup6_capture_search/search.json

search-swingup-chain:
	$(PYTHON) scripts/search_swingup_chain.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --capture-config configs/swingup6_capture_state_list_shaped.yaml --capture-checkpoint runs/swingup6_capture_state_list_shaped/checkpoints/best.safetensors --out runs/swingup6_chain_search/search.json

export-chain-states:
	$(PYTHON) scripts/search_swingup_chain.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --capture-config configs/swingup6_capture_state_list_shaped.yaml --capture-checkpoint runs/swingup6_capture_state_list_shaped/checkpoints/best.safetensors --init-controller-json runs/swingup6_chain_search/search.json --iterations 0 --population 1 --elites 1 --seconds 30 --state-max-angle 0.60 --state-max-hinge-rms 3.0 --state-out runs/swingup6_chain_search/centered_chain_states.json --out runs/swingup6_chain_search/eval_chain_state_export.json

export-swingup-states:
	$(PYTHON) scripts/export_swingup_states.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --out runs/swingup6_expert_chain/swing_states.json

export-policy-handoff-states:
	$(PYTHON) scripts/export_policy_handoff_states.py --config $(SWING_HANDOFF_CONFIG) --checkpoint $(SWING_HANDOFF_CHECKPOINT) --use-checkpoint-progress --episodes $(SWING_HANDOFF_EPISODES) --max-states $(SWING_HANDOFF_MAX_STATES) --one-per-episode --out $(SWING_HANDOFF_OUT)

eval-expert-chain:
	$(PYTHON) scripts/evaluate_expert_chain.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --out runs/swingup6_expert_chain/eval_chain.json

eval-expert-chain-learned:
	$(PYTHON) scripts/evaluate_expert_chain.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --capture-config configs/swingup6_capture_state_list_shaped.yaml --capture-checkpoint runs/swingup6_capture_state_list_shaped/checkpoints/best.safetensors --stabilize-config configs/swingup6_chain_state_list_shaped.yaml --stabilize-checkpoint runs/swingup6_chain_state_list_shaped/checkpoints/best.safetensors --out runs/swingup6_expert_chain/eval_chain_learned.json

capture-state-list6:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_capture_state_list.yaml

eval-capture-state-list6:
	$(PYTHON) scripts/evaluate.py --config configs/swingup6_capture_state_list.yaml --checkpoint runs/swingup6_capture_state_list/checkpoints/best.safetensors --episodes 20 --progress 1.0 --out runs/swingup6_capture_state_list/eval_capture_state_list6.json

capture-state-list-shaped6:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_capture_state_list_shaped.yaml

eval-capture-state-list-shaped6:
	$(PYTHON) scripts/evaluate.py --config configs/swingup6_capture_state_list_shaped.yaml --checkpoint runs/swingup6_capture_state_list_shaped/checkpoints/best.safetensors --episodes 20 --progress 1.0 --out runs/swingup6_capture_state_list_shaped/eval_capture_state_list_shaped6.json

capture-policy-handoff-shaped6:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_policy_handoff_capture_shaped.yaml

capture-policy-handoff-stage:
	$(PYTHON) scripts/train_mlx_ppo.py --config $(SWING_HANDOFF_CONFIG) --override experiment.out_dir=$(CAPTURE_STAGE_OUT) --override env.init_mode=state_list --override env.init_states_path=$(SWING_HANDOFF_OUT) --override env.init_cart_noise=0.02 --override env.init_cart_vel_noise=0.04 --override env.init_angle_noise=0.012 --override env.init_vel_noise=0.05 --override env.episode_seconds=15.0 --override env.rail_limit_start=$(CAPTURE_STAGE_RAIL_LIMIT) --override env.rail_limit_end=$(CAPTURE_STAGE_RAIL_LIMIT) --override env.reward.centered_upright_max_cart_abs=$(CAPTURE_STAGE_CENTERED_MAX_CART_ABS) --override env.reward.low_momentum_max_cart_abs=$(CAPTURE_STAGE_CENTERED_MAX_CART_ABS) --override env.reward.cart_pos=$(CAPTURE_STAGE_CART_POS_COST) --override env.reward.rail_margin=$(CAPTURE_STAGE_RAIL_MARGIN) --override env.reward.rail_margin_start=$(CAPTURE_STAGE_RAIL_MARGIN_START) --override env.reward.centered_upright_streak=$(CAPTURE_STAGE_CENTERED_STREAK) --override env.reward.centered_upright_streak_cap_seconds=5.0 --override env.reward.low_momentum_upright_streak=$(CAPTURE_STAGE_LOW_MOMENTUM_STREAK) --override env.reward.low_momentum_upright_streak_cap_seconds=5.0 --override ppo.curriculum_mode=gated --override ppo.curriculum_start_progress=$(CAPTURE_STAGE_PROGRESS) --override ppo.curriculum_step=0.0 --override ppo.curriculum_min_updates_per_stage=999999 --override ppo.eval_progress=current --override ppo.total_updates=$(CAPTURE_STAGE_UPDATES) --override ppo.num_envs=$(CAPTURE_STAGE_ENVS) --override ppo.rollout_steps=$(CAPTURE_STAGE_ROLLOUT_STEPS) --override ppo.eval_every=$(CAPTURE_STAGE_EVAL_EVERY) --override ppo.eval_episodes=$(CAPTURE_STAGE_EVAL_EPISODES) --override ppo.entropy_coef=$(CAPTURE_STAGE_ENTROPY)

eval-capture-policy-handoff-shaped6:
	$(PYTHON) scripts/evaluate.py --config configs/swingup6_policy_handoff_capture_shaped.yaml --checkpoint runs/swingup6_policy_handoff_capture_shaped/checkpoints/best.safetensors --episodes 20 --progress 1.0 --out runs/swingup6_policy_handoff_capture_shaped/eval_capture_policy_handoff_shaped6.json

chain-state-list-shaped6:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_chain_state_list_shaped.yaml

eval-chain-state-list-shaped6:
	$(PYTHON) scripts/evaluate.py --config configs/swingup6_chain_state_list_shaped.yaml --checkpoint runs/swingup6_chain_state_list_shaped/checkpoints/best.safetensors --episodes 20 --progress 1.0 --out runs/swingup6_chain_state_list_shaped/eval_chain_state_list_shaped6.json
