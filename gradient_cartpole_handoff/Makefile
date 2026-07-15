.PHONY: setup smoke roadmap-p0 test-roadmap-p0 verify-roadmap-p0 export-swingup7-xml swingup7 eval-swingup7-20 eval-swingup7-100 render-swingup7 verify-swingup7 debug sweep train6 eval6 render6 uniform6 lqr6 swingup-debug swingup6 swingup6-gradient-low-momentum swingup6-uniform-low-momentum eval-swingup6 eval-swingup6-low-momentum render-swingup6 probe-swingup-trajectory search-swingup-trajectory search-swingup-low-momentum search-swingup-sustain search-swingup-action-low-momentum search-swingup-action-cold search-linear-policy search-swingup-capture search-capture-sequence eval-mpc-capture eval-mpc-policy-handoff-stage search-swingup-chain export-swingup-states export-low-momentum-swingup-states export-policy-handoff-states export-chain-states eval-expert-chain eval-expert-chain-learned capture-handoff6 eval-capture-handoff6 capture-state-list6 eval-capture-state-list6 capture-state-list-shaped6 eval-capture-state-list-shaped6 capture-wide6 eval-capture-wide6 capture-low-momentum-velocity-curriculum capture-state-curriculum eval-capture-state-curriculum capture-lqr-residual-velocity-curriculum eval-capture-lqr-residual capture-state-curriculum-lqr-init eval-capture-state-curriculum-lqr-init capture-policy-handoff-shaped6 capture-policy-handoff-stage eval-capture-policy-handoff-stage capture-policy-handoff-curriculum eval-capture-policy-handoff-curriculum eval-capture-policy-handoff-curriculum-qvel0 eval-capture-policy-handoff-shaped6 chain-state-list-shaped6 eval-chain-state-list-shaped6

PYTHON ?= .venv/bin/python
SWINGUP7_RUN ?= runs/swingup7_uniform
SWINGUP7_CHECKPOINT ?= $(SWINGUP7_RUN)/checkpoints/best.safetensors
SWING_TRAJECTORY_SEARCH_OUT ?= runs/swingup6_trajectory_search/low_momentum_search.json
SWING_TRAJECTORY_SUSTAIN_OUT ?= runs/swingup6_trajectory_search/sustain_search.json
SWING_TRAJECTORY_STATES_OUT ?= runs/swingup6_expert_chain/swing_states_low_momentum.json
SWING_ACTION_SEARCH_OUT ?= runs/swingup6_action_search/low_momentum_search.json
SWING_ACTION_STATES_OUT ?= runs/swingup6_expert_chain/swing_states_action_low_momentum.json
SWING_ACTION_COLD_OUT ?= runs/swingup6_action_search/cold_handoff_search.json
SWING_ACTION_COLD_STATES_OUT ?= runs/swingup6_expert_chain/swing_states_action_cold_handoff.json
LINEAR_POLICY_SEARCH_OUT ?= runs/swingup6_uniform_time_linear_search
LINEAR_POLICY_SEARCH_ITERATIONS ?= 80
LINEAR_POLICY_SEARCH_POPULATION ?= 128
LINEAR_POLICY_SEARCH_ELITES ?= 12
LINEAR_POLICY_SEARCH_WORKERS ?= 8
CAPTURE_SEQUENCE_OUT ?= runs/swingup6_capture_sequence/search.json
CAPTURE_WIDE_OUT ?= runs/uniform6_capture_wide
CAPTURE_WIDE_UPDATES ?= 5000
CAPTURE_WIDE_EVAL_OUT ?= $(CAPTURE_WIDE_OUT)/eval_capture_wide20.json
CAPTURE_VEL_CURRICULUM_OUT ?= runs/swingup6_capture_low_momentum_velocity_curriculum
CAPTURE_VEL_CURRICULUM_UPDATES ?= 5000
CAPTURE_BACKWARD_CURRICULUM_OUT ?= runs/swingup6_capture_state_curriculum
CAPTURE_BACKWARD_CURRICULUM_UPDATES ?= 5000
CAPTURE_BACKWARD_CURRICULUM_CHECKPOINT ?= $(CAPTURE_BACKWARD_CURRICULUM_OUT)/checkpoints/best.safetensors
CAPTURE_BACKWARD_CURRICULUM_EVAL_OUT ?= $(CAPTURE_BACKWARD_CURRICULUM_OUT)/eval_capture_state_curriculum20.json
CAPTURE_LQR_RESIDUAL_OUT ?= runs/swingup6_capture_lqr_residual
CAPTURE_LQR_RESIDUAL_UPDATES ?= 5000
CAPTURE_LQR_RESIDUAL_CHECKPOINT ?= $(CAPTURE_LQR_RESIDUAL_OUT)/checkpoints/best.safetensors
CAPTURE_LQR_RESIDUAL_EVAL_OUT ?= $(CAPTURE_LQR_RESIDUAL_OUT)/eval_capture_lqr_residual20.json
CAPTURE_STATE_CURRICULUM_OUT ?= runs/swingup6_capture_state_curriculum_lqr_init
CAPTURE_STATE_CURRICULUM_UPDATES ?= 5000
CAPTURE_STATE_CURRICULUM_INIT ?= runs/uniform6_finetune/checkpoints/best.safetensors
CAPTURE_STATE_CURRICULUM_CHECKPOINT ?= $(CAPTURE_STATE_CURRICULUM_OUT)/checkpoints/best.safetensors
CAPTURE_STATE_CURRICULUM_EVAL_OUT ?= $(CAPTURE_STATE_CURRICULUM_OUT)/eval_capture_state_curriculum_lqr_init20.json
MPC_CAPTURE_OUT ?= runs/swingup6_mpc_capture/eval.json
MPC_CAPTURE_STATE_INDEX ?= best
MPC_CAPTURE_SECONDS ?= 8
MPC_CAPTURE_HORIZON_STEPS ?= 30
MPC_CAPTURE_REPLAN_STEPS ?= 3
MPC_CAPTURE_SAMPLES ?= 96
MPC_CAPTURE_ELITES ?= 8
MPC_CAPTURE_ITERATIONS ?= 2
MPC_CAPTURE_SIGMA ?= 0.65
MPC_CAPTURE_LQR_TARGET ?= 1.0
MPC_CAPTURE_LQR_SCALE ?= 0.5
MPC_CAPTURE_QVEL_SCALE ?= 1.0
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
CAPTURE_STAGE_HELDOUT_EPISODES ?= 20
CAPTURE_STAGE_LEARNING_RATE ?= 0.00020
CAPTURE_STAGE_ENTROPY ?= 0.006
CAPTURE_STAGE_ACTION_STD ?= 0.35
CAPTURE_STAGE_RAIL_LIMIT ?= 3.0
CAPTURE_STAGE_CENTERED_MAX_CART_ABS ?= 1.25
CAPTURE_STAGE_CART_POS_COST ?= 8.0
CAPTURE_STAGE_RAIL_MARGIN ?= 220.0
CAPTURE_STAGE_RAIL_MARGIN_START ?= 0.55
CAPTURE_STAGE_CENTERED_STREAK ?= 80.0
CAPTURE_STAGE_LOW_MOMENTUM_STREAK ?= 120.0
CAPTURE_STAGE_CHECKPOINT ?= $(CAPTURE_STAGE_OUT)/checkpoints/best.safetensors
CAPTURE_STAGE_EVAL_OUT ?= $(CAPTURE_STAGE_OUT)/eval_capture_policy_handoff_stage20.json
CAPTURE_HANDOFF_CURRICULUM_OUT ?= runs/swingup6_policy_handoff_capture_curriculum
CAPTURE_HANDOFF_CURRICULUM_UPDATES ?= 5000
CAPTURE_HANDOFF_CURRICULUM_CHECKPOINT ?= $(CAPTURE_HANDOFF_CURRICULUM_OUT)/checkpoints/best.safetensors
CAPTURE_HANDOFF_CURRICULUM_EVAL_OUT ?= $(CAPTURE_HANDOFF_CURRICULUM_OUT)/eval_capture_policy_handoff_curriculum20.json
CAPTURE_HANDOFF_CURRICULUM_QVEL0_EVAL_OUT ?= $(CAPTURE_HANDOFF_CURRICULUM_OUT)/eval_capture_policy_handoff_curriculum20_qvel0.json
CAPTURE_HANDOFF_INIT_STATE_MIN_COUNT ?= 1
CAPTURE_HANDOFF_INIT_QVEL_SCALE_START ?= 0.0
CAPTURE_HANDOFF_INIT_QVEL_SCALE_END ?= 1.0
CAPTURE_HANDOFF_CART_NOISE_START ?= 0.0
CAPTURE_HANDOFF_CART_NOISE_END ?= 0.02
CAPTURE_HANDOFF_CART_VEL_NOISE_START ?= 0.0
CAPTURE_HANDOFF_CART_VEL_NOISE_END ?= 0.04
CAPTURE_HANDOFF_ANGLE_NOISE_START ?= 0.0
CAPTURE_HANDOFF_ANGLE_NOISE_END ?= 0.012
CAPTURE_HANDOFF_VEL_NOISE_START ?= 0.0
CAPTURE_HANDOFF_VEL_NOISE_END ?= 0.05

setup:
	bash scripts/setup_mac.sh

roadmap-p0: test-roadmap-p0 export-swingup7-xml verify-roadmap-p0

test-roadmap-p0:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

verify-roadmap-p0:
	$(PYTHON) scripts/verify_roadmap.py --benchmark-only --report runs/roadmap_p0/benchmark_verification.json

export-swingup7-xml:
	$(PYTHON) scripts/export_xml.py --config configs/swingup7_uniform.yaml --progress 1.0 --out runs/swingup7_uniform/model.xml

swingup7:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup7_uniform.yaml

eval-swingup7-20:
	$(PYTHON) scripts/evaluate.py --config configs/swingup7_uniform.yaml --checkpoint $(SWINGUP7_CHECKPOINT) --episodes 20 --progress 1.0 --out $(SWINGUP7_RUN)/eval_swingup7_20.json

eval-swingup7-100:
	$(PYTHON) scripts/evaluate.py --config configs/swingup7_uniform.yaml --checkpoint $(SWINGUP7_CHECKPOINT) --episodes 100 --progress 1.0 --out $(SWINGUP7_RUN)/eval_swingup7_100.json

render-swingup7:
	$(PYTHON) scripts/render_video.py --config configs/swingup7_uniform.yaml --checkpoint $(SWINGUP7_CHECKPOINT) --out $(SWINGUP7_RUN)/seven_link_swingup_success.mp4 --seconds 30 --progress 1.0 --seed 14551 --metadata-out $(SWINGUP7_RUN)/seven_link_swingup_success.video.json --no-reset-on-done --fail-on-reset

verify-swingup7:
	$(PYTHON) scripts/verify_roadmap.py --run-dir $(SWINGUP7_RUN) --report $(SWINGUP7_RUN)/verification.json

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

search-swingup-low-momentum:
	$(PYTHON) scripts/search_swingup_trajectory.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --seconds 16 --score-mode low_momentum --handoff-min-time 4.0 --handoff-max-cart-abs 1.25 --out $(SWING_TRAJECTORY_SEARCH_OUT)

search-swingup-sustain:
	$(PYTHON) scripts/search_swingup_trajectory.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --seconds 30 --score-mode sustain --init-controller-json $(SWING_TRAJECTORY_SEARCH_OUT) --trajectory-seconds-min 6 --trajectory-seconds-max 30 --handoff-max-cart-abs 1.50 --out $(SWING_TRAJECTORY_SUSTAIN_OUT)

search-swingup-action-low-momentum:
	$(PYTHON) scripts/search_swingup_action_sequence.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --seconds 16 --score-mode low_momentum --handoff-min-time 4.0 --handoff-max-cart-abs 1.25 --init-controller-json $(SWING_TRAJECTORY_SEARCH_OUT) --out $(SWING_ACTION_SEARCH_OUT) --state-out $(SWING_ACTION_STATES_OUT)

search-swingup-action-cold:
	$(PYTHON) scripts/search_swingup_action_sequence.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --seconds 20 --score-mode cold_handoff --handoff-min-time 4.0 --handoff-max-cart-abs 2.5 --init-controller-json $(SWING_ACTION_SEARCH_OUT) --out $(SWING_ACTION_COLD_OUT) --state-out $(SWING_ACTION_COLD_STATES_OUT)

search-linear-policy:
	$(PYTHON) scripts/search_linear_policy.py --config configs/swingup6_uniform_time_linear.yaml --out-dir $(LINEAR_POLICY_SEARCH_OUT) --progress 1.0 --zero-noise --seconds 8 --iterations $(LINEAR_POLICY_SEARCH_ITERATIONS) --population $(LINEAR_POLICY_SEARCH_POPULATION) --elites $(LINEAR_POLICY_SEARCH_ELITES) --workers $(LINEAR_POLICY_SEARCH_WORKERS)

search-swingup-capture:
	$(PYTHON) scripts/search_swingup_capture.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --out runs/swingup6_capture_search/search.json

search-capture-sequence:
	$(PYTHON) scripts/search_capture_sequence.py --config configs/swingup6_uniform.yaml --state-json $(SWING_TRAJECTORY_STATES_OUT) --state-index best --seconds 8 --out $(CAPTURE_SEQUENCE_OUT)

eval-mpc-capture:
	$(PYTHON) scripts/evaluate_mpc_capture.py --config configs/swingup6_uniform.yaml --state-json $(SWING_TRAJECTORY_STATES_OUT) --state-index $(MPC_CAPTURE_STATE_INDEX) --seconds $(MPC_CAPTURE_SECONDS) --horizon-steps $(MPC_CAPTURE_HORIZON_STEPS) --replan-steps $(MPC_CAPTURE_REPLAN_STEPS) --samples $(MPC_CAPTURE_SAMPLES) --elites $(MPC_CAPTURE_ELITES) --iterations $(MPC_CAPTURE_ITERATIONS) --sigma $(MPC_CAPTURE_SIGMA) --lqr-baseline --lqr-target $(MPC_CAPTURE_LQR_TARGET) --lqr-scale $(MPC_CAPTURE_LQR_SCALE) --override env.init_qvel_scale=$(MPC_CAPTURE_QVEL_SCALE) --out $(MPC_CAPTURE_OUT)

eval-mpc-policy-handoff-stage:
	$(PYTHON) scripts/evaluate_mpc_capture.py --config $(SWING_HANDOFF_CONFIG) --state-json $(SWING_HANDOFF_OUT) --state-index $(MPC_CAPTURE_STATE_INDEX) --progress $(CAPTURE_STAGE_PROGRESS) --seconds $(MPC_CAPTURE_SECONDS) --horizon-steps $(MPC_CAPTURE_HORIZON_STEPS) --replan-steps $(MPC_CAPTURE_REPLAN_STEPS) --samples $(MPC_CAPTURE_SAMPLES) --elites $(MPC_CAPTURE_ELITES) --iterations $(MPC_CAPTURE_ITERATIONS) --sigma $(MPC_CAPTURE_SIGMA) --lqr-baseline --lqr-target $(MPC_CAPTURE_LQR_TARGET) --lqr-scale $(MPC_CAPTURE_LQR_SCALE) --override env.rail_limit_start=$(CAPTURE_STAGE_RAIL_LIMIT) --override env.rail_limit_end=$(CAPTURE_STAGE_RAIL_LIMIT) --override env.init_qvel_scale=$(MPC_CAPTURE_QVEL_SCALE) --out $(MPC_CAPTURE_OUT)

search-swingup-chain:
	$(PYTHON) scripts/search_swingup_chain.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --capture-config configs/swingup6_capture_state_list_shaped.yaml --capture-checkpoint runs/swingup6_capture_state_list_shaped/checkpoints/best.safetensors --out runs/swingup6_chain_search/search.json

export-chain-states:
	$(PYTHON) scripts/search_swingup_chain.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --capture-config configs/swingup6_capture_state_list_shaped.yaml --capture-checkpoint runs/swingup6_capture_state_list_shaped/checkpoints/best.safetensors --init-controller-json runs/swingup6_chain_search/search.json --iterations 0 --population 1 --elites 1 --seconds 30 --state-max-angle 0.60 --state-max-hinge-rms 3.0 --state-out runs/swingup6_chain_search/centered_chain_states.json --out runs/swingup6_chain_search/eval_chain_state_export.json

export-swingup-states:
	$(PYTHON) scripts/export_swingup_states.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --out runs/swingup6_expert_chain/swing_states.json

export-low-momentum-swingup-states:
	$(PYTHON) scripts/export_swingup_states.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --controller-json $(SWING_TRAJECTORY_SEARCH_OUT) --max-angle 0.35 --min-time 4.0 --max-hinge-rms 2.5 --max-cart-abs 2.75 --max-cart-velocity 2.0 --keep-best 96 --out $(SWING_TRAJECTORY_STATES_OUT)

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

capture-wide6:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/uniform6_capture_wide.yaml --override experiment.out_dir=$(CAPTURE_WIDE_OUT) --override ppo.total_updates=$(CAPTURE_WIDE_UPDATES)

eval-capture-wide6:
	$(PYTHON) scripts/evaluate.py --config configs/uniform6_capture_wide.yaml --checkpoint $(CAPTURE_WIDE_OUT)/checkpoints/best.safetensors --episodes 20 --progress 1.0 --override experiment.out_dir=$(CAPTURE_WIDE_OUT) --out $(CAPTURE_WIDE_EVAL_OUT)

capture-low-momentum-velocity-curriculum:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_capture_state_list_shaped.yaml --override experiment.out_dir=$(CAPTURE_VEL_CURRICULUM_OUT) --override env.init_states_path=$(SWING_TRAJECTORY_STATES_OUT) --override env.init_qvel_scale_start=0.0 --override env.init_qvel_scale_end=1.0 --override env.init_cart_noise=0.02 --override env.init_cart_vel_noise=0.04 --override env.init_angle_noise=0.012 --override env.init_vel_noise=0.05 --override env.reward.centered_upright_max_cart_abs=1.25 --override env.reward.low_momentum_max_cart_abs=1.25 --override env.reward.cart_pos=8.0 --override env.reward.rail_margin=220.0 --override env.reward.rail_margin_start=0.55 --override env.reward.centered_upright_streak=80.0 --override env.reward.centered_upright_streak_cap_seconds=5.0 --override env.reward.low_momentum_upright_streak=120.0 --override env.reward.low_momentum_upright_streak_cap_seconds=5.0 --override ppo.total_updates=$(CAPTURE_VEL_CURRICULUM_UPDATES) --override ppo.entropy_coef=0.006

capture-state-curriculum:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_capture_state_list_shaped.yaml --override experiment.out_dir=$(CAPTURE_BACKWARD_CURRICULUM_OUT) --override env.init_states_path=$(SWING_TRAJECTORY_STATES_OUT) --override env.init_state_curriculum=quality_prefix --override env.init_state_curriculum_min_count=1 --override env.init_qpos_scale_start=0.0 --override env.init_qpos_scale_end=1.0 --override env.init_qpos_cart_target=0.0 --override env.init_qvel_scale_start=0.0 --override env.init_qvel_scale_end=1.0 --override env.init_cart_noise=0.015 --override env.init_cart_vel_noise=0.03 --override env.init_angle_noise=0.01 --override env.init_vel_noise=0.04 --override env.reward.centered_upright_max_cart_abs=1.25 --override env.reward.low_momentum_max_cart_abs=1.25 --override env.reward.cart_pos=8.0 --override env.reward.rail_margin=220.0 --override env.reward.rail_margin_start=0.55 --override env.reward.centered_upright_streak=80.0 --override env.reward.centered_upright_streak_cap_seconds=5.0 --override env.reward.low_momentum_upright_streak=120.0 --override env.reward.low_momentum_upright_streak_cap_seconds=5.0 --override ppo.total_updates=$(CAPTURE_BACKWARD_CURRICULUM_UPDATES) --override ppo.entropy_coef=0.006 --override ppo.action_std_init=0.35

eval-capture-state-curriculum:
	$(PYTHON) scripts/evaluate.py --config configs/swingup6_capture_state_list_shaped.yaml --checkpoint $(CAPTURE_BACKWARD_CURRICULUM_CHECKPOINT) --episodes 20 --progress 1.0 --override env.init_states_path=$(SWING_TRAJECTORY_STATES_OUT) --override env.init_state_curriculum=quality_prefix --override env.init_state_curriculum_min_count=1 --override env.init_qpos_scale_start=0.0 --override env.init_qpos_scale_end=1.0 --override env.init_qpos_cart_target=0.0 --override env.init_qvel_scale_start=0.0 --override env.init_qvel_scale_end=1.0 --override env.init_cart_noise=0.015 --override env.init_cart_vel_noise=0.03 --override env.init_angle_noise=0.01 --override env.init_vel_noise=0.04 --out $(CAPTURE_BACKWARD_CURRICULUM_EVAL_OUT)

capture-lqr-residual-velocity-curriculum:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_capture_lqr_residual.yaml --override experiment.out_dir=$(CAPTURE_LQR_RESIDUAL_OUT) --override env.init_states_path=$(SWING_TRAJECTORY_STATES_OUT) --override ppo.total_updates=$(CAPTURE_LQR_RESIDUAL_UPDATES)

eval-capture-lqr-residual:
	$(PYTHON) scripts/evaluate.py --config configs/swingup6_capture_lqr_residual.yaml --checkpoint $(CAPTURE_LQR_RESIDUAL_CHECKPOINT) --episodes 20 --progress 1.0 --override env.init_states_path=$(SWING_TRAJECTORY_STATES_OUT) --out $(CAPTURE_LQR_RESIDUAL_EVAL_OUT)

capture-state-curriculum-lqr-init:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_capture_state_curriculum_lqr_init.yaml --init-checkpoint $(CAPTURE_STATE_CURRICULUM_INIT) --override experiment.out_dir=$(CAPTURE_STATE_CURRICULUM_OUT) --override env.init_states_path=$(SWING_TRAJECTORY_STATES_OUT) --override ppo.total_updates=$(CAPTURE_STATE_CURRICULUM_UPDATES)

eval-capture-state-curriculum-lqr-init:
	$(PYTHON) scripts/evaluate.py --config configs/swingup6_capture_state_curriculum_lqr_init.yaml --checkpoint $(CAPTURE_STATE_CURRICULUM_CHECKPOINT) --episodes 20 --progress 1.0 --override env.init_states_path=$(SWING_TRAJECTORY_STATES_OUT) --out $(CAPTURE_STATE_CURRICULUM_EVAL_OUT)

eval-capture-state-list-shaped6:
	$(PYTHON) scripts/evaluate.py --config configs/swingup6_capture_state_list_shaped.yaml --checkpoint runs/swingup6_capture_state_list_shaped/checkpoints/best.safetensors --episodes 20 --progress 1.0 --out runs/swingup6_capture_state_list_shaped/eval_capture_state_list_shaped6.json

capture-policy-handoff-shaped6:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_policy_handoff_capture_shaped.yaml

capture-policy-handoff-stage:
	$(PYTHON) scripts/train_mlx_ppo.py --config $(SWING_HANDOFF_CONFIG) --override experiment.out_dir=$(CAPTURE_STAGE_OUT) --override env.init_mode=state_list --override env.init_states_path=$(SWING_HANDOFF_OUT) --override env.init_cart_noise=0.02 --override env.init_cart_vel_noise=0.04 --override env.init_angle_noise=0.012 --override env.init_vel_noise=0.05 --override env.episode_seconds=15.0 --override env.rail_limit_start=$(CAPTURE_STAGE_RAIL_LIMIT) --override env.rail_limit_end=$(CAPTURE_STAGE_RAIL_LIMIT) --override env.reward.centered_upright_max_cart_abs=$(CAPTURE_STAGE_CENTERED_MAX_CART_ABS) --override env.reward.low_momentum_max_cart_abs=$(CAPTURE_STAGE_CENTERED_MAX_CART_ABS) --override env.reward.cart_pos=$(CAPTURE_STAGE_CART_POS_COST) --override env.reward.rail_margin=$(CAPTURE_STAGE_RAIL_MARGIN) --override env.reward.rail_margin_start=$(CAPTURE_STAGE_RAIL_MARGIN_START) --override env.reward.centered_upright_streak=$(CAPTURE_STAGE_CENTERED_STREAK) --override env.reward.centered_upright_streak_cap_seconds=5.0 --override env.reward.low_momentum_upright_streak=$(CAPTURE_STAGE_LOW_MOMENTUM_STREAK) --override env.reward.low_momentum_upright_streak_cap_seconds=5.0 --override ppo.curriculum_mode=gated --override ppo.curriculum_start_progress=$(CAPTURE_STAGE_PROGRESS) --override ppo.curriculum_step=0.0 --override ppo.curriculum_min_updates_per_stage=999999 --override ppo.eval_progress=current --override ppo.total_updates=$(CAPTURE_STAGE_UPDATES) --override ppo.num_envs=$(CAPTURE_STAGE_ENVS) --override ppo.rollout_steps=$(CAPTURE_STAGE_ROLLOUT_STEPS) --override ppo.learning_rate=$(CAPTURE_STAGE_LEARNING_RATE) --override ppo.eval_every=$(CAPTURE_STAGE_EVAL_EVERY) --override ppo.eval_episodes=$(CAPTURE_STAGE_EVAL_EPISODES) --override ppo.entropy_coef=$(CAPTURE_STAGE_ENTROPY) --override ppo.action_std_init=$(CAPTURE_STAGE_ACTION_STD)

eval-capture-policy-handoff-stage:
	$(PYTHON) scripts/evaluate.py --config $(SWING_HANDOFF_CONFIG) --checkpoint $(CAPTURE_STAGE_CHECKPOINT) --episodes $(CAPTURE_STAGE_HELDOUT_EPISODES) --progress $(CAPTURE_STAGE_PROGRESS) --override env.init_mode=state_list --override env.init_states_path=$(SWING_HANDOFF_OUT) --override env.init_cart_noise=0.02 --override env.init_cart_vel_noise=0.04 --override env.init_angle_noise=0.012 --override env.init_vel_noise=0.05 --override env.episode_seconds=15.0 --override env.rail_limit_start=$(CAPTURE_STAGE_RAIL_LIMIT) --override env.rail_limit_end=$(CAPTURE_STAGE_RAIL_LIMIT) --override env.reward.centered_upright_max_cart_abs=$(CAPTURE_STAGE_CENTERED_MAX_CART_ABS) --override env.reward.low_momentum_max_cart_abs=$(CAPTURE_STAGE_CENTERED_MAX_CART_ABS) --override env.reward.cart_pos=$(CAPTURE_STAGE_CART_POS_COST) --override env.reward.rail_margin=$(CAPTURE_STAGE_RAIL_MARGIN) --override env.reward.rail_margin_start=$(CAPTURE_STAGE_RAIL_MARGIN_START) --override env.reward.centered_upright_streak=$(CAPTURE_STAGE_CENTERED_STREAK) --override env.reward.centered_upright_streak_cap_seconds=5.0 --override env.reward.low_momentum_upright_streak=$(CAPTURE_STAGE_LOW_MOMENTUM_STREAK) --override env.reward.low_momentum_upright_streak_cap_seconds=5.0 --out $(CAPTURE_STAGE_EVAL_OUT)

capture-policy-handoff-curriculum:
	$(PYTHON) scripts/train_mlx_ppo.py --config $(SWING_HANDOFF_CONFIG) --override experiment.out_dir=$(CAPTURE_HANDOFF_CURRICULUM_OUT) --override env.plant_progress=$(CAPTURE_STAGE_PROGRESS) --override env.init_mode=state_list --override env.init_states_path=$(SWING_HANDOFF_OUT) --override env.init_state_curriculum=quality_prefix --override env.init_state_curriculum_min_count=$(CAPTURE_HANDOFF_INIT_STATE_MIN_COUNT) --override env.init_qvel_scale_start=$(CAPTURE_HANDOFF_INIT_QVEL_SCALE_START) --override env.init_qvel_scale_end=$(CAPTURE_HANDOFF_INIT_QVEL_SCALE_END) --override env.init_cart_noise_start=$(CAPTURE_HANDOFF_CART_NOISE_START) --override env.init_cart_noise_end=$(CAPTURE_HANDOFF_CART_NOISE_END) --override env.init_cart_vel_noise_start=$(CAPTURE_HANDOFF_CART_VEL_NOISE_START) --override env.init_cart_vel_noise_end=$(CAPTURE_HANDOFF_CART_VEL_NOISE_END) --override env.init_angle_noise_start=$(CAPTURE_HANDOFF_ANGLE_NOISE_START) --override env.init_angle_noise_end=$(CAPTURE_HANDOFF_ANGLE_NOISE_END) --override env.init_vel_noise_start=$(CAPTURE_HANDOFF_VEL_NOISE_START) --override env.init_vel_noise_end=$(CAPTURE_HANDOFF_VEL_NOISE_END) --override env.episode_seconds=15.0 --override env.rail_limit_start=$(CAPTURE_STAGE_RAIL_LIMIT) --override env.rail_limit_end=$(CAPTURE_STAGE_RAIL_LIMIT) --override env.reward.centered_upright_max_cart_abs=$(CAPTURE_STAGE_CENTERED_MAX_CART_ABS) --override env.reward.low_momentum_max_cart_abs=$(CAPTURE_STAGE_CENTERED_MAX_CART_ABS) --override env.reward.cart_pos=$(CAPTURE_STAGE_CART_POS_COST) --override env.reward.rail_margin=$(CAPTURE_STAGE_RAIL_MARGIN) --override env.reward.rail_margin_start=$(CAPTURE_STAGE_RAIL_MARGIN_START) --override env.reward.centered_upright_streak=$(CAPTURE_STAGE_CENTERED_STREAK) --override env.reward.centered_upright_streak_cap_seconds=5.0 --override env.reward.low_momentum_upright_streak=$(CAPTURE_STAGE_LOW_MOMENTUM_STREAK) --override env.reward.low_momentum_upright_streak_cap_seconds=5.0 --override ppo.curriculum_mode=linear --override ppo.eval_progress=final --override ppo.total_updates=$(CAPTURE_HANDOFF_CURRICULUM_UPDATES) --override ppo.num_envs=$(CAPTURE_STAGE_ENVS) --override ppo.rollout_steps=$(CAPTURE_STAGE_ROLLOUT_STEPS) --override ppo.learning_rate=$(CAPTURE_STAGE_LEARNING_RATE) --override ppo.eval_every=$(CAPTURE_STAGE_EVAL_EVERY) --override ppo.eval_episodes=$(CAPTURE_STAGE_EVAL_EPISODES) --override ppo.entropy_coef=$(CAPTURE_STAGE_ENTROPY) --override ppo.action_std_init=$(CAPTURE_STAGE_ACTION_STD)

eval-capture-policy-handoff-curriculum:
	$(PYTHON) scripts/evaluate.py --config $(SWING_HANDOFF_CONFIG) --checkpoint $(CAPTURE_HANDOFF_CURRICULUM_CHECKPOINT) --episodes $(CAPTURE_STAGE_HELDOUT_EPISODES) --progress 1.0 --override env.plant_progress=$(CAPTURE_STAGE_PROGRESS) --override env.init_mode=state_list --override env.init_states_path=$(SWING_HANDOFF_OUT) --override env.init_state_curriculum=quality_prefix --override env.init_state_curriculum_min_count=$(CAPTURE_HANDOFF_INIT_STATE_MIN_COUNT) --override env.init_qvel_scale_start=$(CAPTURE_HANDOFF_INIT_QVEL_SCALE_START) --override env.init_qvel_scale_end=$(CAPTURE_HANDOFF_INIT_QVEL_SCALE_END) --override env.init_cart_noise_start=$(CAPTURE_HANDOFF_CART_NOISE_START) --override env.init_cart_noise_end=$(CAPTURE_HANDOFF_CART_NOISE_END) --override env.init_cart_vel_noise_start=$(CAPTURE_HANDOFF_CART_VEL_NOISE_START) --override env.init_cart_vel_noise_end=$(CAPTURE_HANDOFF_CART_VEL_NOISE_END) --override env.init_angle_noise_start=$(CAPTURE_HANDOFF_ANGLE_NOISE_START) --override env.init_angle_noise_end=$(CAPTURE_HANDOFF_ANGLE_NOISE_END) --override env.init_vel_noise_start=$(CAPTURE_HANDOFF_VEL_NOISE_START) --override env.init_vel_noise_end=$(CAPTURE_HANDOFF_VEL_NOISE_END) --override env.episode_seconds=15.0 --override env.rail_limit_start=$(CAPTURE_STAGE_RAIL_LIMIT) --override env.rail_limit_end=$(CAPTURE_STAGE_RAIL_LIMIT) --override env.reward.centered_upright_max_cart_abs=$(CAPTURE_STAGE_CENTERED_MAX_CART_ABS) --override env.reward.low_momentum_max_cart_abs=$(CAPTURE_STAGE_CENTERED_MAX_CART_ABS) --override env.reward.cart_pos=$(CAPTURE_STAGE_CART_POS_COST) --override env.reward.rail_margin=$(CAPTURE_STAGE_RAIL_MARGIN) --override env.reward.rail_margin_start=$(CAPTURE_STAGE_RAIL_MARGIN_START) --override env.reward.centered_upright_streak=$(CAPTURE_STAGE_CENTERED_STREAK) --override env.reward.centered_upright_streak_cap_seconds=5.0 --override env.reward.low_momentum_upright_streak=$(CAPTURE_STAGE_LOW_MOMENTUM_STREAK) --override env.reward.low_momentum_upright_streak_cap_seconds=5.0 --out $(CAPTURE_HANDOFF_CURRICULUM_EVAL_OUT)

eval-capture-policy-handoff-curriculum-qvel0:
	$(PYTHON) scripts/evaluate.py --config $(SWING_HANDOFF_CONFIG) --checkpoint $(CAPTURE_HANDOFF_CURRICULUM_CHECKPOINT) --episodes $(CAPTURE_STAGE_HELDOUT_EPISODES) --progress 0.0 --override env.plant_progress=$(CAPTURE_STAGE_PROGRESS) --override env.init_mode=state_list --override env.init_states_path=$(SWING_HANDOFF_OUT) --override env.init_state_curriculum=quality_prefix --override env.init_state_curriculum_min_count=$(CAPTURE_HANDOFF_INIT_STATE_MIN_COUNT) --override env.init_qvel_scale_start=$(CAPTURE_HANDOFF_INIT_QVEL_SCALE_START) --override env.init_qvel_scale_end=$(CAPTURE_HANDOFF_INIT_QVEL_SCALE_END) --override env.init_cart_noise_start=$(CAPTURE_HANDOFF_CART_NOISE_START) --override env.init_cart_noise_end=$(CAPTURE_HANDOFF_CART_NOISE_END) --override env.init_cart_vel_noise_start=$(CAPTURE_HANDOFF_CART_VEL_NOISE_START) --override env.init_cart_vel_noise_end=$(CAPTURE_HANDOFF_CART_VEL_NOISE_END) --override env.init_angle_noise_start=$(CAPTURE_HANDOFF_ANGLE_NOISE_START) --override env.init_angle_noise_end=$(CAPTURE_HANDOFF_ANGLE_NOISE_END) --override env.init_vel_noise_start=$(CAPTURE_HANDOFF_VEL_NOISE_START) --override env.init_vel_noise_end=$(CAPTURE_HANDOFF_VEL_NOISE_END) --override env.episode_seconds=15.0 --override env.rail_limit_start=$(CAPTURE_STAGE_RAIL_LIMIT) --override env.rail_limit_end=$(CAPTURE_STAGE_RAIL_LIMIT) --override env.reward.centered_upright_max_cart_abs=$(CAPTURE_STAGE_CENTERED_MAX_CART_ABS) --override env.reward.low_momentum_max_cart_abs=$(CAPTURE_STAGE_CENTERED_MAX_CART_ABS) --override env.reward.cart_pos=$(CAPTURE_STAGE_CART_POS_COST) --override env.reward.rail_margin=$(CAPTURE_STAGE_RAIL_MARGIN) --override env.reward.rail_margin_start=$(CAPTURE_STAGE_RAIL_MARGIN_START) --override env.reward.centered_upright_streak=$(CAPTURE_STAGE_CENTERED_STREAK) --override env.reward.centered_upright_streak_cap_seconds=5.0 --override env.reward.low_momentum_upright_streak=$(CAPTURE_STAGE_LOW_MOMENTUM_STREAK) --override env.reward.low_momentum_upright_streak_cap_seconds=5.0 --out $(CAPTURE_HANDOFF_CURRICULUM_QVEL0_EVAL_OUT)

eval-capture-policy-handoff-shaped6:
	$(PYTHON) scripts/evaluate.py --config configs/swingup6_policy_handoff_capture_shaped.yaml --checkpoint runs/swingup6_policy_handoff_capture_shaped/checkpoints/best.safetensors --episodes 20 --progress 1.0 --out runs/swingup6_policy_handoff_capture_shaped/eval_capture_policy_handoff_shaped6.json

chain-state-list-shaped6:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_chain_state_list_shaped.yaml

eval-chain-state-list-shaped6:
	$(PYTHON) scripts/evaluate.py --config configs/swingup6_chain_state_list_shaped.yaml --checkpoint runs/swingup6_chain_state_list_shaped/checkpoints/best.safetensors --episodes 20 --progress 1.0 --out runs/swingup6_chain_state_list_shaped/eval_chain_state_list_shaped6.json
