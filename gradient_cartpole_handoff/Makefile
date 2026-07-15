.PHONY: setup setup-sb3 setup-fddp smoke roadmap-p0 test-roadmap-p0 verify-roadmap-p0 export-swingup7-xml generate-capture-envelope capture-envelope6 capture-hybrid-policy6 capture-boundary-lqr mine-capture-boundary fit-capture-funnel search-capture-target-teacher build-capture-target-teachers build-feedback-mpc-teachers build-capture-supervisor-dataset distill-capture-supervisor eval-capture-supervisor collect-capture-dagger label-capture-dagger-lqr label-capture-dagger-target label-capture-dagger-feedback eval-adaptive-capture-boundary analyze-capture-modal sweep-capture-modal-feedback refine-adaptive-capture-boundary eval-transfer-capture-boundary search-capture-hybrid eval-feedback-mpc-state eval-predictive-sampling-state refine-feedback-mpc-boundary search-ilqr-capture search-fddp-capture continue-fddp-capture refine-ilqr-capture-chain eval-ilqr-chain-basin recenter-trajectory-controller eval-trajectory-funnel-library search-multiple-shooting-capture analyze-refined-capture-modal capture-sac-boundary mine-capture-failures distill-capture-lqr eval-capture-envelope6 eval-capture-envelope-lqr eval-capture-envelope-mpc swingup7 eval-swingup7-20 eval-swingup7-100 render-swingup7 verify-swingup7 debug sweep train6 eval6 render6 uniform6 lqr6 swingup-debug swingup6 swingup6-gradient-low-momentum swingup6-uniform-low-momentum eval-swingup6 eval-swingup6-low-momentum render-swingup6 probe-swingup-trajectory search-swingup-trajectory search-swingup-low-momentum search-swingup-sustain search-swingup-action-low-momentum search-swingup-action-cold search-linear-policy search-swingup-capture search-capture-sequence eval-mpc-capture eval-mpc-policy-handoff-stage search-swingup-chain search-swingup-funnel export-swingup-states export-low-momentum-swingup-states export-policy-handoff-states export-chain-states eval-expert-chain eval-expert-chain-learned capture-handoff6 eval-capture-handoff6 capture-state-list6 eval-capture-state-list6 capture-state-list-shaped6 eval-capture-state-list-shaped6 capture-wide6 eval-capture-wide6 capture-low-momentum-velocity-curriculum capture-state-curriculum eval-capture-state-curriculum capture-lqr-residual-velocity-curriculum eval-capture-lqr-residual capture-state-curriculum-lqr-init eval-capture-state-curriculum-lqr-init capture-policy-handoff-shaped6 capture-policy-handoff-stage eval-capture-policy-handoff-stage capture-policy-handoff-curriculum eval-capture-policy-handoff-curriculum eval-capture-policy-handoff-curriculum-qvel0 eval-capture-policy-handoff-shaped6 chain-state-list-shaped6 eval-chain-state-list-shaped6

PYTHON ?= .venv/bin/python
SWINGUP7_RUN ?= runs/swingup7_uniform
SWINGUP7_CHECKPOINT ?= $(SWINGUP7_RUN)/checkpoints/best.safetensors
CAPTURE_ENVELOPE_DIR ?= runs/p1_capture_envelope
CAPTURE_ENVELOPE_RUN ?= runs/swingup6_capture_envelope
CAPTURE_ENVELOPE_CHECKPOINT ?= $(CAPTURE_ENVELOPE_RUN)/checkpoints/best.safetensors
CAPTURE_HARD_STATES ?= $(CAPTURE_ENVELOPE_DIR)/train_hard.json
CAPTURE_BOUNDARY_PROGRESS ?= 0.0625
CAPTURE_BOUNDARY_LQR ?= $(CAPTURE_ENVELOPE_DIR)/lqr_direct_scale130/best.safetensors
CAPTURE_BOUNDARY_ROLLOUTS ?= $(CAPTURE_ENVELOPE_DIR)/train_hard_00625.json
CAPTURE_FUNNEL_MODEL ?= $(CAPTURE_ENVELOPE_DIR)/funnel_lqr_scale130_p00625.json
CAPTURE_TARGET_TEACHER_STATE ?= 735
CAPTURE_TARGET_TEACHER_OUT ?= runs/p1_capture_target_teachers/validation_$(CAPTURE_TARGET_TEACHER_STATE).json
CAPTURE_TARGET_TEACHERS_OUT ?= runs/p1_capture_target_teachers/train_teachers_32.json
CAPTURE_FEEDBACK_TEACHERS_OUT ?= runs/p1_capture_feedback_mpc/train_teachers_8_p007.json
CAPTURE_SUPERVISOR_LABELS ?= runs/p1_capture_supervisor/train_labels_p007.npz
CAPTURE_SUPERVISOR_RUN ?= runs/swingup6_capture_supervisor_distilled
CAPTURE_SUPERVISOR_CHECKPOINT ?= $(CAPTURE_SUPERVISOR_RUN)/checkpoints/best.safetensors
CAPTURE_DAGGER_CHECKPOINT ?= runs/swingup6_capture_hybrid_lyapunov_ppo_probe_100/checkpoints/best.safetensors
CAPTURE_DAGGER_QUERIES ?= runs/p1_capture_dagger/round1_train_queries.json
CAPTURE_DAGGER_LQR_LABELS ?= runs/p1_capture_dagger/round1_lqr_labels.json
CAPTURE_ADAPTIVE_PROGRESS ?= 0.065
CAPTURE_ADAPTIVE_LIMIT ?= 256
CAPTURE_ADAPTIVE_OUT ?= runs/p1_capture_target_teachers/eval_adaptive_validation$(CAPTURE_ADAPTIVE_LIMIT)_p$(subst .,,$(CAPTURE_ADAPTIVE_PROGRESS)).json
CAPTURE_MODAL_ANALYSIS_OUT ?= runs/p1_capture_modal/modal_analysis_validation256_p0065.json
CAPTURE_MODAL_SWEEP_OUT ?= runs/p1_capture_modal/modal_group10_gain_sweep_fine_p0065.json
CAPTURE_REFINED_ADAPTIVE_OUT ?= runs/p1_capture_target_teachers/eval_adaptive_refined_validation256_p0065.json
CAPTURE_REFINED_MODAL_ANALYSIS_OUT ?= runs/p1_capture_modal/modal_analysis_refined_validation256_p0065.json
CAPTURE_TRANSFER_PROGRESS ?= 0.0675
CAPTURE_TRANSFER_OUT ?= runs/p1_capture_target_teachers/eval_transfer_validation256_p00675.json
CAPTURE_HYBRID_STATE ?= 475
CAPTURE_HYBRID_SEED ?= 62856
CAPTURE_HYBRID_OUT ?= runs/p1_capture_hybrid/validation_$(CAPTURE_HYBRID_STATE)_p0065.json
CAPTURE_FEEDBACK_MPC_STATE ?= 1998
CAPTURE_FEEDBACK_MPC_SEED ?= 62636
CAPTURE_FEEDBACK_MPC_OUT ?= runs/p1_capture_feedback_mpc/validation_$(CAPTURE_FEEDBACK_MPC_STATE)_p0065.json
CAPTURE_PREDICTIVE_STATE ?= 674
CAPTURE_PREDICTIVE_PROGRESS ?= 1.0
CAPTURE_PREDICTIVE_OUT ?= runs/p1_capture_predictive_sampling/validation_$(CAPTURE_PREDICTIVE_STATE).json
CAPTURE_FEEDBACK_MPC_INITIAL ?= $(CAPTURE_REFINED_ADAPTIVE_OUT)
CAPTURE_FEEDBACK_MPC_BATCH_OUT ?= runs/p1_capture_feedback_mpc/eval_refined_validation256_p0065.json
CAPTURE_FEEDBACK_MPC_SECONDS ?= 2.0
CAPTURE_FEEDBACK_MPC_HANDOFF_CART ?= 2.4
CAPTURE_FEEDBACK_MPC_ITERATIONS ?= 3
CAPTURE_FEEDBACK_MPC_POPULATION ?= 64
CAPTURE_FEEDBACK_MPC_ELITES ?= 8
CAPTURE_FEEDBACK_MPC_PLANNING_SCALE ?= 1.30
CAPTURE_ILQR_STATE ?= 674
CAPTURE_ILQR_SEED ?= 62601
CAPTURE_ILQR_OUT ?= runs/p1_capture_ilqr/validation_$(CAPTURE_ILQR_STATE)_p1.json
CAPTURE_ILQR_HORIZON ?= 3.0
CAPTURE_ILQR_ITERATIONS ?= 80
CAPTURE_ILQR_TERMINAL_WEIGHT ?= 1.0
CAPTURE_ILQR_TERMINAL_STATE_WEIGHT ?= 0.0
CAPTURE_ILQR_INITIAL_ARGS ?= --initial-lqr-scale 0
CAPTURE_ILQR_CHAIN_SOURCE ?= runs/p1_capture_ilqr/validation_674_p1_boxddp_from_cem_refine100x.json
CAPTURE_ILQR_CHAIN_OUT ?= runs/p1_capture_ilqr/validation_$(CAPTURE_ILQR_STATE)_p1_chain.json
CAPTURE_ILQR_CHAIN_TAIL_SECONDS ?= 1.5
CAPTURE_ILQR_CHAIN_ITERATIONS ?= 400
CAPTURE_ILQR_CHAIN_INITIAL_LQR_SCALE ?= 0.0
CAPTURE_ILQR_CHAIN_INITIAL_ARGS ?= --initial-action-cem --action-knots 24 --cem-iterations 20 --cem-population 512 --cem-elites 40
CAPTURE_ILQR_CHAIN_BASIN_OUT ?= runs/p1_capture_ilqr/validation_674_p1_chain_basin.json
CAPTURE_FDDP_INITIAL ?= $(CAPTURE_ILQR_CHAIN_OUT)
CAPTURE_FDDP_OUT ?= runs/p1_capture_fddp/validation_$(CAPTURE_ILQR_STATE).json
CAPTURE_FDDP_ITERATIONS ?= 100
CAPTURE_FDDP_SOURCE_STATE ?= 674
CAPTURE_FDDP_INITIAL_ALPHA ?= 0.0
CAPTURE_FDDP_TARGET_ALPHA ?= 1.0
CAPTURE_FDDP_CONTINUATION_OUT ?= runs/p1_capture_fddp/continuation_$(CAPTURE_FDDP_SOURCE_STATE)_to_$(CAPTURE_ILQR_STATE).json
CAPTURE_RECENTER_CONTROLLER ?= $(CAPTURE_FDDP_OUT)
CAPTURE_RECENTER_STATE ?= $(CAPTURE_ILQR_STATE)
CAPTURE_RECENTER_GAIN_SCALE ?= 1.0
CAPTURE_RECENTER_OUT ?= runs/p1_capture_fddp/recentered_validation_$(CAPTURE_RECENTER_STATE).json
CAPTURE_FUNNEL_LIBRARY_ARGS ?= --controller $(CAPTURE_RECENTER_OUT) --gain-scale 1.0
CAPTURE_FUNNEL_LIBRARY_LIMIT ?= 256
CAPTURE_FUNNEL_LIBRARY_OUT ?= runs/p1_capture_fddp/funnel_library_validation$(CAPTURE_FUNNEL_LIBRARY_LIMIT).json
CAPTURE_MULTIPLE_SHOOTING_OUT ?= runs/p1_capture_multiple_shooting/validation_$(CAPTURE_ILQR_STATE)_p1.json
CAPTURE_MULTIPLE_SHOOTING_SEGMENT_STEPS ?= 1
CAPTURE_MULTIPLE_SHOOTING_ITERATIONS ?= 30
CAPTURE_MULTIPLE_SHOOTING_INITIAL_ARGS ?=
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

setup-sb3:
	$(PYTHON) -m pip install -r requirements-sb3.txt

setup-fddp:
	$(PYTHON) -m pip install -r requirements-fddp.txt

roadmap-p0: test-roadmap-p0 export-swingup7-xml verify-roadmap-p0

test-roadmap-p0:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

verify-roadmap-p0:
	$(PYTHON) scripts/verify_roadmap.py --benchmark-only --report runs/roadmap_p0/benchmark_verification.json

export-swingup7-xml:
	$(PYTHON) scripts/export_xml.py --config configs/swingup7_uniform.yaml --progress 1.0 --out runs/swingup7_uniform/model.xml

generate-capture-envelope:
	$(PYTHON) scripts/generate_capture_envelope.py --spec benchmarks/p1_capture_envelope.yaml --out-dir $(CAPTURE_ENVELOPE_DIR)

capture-envelope6: generate-capture-envelope
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_capture_envelope.yaml

capture-hybrid-policy6: generate-capture-envelope
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_capture_hybrid_policy.yaml

capture-boundary-lqr: generate-capture-envelope
	$(PYTHON) scripts/make_lqr_checkpoint.py --config configs/swingup6_capture_envelope.yaml --checkpoint $(CAPTURE_BOUNDARY_LQR) --progress $(CAPTURE_BOUNDARY_PROGRESS) --control-cost 1000 --policy-scale 1.30 --override 'ppo.hidden_sizes=[]' --override 'env.action_lqr_residual.enabled=false'

mine-capture-boundary: capture-boundary-lqr
	$(PYTHON) scripts/mine_capture_failures.py --config configs/swingup6_capture_envelope.yaml --checkpoint $(CAPTURE_BOUNDARY_LQR) --dataset $(CAPTURE_ENVELOPE_DIR)/train.json --progress $(CAPTURE_BOUNDARY_PROGRESS) --sample-count 1024 --hard-repeat 5 --batch-size 64 --seed 61621 --out $(CAPTURE_BOUNDARY_ROLLOUTS) --override 'ppo.hidden_sizes=[]' --override 'env.action_lqr_residual.enabled=false'

fit-capture-funnel: mine-capture-boundary
	$(PYTHON) scripts/fit_capture_funnel.py --config configs/swingup6_capture_envelope.yaml --dataset $(CAPTURE_ENVELOPE_DIR)/train.json --rollouts $(CAPTURE_BOUNDARY_ROLLOUTS) --progress $(CAPTURE_BOUNDARY_PROGRESS) --out $(CAPTURE_FUNNEL_MODEL)

search-capture-target-teacher: generate-capture-envelope
	$(PYTHON) scripts/search_capture_target_schedule.py --config configs/swingup6_capture_envelope.yaml --state-json $(CAPTURE_ENVELOPE_DIR)/validation.json --state-index $(CAPTURE_TARGET_TEACHER_STATE) --progress $(CAPTURE_BOUNDARY_PROGRESS) --out $(CAPTURE_TARGET_TEACHER_OUT)

build-capture-target-teachers: mine-capture-boundary
	$(PYTHON) scripts/build_capture_target_teachers.py --config configs/swingup6_capture_envelope.yaml --dataset $(CAPTURE_ENVELOPE_DIR)/train.json --rollouts $(CAPTURE_BOUNDARY_ROLLOUTS) --progress $(CAPTURE_BOUNDARY_PROGRESS) --out $(CAPTURE_TARGET_TEACHERS_OUT)

build-feedback-mpc-teachers:
	$(PYTHON) scripts/build_feedback_mpc_teachers.py --out $(CAPTURE_FEEDBACK_TEACHERS_OUT)

build-capture-supervisor-dataset:
	$(PYTHON) scripts/build_capture_supervisor_dataset.py --feedback-teachers $(CAPTURE_FEEDBACK_TEACHERS_OUT) --out $(CAPTURE_SUPERVISOR_LABELS)

distill-capture-supervisor:
	$(PYTHON) scripts/distill_capture_supervisor.py --labels $(CAPTURE_SUPERVISOR_LABELS) --out-dir $(CAPTURE_SUPERVISOR_RUN)

eval-capture-supervisor:
	$(PYTHON) scripts/evaluate_capture_gate.py --config configs/swingup6_capture_hybrid_policy.yaml --checkpoint $(CAPTURE_SUPERVISOR_CHECKPOINT) --dataset $(CAPTURE_ENVELOPE_DIR)/validation.json --split validation --limit 256 --progress 0.07 --out $(CAPTURE_SUPERVISOR_RUN)/eval_validation256_p007.json

collect-capture-dagger:
	$(PYTHON) scripts/collect_capture_dagger_states.py --checkpoint $(CAPTURE_DAGGER_CHECKPOINT) --out $(CAPTURE_DAGGER_QUERIES)

label-capture-dagger-lqr:
	$(PYTHON) scripts/label_capture_dagger_lqr.py --queries $(CAPTURE_DAGGER_QUERIES) --out $(CAPTURE_DAGGER_LQR_LABELS)

label-capture-dagger-target:
	$(PYTHON) scripts/label_capture_dagger_target.py

label-capture-dagger-feedback:
	$(PYTHON) scripts/label_capture_dagger_feedback_mpc.py

eval-adaptive-capture-boundary: generate-capture-envelope
	$(PYTHON) scripts/evaluate_adaptive_target_capture.py --config configs/swingup6_capture_envelope.yaml --dataset $(CAPTURE_ENVELOPE_DIR)/validation.json --split validation --progress $(CAPTURE_ADAPTIVE_PROGRESS) --limit $(CAPTURE_ADAPTIVE_LIMIT) --index-seed 61201 --iterations 6 --population 128 --elites 12 --out $(CAPTURE_ADAPTIVE_OUT)

analyze-capture-modal:
	$(PYTHON) scripts/analyze_capture_modal_failures.py --evaluation $(CAPTURE_ADAPTIVE_OUT) --out $(CAPTURE_MODAL_ANALYSIS_OUT)

sweep-capture-modal-feedback:
	$(PYTHON) scripts/sweep_capture_modal_feedback.py --evaluation $(CAPTURE_ADAPTIVE_OUT) --analysis $(CAPTURE_MODAL_ANALYSIS_OUT) --multipliers 0.90,0.95,0.975,0.99,1.0,1.01,1.025,1.05,1.10 --out $(CAPTURE_MODAL_SWEEP_OUT)

refine-adaptive-capture-boundary:
	$(PYTHON) scripts/refine_adaptive_target_capture.py --initial-evaluation $(CAPTURE_ADAPTIVE_OUT) --out $(CAPTURE_REFINED_ADAPTIVE_OUT)

eval-transfer-capture-boundary:
	$(PYTHON) scripts/refine_adaptive_target_capture.py --initial-evaluation $(CAPTURE_REFINED_ADAPTIVE_OUT) --progress $(CAPTURE_TRANSFER_PROGRESS) --replay-only --out $(CAPTURE_TRANSFER_OUT)

search-capture-hybrid:
	$(PYTHON) scripts/search_capture_hybrid_schedule.py --state-index $(CAPTURE_HYBRID_STATE) --initial-evaluation $(CAPTURE_REFINED_ADAPTIVE_OUT) --seed $(CAPTURE_HYBRID_SEED) --out $(CAPTURE_HYBRID_OUT)

eval-feedback-mpc-state:
	$(PYTHON) scripts/evaluate_feedback_mpc_capture.py --state-index $(CAPTURE_FEEDBACK_MPC_STATE) --seed $(CAPTURE_FEEDBACK_MPC_SEED) --mpc-seconds $(CAPTURE_FEEDBACK_MPC_SECONDS) --handoff-cart-abs $(CAPTURE_FEEDBACK_MPC_HANDOFF_CART) --iterations $(CAPTURE_FEEDBACK_MPC_ITERATIONS) --population $(CAPTURE_FEEDBACK_MPC_POPULATION) --elites $(CAPTURE_FEEDBACK_MPC_ELITES) --planning-lqr-scale $(CAPTURE_FEEDBACK_MPC_PLANNING_SCALE) --out $(CAPTURE_FEEDBACK_MPC_OUT)

eval-predictive-sampling-state:
	$(PYTHON) scripts/evaluate_predictive_sampling_capture.py --state-index $(CAPTURE_PREDICTIVE_STATE) --progress $(CAPTURE_PREDICTIVE_PROGRESS) --out $(CAPTURE_PREDICTIVE_OUT)

refine-feedback-mpc-boundary:
	$(PYTHON) scripts/refine_feedback_mpc_capture.py --initial-evaluation $(CAPTURE_FEEDBACK_MPC_INITIAL) --mpc-seconds $(CAPTURE_FEEDBACK_MPC_SECONDS) --handoff-cart-abs $(CAPTURE_FEEDBACK_MPC_HANDOFF_CART) --iterations $(CAPTURE_FEEDBACK_MPC_ITERATIONS) --population $(CAPTURE_FEEDBACK_MPC_POPULATION) --elites $(CAPTURE_FEEDBACK_MPC_ELITES) --planning-lqr-scale $(CAPTURE_FEEDBACK_MPC_PLANNING_SCALE) --out $(CAPTURE_FEEDBACK_MPC_BATCH_OUT)

search-ilqr-capture:
	$(PYTHON) scripts/search_ilqr_capture.py --state-index $(CAPTURE_ILQR_STATE) --seed $(CAPTURE_ILQR_SEED) --horizon-seconds $(CAPTURE_ILQR_HORIZON) --iterations $(CAPTURE_ILQR_ITERATIONS) --terminal-weight $(CAPTURE_ILQR_TERMINAL_WEIGHT) --terminal-state-weight $(CAPTURE_ILQR_TERMINAL_STATE_WEIGHT) --rail-weight 100000000 $(CAPTURE_ILQR_INITIAL_ARGS) --out $(CAPTURE_ILQR_OUT)

search-fddp-capture:
	$(PYTHON) scripts/search_fddp_capture.py --state-index $(CAPTURE_ILQR_STATE) --initial-controller $(CAPTURE_FDDP_INITIAL) --seed $(CAPTURE_ILQR_SEED) --iterations $(CAPTURE_FDDP_ITERATIONS) --terminal-weight 100000 --terminal-state-weight 100000 --out $(CAPTURE_FDDP_OUT)

continue-fddp-capture:
	$(PYTHON) scripts/continue_fddp_homotopy.py --state-index $(CAPTURE_ILQR_STATE) --source-state-index $(CAPTURE_FDDP_SOURCE_STATE) --initial-alpha $(CAPTURE_FDDP_INITIAL_ALPHA) --target-alpha $(CAPTURE_FDDP_TARGET_ALPHA) --initial-controller $(CAPTURE_FDDP_INITIAL) --iterations $(CAPTURE_FDDP_ITERATIONS) --summary-out $(CAPTURE_FDDP_CONTINUATION_OUT)

refine-ilqr-capture-chain:
	$(PYTHON) scripts/refine_ilqr_capture_chain.py --source-controller $(CAPTURE_ILQR_CHAIN_SOURCE) --seed $(CAPTURE_ILQR_SEED) --tail-seconds $(CAPTURE_ILQR_CHAIN_TAIL_SECONDS) --iterations $(CAPTURE_ILQR_CHAIN_ITERATIONS) --initial-lqr-scale $(CAPTURE_ILQR_CHAIN_INITIAL_LQR_SCALE) $(CAPTURE_ILQR_CHAIN_INITIAL_ARGS) --out $(CAPTURE_ILQR_CHAIN_OUT)

eval-ilqr-chain-basin:
	$(PYTHON) scripts/evaluate_ilqr_chain_basin.py --controller $(CAPTURE_ILQR_CHAIN_OUT) --out $(CAPTURE_ILQR_CHAIN_BASIN_OUT)

recenter-trajectory-controller:
	$(PYTHON) scripts/recenter_trajectory_controller.py --controller $(CAPTURE_RECENTER_CONTROLLER) --state-index $(CAPTURE_RECENTER_STATE) --feedback-gain-source solver --feedback-gain-scale $(CAPTURE_RECENTER_GAIN_SCALE) --out $(CAPTURE_RECENTER_OUT)

eval-trajectory-funnel-library:
	$(PYTHON) scripts/evaluate_trajectory_funnel_library.py $(CAPTURE_FUNNEL_LIBRARY_ARGS) --limit $(CAPTURE_FUNNEL_LIBRARY_LIMIT) --out $(CAPTURE_FUNNEL_LIBRARY_OUT)

search-multiple-shooting-capture:
	$(PYTHON) scripts/search_multiple_shooting_capture.py --state-index $(CAPTURE_ILQR_STATE) --seed $(CAPTURE_ILQR_SEED) --horizon-seconds $(CAPTURE_ILQR_HORIZON) --segment-steps $(CAPTURE_MULTIPLE_SHOOTING_SEGMENT_STEPS) --max-evaluations $(CAPTURE_MULTIPLE_SHOOTING_ITERATIONS) $(CAPTURE_MULTIPLE_SHOOTING_INITIAL_ARGS) --out $(CAPTURE_MULTIPLE_SHOOTING_OUT)

analyze-refined-capture-modal:
	$(PYTHON) scripts/analyze_capture_modal_failures.py --evaluation $(CAPTURE_REFINED_ADAPTIVE_OUT) --out $(CAPTURE_REFINED_MODAL_ANALYSIS_OUT)

capture-sac-boundary: mine-capture-boundary
	$(PYTHON) scripts/train_sb3_sac.py --config configs/swingup6_capture_sac_boundary.yaml

mine-capture-failures: generate-capture-envelope
	$(PYTHON) scripts/mine_capture_failures.py --config configs/swingup6_capture_envelope.yaml --checkpoint $(CAPTURE_ENVELOPE_CHECKPOINT) --dataset $(CAPTURE_ENVELOPE_DIR)/train.json --progress 0.055 --out $(CAPTURE_HARD_STATES)

distill-capture-lqr: generate-capture-envelope
	$(PYTHON) scripts/distill_capture_lqr.py --config configs/swingup6_capture_envelope.yaml --dataset $(CAPTURE_ENVELOPE_DIR)/train.json --out-dir runs/swingup6_capture_lqr_distilled

eval-capture-envelope6: generate-capture-envelope
	$(PYTHON) scripts/evaluate_capture_gate.py --config configs/swingup6_capture_envelope.yaml --checkpoint $(CAPTURE_ENVELOPE_CHECKPOINT) --dataset $(CAPTURE_ENVELOPE_DIR)/test.json --out $(CAPTURE_ENVELOPE_RUN)/eval_capture_envelope_test1000.json

eval-capture-envelope-lqr: generate-capture-envelope
	$(PYTHON) scripts/evaluate_capture_gate.py --config configs/swingup6_capture_envelope.yaml --checkpoint runs/uniform6_finetune/checkpoints/best.safetensors --dataset $(CAPTURE_ENVELOPE_DIR)/test.json --override 'ppo.hidden_sizes=[]' --out $(CAPTURE_ENVELOPE_DIR)/eval_lqr_test1000.json

eval-capture-envelope-mpc: generate-capture-envelope
	$(PYTHON) scripts/evaluate_linear_mpc_capture.py --config configs/swingup6_capture_envelope.yaml --dataset $(CAPTURE_ENVELOPE_DIR)/validation.json --progress 0.06 --limit 64 --horizon 20 --out $(CAPTURE_ENVELOPE_DIR)/eval_linear_mpc_progress006_validation64.json

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

search-swingup-funnel: fit-capture-funnel
	$(PYTHON) scripts/search_swingup_chain.py --config configs/swingup6_uniform.yaml --progress 1.0 --zero-noise --capture-funnel-model $(CAPTURE_FUNNEL_MODEL) --lqr-capture-scale 1.30 --out runs/swingup6_funnel_search/search.json

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
