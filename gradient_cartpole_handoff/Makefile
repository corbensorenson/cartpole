.PHONY: setup smoke debug sweep train6 eval6 render6 uniform6 lqr6 swingup-debug swingup6 swingup6-gradient-low-momentum swingup6-uniform-low-momentum eval-swingup6 eval-swingup6-low-momentum render-swingup6 probe-swingup-trajectory search-swingup-trajectory search-swingup-capture search-swingup-chain export-swingup-states export-chain-states eval-expert-chain eval-expert-chain-learned capture-handoff6 eval-capture-handoff6 capture-state-list6 eval-capture-state-list6 capture-state-list-shaped6 eval-capture-state-list-shaped6 chain-state-list-shaped6 eval-chain-state-list-shaped6

PYTHON ?= .venv/bin/python

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
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_uniform_low_momentum_finetune.yaml --init-checkpoint runs/swingup6_gradient_low_momentum/checkpoints/best.safetensors

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

chain-state-list-shaped6:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_chain_state_list_shaped.yaml

eval-chain-state-list-shaped6:
	$(PYTHON) scripts/evaluate.py --config configs/swingup6_chain_state_list_shaped.yaml --checkpoint runs/swingup6_chain_state_list_shaped/checkpoints/best.safetensors --episodes 20 --progress 1.0 --out runs/swingup6_chain_state_list_shaped/eval_chain_state_list_shaped6.json
