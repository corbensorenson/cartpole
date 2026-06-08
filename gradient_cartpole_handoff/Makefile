.PHONY: setup smoke debug sweep train6 eval6 render6 uniform6 lqr6 swingup-debug swingup6 eval-swingup6 render-swingup6 probe-swingup-trajectory search-swingup-trajectory search-swingup-capture capture-handoff6 eval-capture-handoff6

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

capture-handoff6:
	$(PYTHON) scripts/train_mlx_ppo.py --config configs/swingup6_capture_handoff.yaml

eval-swingup6:
	$(PYTHON) scripts/evaluate.py --config configs/swingup6_uniform.yaml --checkpoint runs/swingup6_uniform/checkpoints/best.safetensors --episodes 20 --progress 1.0 --out runs/swingup6_uniform/eval_swingup6.json

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
