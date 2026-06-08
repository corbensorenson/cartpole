.PHONY: setup smoke debug sweep train6 eval6 render6 uniform6 lqr6 swingup-debug swingup6 eval-swingup6 render-swingup6

setup:
	bash scripts/setup_mac.sh

smoke:
	python scripts/smoke_test.py

debug:
	python scripts/train_mlx_ppo.py --config configs/debug6_fast.yaml

sweep:
	python scripts/linear_sweep.py --config configs/sweep_n.yaml

train6:
	python scripts/train_mlx_ppo.py --config configs/gradient6_curriculum.yaml

uniform6:
	python scripts/train_mlx_ppo.py --config configs/uniform6_finetune.yaml --init-checkpoint runs/gradient6_curriculum/checkpoints/best.safetensors

lqr6:
	python scripts/make_lqr_checkpoint.py --config configs/uniform6_near_upright_lqr.yaml

eval6:
	python scripts/evaluate.py --config configs/uniform6_near_upright_lqr.yaml --checkpoint runs/uniform6_finetune/checkpoints/best.safetensors --episodes 20 --progress 1.0 --out runs/uniform6_finetune/eval_uniform6.json

render6:
	python scripts/render_video.py --config configs/uniform6_near_upright_lqr.yaml --checkpoint runs/uniform6_finetune/checkpoints/best.safetensors --out runs/uniform6_finetune/six_link_uniform_success.mp4 --seconds 30 --progress 1.0 --metadata-out runs/uniform6_finetune/six_link_uniform_success.video.json --fail-on-reset

swingup-debug:
	python scripts/train_mlx_ppo.py --config configs/swingup6_debug.yaml

swingup6:
	python scripts/train_mlx_ppo.py --config configs/swingup6_uniform.yaml

eval-swingup6:
	python scripts/evaluate.py --config configs/swingup6_uniform.yaml --checkpoint runs/swingup6_uniform/checkpoints/best.safetensors --episodes 20 --progress 1.0 --out runs/swingup6_uniform/eval_swingup6.json

render-swingup6:
	python scripts/render_video.py --config configs/swingup6_uniform.yaml --checkpoint runs/swingup6_uniform/checkpoints/best.safetensors --out runs/swingup6_uniform/six_link_swingup_success.mp4 --seconds 30 --progress 1.0 --metadata-out runs/swingup6_uniform/six_link_swingup_success.video.json --fail-on-reset
