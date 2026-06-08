from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
except Exception as exc:  # pragma: no cover - checked on target machine
    mx = None
    nn = None
    optim = None
    tree_flatten = None
    tree_map = None
    tree_unflatten = None
    _MLX_IMPORT_ERROR = exc
else:
    _MLX_IMPORT_ERROR = None

from .config import dump_json, save_config
from .vecenv import make_vec_env

LOG_2PI = float(math.log(2.0 * math.pi))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int], action_std_init: float = 0.7):
        super().__init__()
        sizes = [obs_dim] + list(hidden_sizes)
        self.actor_layers = [nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])]
        self.actor_out = nn.Linear(sizes[-1], act_dim)
        self.critic_layers = [nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])]
        self.critic_out = nn.Linear(sizes[-1], 1)
        self.log_std = mx.full((act_dim,), float(math.log(action_std_init)))

    def __call__(self, obs):
        x = obs
        for layer in self.actor_layers:
            x = mx.tanh(layer(x))
        mean = mx.tanh(self.actor_out(x))

        v = obs
        for layer in self.critic_layers:
            v = mx.tanh(layer(v))
        value = mx.squeeze(self.critic_out(v), axis=-1)
        return mean, value


def require_mlx() -> None:
    if mx is None:
        raise RuntimeError(f"Could not import MLX. On Apple Silicon use: pip install mlx. Error: {_MLX_IMPORT_ERROR}")


def clipped_log_std(model: ActorCritic):
    return mx.clip(model.log_std, -5.0, 2.0)


def gaussian_log_prob(action, mean, log_std):
    return -0.5 * mx.sum(((action - mean) / mx.exp(log_std)) ** 2 + 2.0 * log_std + LOG_2PI, axis=-1)


def gaussian_entropy(log_std):
    return mx.sum(0.5 + 0.5 * LOG_2PI + log_std)


def sample_action(model: ActorCritic, obs_np: np.ndarray, deterministic: bool = False):
    obs = mx.array(obs_np.astype(np.float32))
    mean, value = model(obs)
    log_std = clipped_log_std(model)
    if deterministic:
        action = mean
    else:
        eps = mx.random.normal(mean.shape)
        action = mx.clip(mean + mx.exp(log_std) * eps, -1.0, 1.0)
    logp = gaussian_log_prob(action, mean, log_std)
    mx.eval(action, logp, value)
    return np.asarray(action, dtype=np.float32), np.asarray(logp, dtype=np.float32), np.asarray(value, dtype=np.float32)


def value_only(model: ActorCritic, obs_np: np.ndarray) -> np.ndarray:
    obs = mx.array(obs_np.astype(np.float32))
    _, value = model(obs)
    mx.eval(value)
    return np.asarray(value, dtype=np.float32)


def clip_grads(grads, max_norm: float):
    if max_norm is None or max_norm <= 0:
        return grads
    leaves = [v for _, v in tree_flatten(grads) if v is not None]
    if not leaves:
        return grads
    total = mx.sqrt(sum(mx.sum(g * g) for g in leaves))
    scale = mx.minimum(1.0, float(max_norm) / (total + 1e-8))
    return tree_map(lambda g: g * scale if g is not None else None, grads)


def ppo_loss_fn(
    model: ActorCritic,
    obs,
    actions,
    old_logp,
    advantages,
    returns,
    old_values,
    clip_coef: float,
    entropy_coef: float,
    value_coef: float,
):
    mean, values = model(obs)
    log_std = clipped_log_std(model)
    logp = gaussian_log_prob(actions, mean, log_std)
    entropy = gaussian_entropy(log_std)

    logratio = logp - old_logp
    ratio = mx.exp(logratio)
    pg_loss_unclipped = -advantages * ratio
    pg_loss_clipped = -advantages * mx.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    policy_loss = mx.mean(mx.maximum(pg_loss_unclipped, pg_loss_clipped))

    v_clipped = old_values + mx.clip(values - old_values, -clip_coef, clip_coef)
    v_loss_unclipped = (values - returns) ** 2
    v_loss_clipped = (v_clipped - returns) ** 2
    value_loss = 0.5 * mx.mean(mx.maximum(v_loss_unclipped, v_loss_clipped))

    approx_kl = mx.mean((ratio - 1.0) - logratio)
    clip_fraction = mx.mean(mx.abs(ratio - 1.0) > clip_coef)
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total, (policy_loss, value_loss, entropy, approx_kl, clip_fraction)


def save_model(model: ActorCritic, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = tree_flatten(model.parameters(), destination={})
    mx.save_safetensors(str(path), flat)


def load_model(model: ActorCritic, path: str | Path) -> None:
    raw = mx.load(str(path))
    try:
        params = tree_unflatten(raw)
    except Exception:
        params = tree_unflatten(list(raw.items()))
    model.update(params)
    mx.eval(model.parameters())


def evaluate_policy(
    cfg: dict[str, Any],
    model: ActorCritic,
    episodes: int,
    seed: int,
    progress: float = 1.0,
    return_episodes: bool = False,
) -> dict[str, Any]:
    from .env import NLinkCartPoleEnv

    returns = []
    lengths = []
    successes = []
    max_angles = []
    episode_results = []
    for ep in range(episodes):
        env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed + 10_000 + ep)
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        ep_max_angle = 0.0
        while not done:
            action, _, _ = sample_action(model, obs[None, :], deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            ep_return += float(reward)
            ep_len += 1
            ep_max_angle = max(ep_max_angle, float(info.get("max_abs_angle", 0.0)))
            done = bool(terminated or truncated)
        returns.append(ep_return)
        lengths.append(ep_len)
        success = bool(info.get("success", False))
        successes.append(float(success))
        max_angles.append(ep_max_angle)
        if return_episodes:
            episode_results.append(
                {
                    "episode": ep,
                    "seed": seed + 10_000 + ep,
                    "return": float(ep_return),
                    "length": int(ep_len),
                    "success": success,
                    "max_abs_angle": float(ep_max_angle),
                    "terminated": not success and ep_len < env.max_steps,
                    "final_x": float(info.get("x", np.nan)),
                }
            )
        env.close()
    metrics: dict[str, Any] = {
        "episodes": int(episodes),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
        "length_min": int(np.min(lengths)),
        "length_max": int(np.max(lengths)),
        "success_rate": float(np.mean(successes)),
        "max_angle_mean": float(np.mean(max_angles)),
        "max_angle_max": float(np.max(max_angles)),
    }
    if return_episodes:
        metrics["episode_results"] = episode_results
    return metrics


def train(cfg: dict[str, Any], init_checkpoint: str | None = None) -> dict[str, Any]:
    require_mlx()
    seed = int(cfg.get("experiment", {}).get("seed", 0))
    np.random.seed(seed)
    mx.random.seed(seed)

    out_dir = Path(cfg["experiment"]["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.resolved.yaml")

    ppo = cfg["ppo"]
    num_envs = int(ppo["num_envs"])
    rollout_steps = int(ppo["rollout_steps"])
    total_updates = int(ppo["total_updates"])
    vec_backend = str(ppo.get("vec_backend", "serial"))
    rebuild_every = int(cfg["env"].get("curriculum_rebuild_every", 25))

    envs = make_vec_env(cfg, num_envs=num_envs, seed=seed, progress=0.0, backend=vec_backend)
    obs, _ = envs.reset()
    obs_dim = obs.shape[1]
    act_dim = envs.single_action_space.shape[0]

    model = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=list(ppo.get("hidden_sizes", [256, 256])),
        action_std_init=float(ppo.get("action_std_init", 0.7)),
    )
    mx.eval(model.parameters())
    if init_checkpoint:
        load_model(model, init_checkpoint)

    optimizer = optim.Adam(learning_rate=float(ppo["learning_rate"]))
    loss_and_grad = nn.value_and_grad(model, ppo_loss_fn)

    log_path = out_dir / "train_log.csv"
    csv_file = open(log_path, "w", newline="", encoding="utf-8")
    fieldnames = [
        "update", "global_steps", "progress", "steps_per_sec", "mean_ep_return", "mean_ep_len",
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
        "eval_return_mean", "eval_success_rate", "eval_length_mean", "alpha_length", "alpha_mass", "alpha_damping",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    ep_returns = np.zeros(num_envs, dtype=np.float64)
    ep_lengths = np.zeros(num_envs, dtype=np.int64)
    recent_returns: list[float] = []
    recent_lengths: list[int] = []
    best_eval_return = -float("inf")
    global_steps = 0
    start_time = time.time()
    last_loss_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}

    try:
        for update in range(1, total_updates + 1):
            progress = (update - 1) / max(1, total_updates - 1)
            if update == 1 or (rebuild_every > 0 and update % rebuild_every == 0):
                obs, infos = envs.set_progress(progress)
                morph_info = infos[0]
            else:
                morph_info = {}

            obs_buf = np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32)
            action_buf = np.zeros((rollout_steps, num_envs, act_dim), dtype=np.float32)
            logp_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
            reward_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
            done_buf = np.zeros((rollout_steps, num_envs), dtype=bool)
            value_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)

            rollout_start = time.time()
            for t in range(rollout_steps):
                obs_buf[t] = obs
                actions, logp, values = sample_action(model, obs, deterministic=False)
                action_buf[t] = actions
                logp_buf[t] = logp
                value_buf[t] = values

                next_obs, rewards, dones, infos = envs.step(actions)
                reward_buf[t] = rewards
                done_buf[t] = dones
                global_steps += num_envs

                ep_returns += rewards
                ep_lengths += 1
                for i, done in enumerate(dones):
                    if done:
                        recent_returns.append(float(ep_returns[i]))
                        recent_lengths.append(int(ep_lengths[i]))
                        recent_returns = recent_returns[-100:]
                        recent_lengths = recent_lengths[-100:]
                        ep_returns[i] = 0.0
                        ep_lengths[i] = 0
                obs = next_obs

            last_values = value_only(model, obs)
            advantages = np.zeros_like(reward_buf, dtype=np.float32)
            lastgaelam = np.zeros(num_envs, dtype=np.float32)
            gamma = float(ppo["gamma"])
            lam = float(ppo["gae_lambda"])
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
                    next_values = last_values
                else:
                    next_values = value_buf[t + 1]
                next_nonterminal = 1.0 - done_buf[t].astype(np.float32)
                delta = reward_buf[t] + gamma * next_values * next_nonterminal - value_buf[t]
                lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + value_buf

            b_obs = obs_buf.reshape((-1, obs_dim))
            b_actions = action_buf.reshape((-1, act_dim))
            b_logp = logp_buf.reshape(-1)
            b_adv = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = value_buf.reshape(-1)
            if bool(ppo.get("normalize_advantages", True)):
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            batch_size = b_obs.shape[0]
            minibatches = int(ppo["minibatches"])
            mb_size = max(1, batch_size // minibatches)
            inds = np.arange(batch_size)
            loss_sums = np.zeros(5, dtype=np.float64)
            loss_count = 0

            for _epoch in range(int(ppo["epochs"])):
                np.random.shuffle(inds)
                for start in range(0, batch_size, mb_size):
                    mb_inds = inds[start : start + mb_size]
                    loss_out, grads = loss_and_grad(
                        model,
                        mx.array(b_obs[mb_inds]),
                        mx.array(b_actions[mb_inds]),
                        mx.array(b_logp[mb_inds]),
                        mx.array(b_adv[mb_inds]),
                        mx.array(b_returns[mb_inds]),
                        mx.array(b_values[mb_inds]),
                        float(ppo["clip_coef"]),
                        float(ppo["entropy_coef"]),
                        float(ppo["value_coef"]),
                    )
                    grads = clip_grads(grads, float(ppo.get("max_grad_norm", 0.0)))
                    optimizer.update(model, grads)
                    mx.eval(model.parameters(), optimizer.state)
                    loss_value, aux = loss_out
                    mx.eval(loss_value, *aux)
                    loss_sums += np.asarray([float(x) for x in aux], dtype=np.float64)
                    loss_count += 1

            if loss_count:
                means = loss_sums / loss_count
                last_loss_metrics = {
                    "policy_loss": float(means[0]),
                    "value_loss": float(means[1]),
                    "entropy": float(means[2]),
                    "approx_kl": float(means[3]),
                    "clip_fraction": float(means[4]),
                }

            steps_per_sec = (rollout_steps * num_envs) / max(1e-9, time.time() - rollout_start)
            mean_ep_return = float(np.mean(recent_returns[-20:])) if recent_returns else 0.0
            mean_ep_len = float(np.mean(recent_lengths[-20:])) if recent_lengths else 0.0
            eval_metrics = {"return_mean": np.nan, "success_rate": np.nan, "length_mean": np.nan}

            if update % int(ppo.get("eval_every", 50)) == 0 or update == total_updates:
                eval_metrics = evaluate_policy(
                    cfg,
                    model,
                    episodes=int(ppo.get("eval_episodes", 5)),
                    seed=seed + update * 17,
                    progress=1.0,
                )
                if eval_metrics["return_mean"] > best_eval_return:
                    best_eval_return = eval_metrics["return_mean"]
                    save_model(model, ckpt_dir / "best.safetensors")
                    dump_json({"update": update, "eval": eval_metrics, "global_steps": global_steps}, ckpt_dir / "best.meta.json")

            if update % int(ppo.get("checkpoint_every", 50)) == 0 or update == total_updates:
                save_model(model, ckpt_dir / f"update_{update:06d}.safetensors")
                save_model(model, ckpt_dir / "latest.safetensors")
                dump_json({"update": update, "global_steps": global_steps, "elapsed_seconds": time.time() - start_time}, ckpt_dir / "latest.meta.json")

            row = {
                "update": update,
                "global_steps": global_steps,
                "progress": progress,
                "steps_per_sec": steps_per_sec,
                "mean_ep_return": mean_ep_return,
                "mean_ep_len": mean_ep_len,
                **last_loss_metrics,
                "eval_return_mean": eval_metrics.get("return_mean", np.nan),
                "eval_success_rate": eval_metrics.get("success_rate", np.nan),
                "eval_length_mean": eval_metrics.get("length_mean", np.nan),
                "alpha_length": morph_info.get("alpha_length", np.nan),
                "alpha_mass": morph_info.get("alpha_mass", np.nan),
                "alpha_damping": morph_info.get("alpha_damping", np.nan),
            }
            writer.writerow(row)
            csv_file.flush()

            print(
                f"upd={update:05d}/{total_updates} steps={global_steps:,} "
                f"prog={progress:.3f} sps={steps_per_sec:,.0f} "
                f"ep_ret={mean_ep_return:.1f} ep_len={mean_ep_len:.1f} "
                f"eval_ret={eval_metrics.get('return_mean', float('nan')):.1f} "
                f"succ={eval_metrics.get('success_rate', float('nan')):.2f}"
            )

    finally:
        csv_file.close()
        envs.close()

    return {"out_dir": str(out_dir), "best_eval_return": best_eval_return, "global_steps": global_steps}
