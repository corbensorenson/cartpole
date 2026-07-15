from __future__ import annotations

import copy
import csv
import json
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
    reset_state_indices: list[int] | None = None,
) -> dict[str, Any]:
    from .env import NLinkCartPoleEnv

    returns = []
    lengths = []
    successes = []
    max_angles = []
    time_to_uprights = []
    time_to_captures = []
    max_upright_streaks = []
    final_upright_streaks = []
    max_cart_excursions = []
    max_centered_upright_streaks = []
    max_low_momentum_upright_streaks = []
    max_capture_qualities = []
    low_momentum_upright_events = []
    episode_results = []
    reward_cfg = cfg.get("env", {}).get("reward", {})
    low_momentum_hinge_threshold = float(reward_cfg.get("upright_hinge_vel_threshold", 1.0))
    low_momentum_cart_threshold = float(reward_cfg.get("upright_cart_vel_threshold", 1.0))
    low_momentum_min_time = float(reward_cfg.get("low_momentum_min_time_seconds", 0.0))
    low_momentum_max_cart_abs = reward_cfg.get("low_momentum_max_cart_abs")
    if reset_state_indices is not None and len(reset_state_indices) != episodes:
        raise ValueError("reset_state_indices must contain exactly one index per evaluation episode")
    for ep in range(episodes):
        env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed + 10_000 + ep)
        reset_options = None if reset_state_indices is None else {"state_index": int(reset_state_indices[ep])}
        obs, _ = env.reset(options=reset_options)
        done = False
        ep_return = 0.0
        ep_len = 0
        ep_max_angle = 0.0
        ep_max_capture_quality = 0.0
        ep_low_momentum_upright = False
        while not done:
            action, _, _ = sample_action(model, obs[None, :], deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            ep_return += float(reward)
            ep_len += 1
            ep_max_angle = max(ep_max_angle, float(info.get("max_abs_angle", 0.0)))
            ep_max_capture_quality = max(ep_max_capture_quality, float(info.get("capture_quality", 0.0)))
            hinge_rms = float(info.get("hinge_velocity_rms", np.inf))
            cart_vel = abs(float(env.data.qvel[0]))
            time_seconds = float(ep_len * env.dt)
            ep_low_momentum_upright = ep_low_momentum_upright or bool(
                info.get("is_upright", False)
                and time_seconds >= low_momentum_min_time
                and hinge_rms <= low_momentum_hinge_threshold
                and cart_vel <= low_momentum_cart_threshold
                and (
                    low_momentum_max_cart_abs is None
                    or abs(float(env.data.qpos[0])) <= float(low_momentum_max_cart_abs)
                )
            )
            done = bool(terminated or truncated)
        returns.append(ep_return)
        lengths.append(ep_len)
        success = bool(info.get("success", False))
        successes.append(float(success))
        max_angles.append(ep_max_angle)
        max_capture_qualities.append(ep_max_capture_quality)
        low_momentum_upright_events.append(float(ep_low_momentum_upright))
        ttu = info.get("time_to_first_upright")
        ttc = info.get("time_to_capture")
        streak = float(info.get("max_upright_streak_seconds", 0.0))
        final_streak = float(info.get("upright_streak_seconds", 0.0))
        max_cart_excursion = float(info.get("max_cart_excursion", abs(float(info.get("x", 0.0)))))
        centered_streak = float(info.get("max_centered_upright_streak_seconds", 0.0))
        low_momentum_streak = float(info.get("max_low_momentum_upright_streak_seconds", 0.0))
        if ttu is not None:
            time_to_uprights.append(float(ttu))
        if ttc is not None:
            time_to_captures.append(float(ttc))
        max_upright_streaks.append(streak)
        final_upright_streaks.append(final_streak)
        max_cart_excursions.append(max_cart_excursion)
        max_centered_upright_streaks.append(centered_streak)
        max_low_momentum_upright_streaks.append(low_momentum_streak)
        if return_episodes:
            episode_results.append(
                {
                    "episode": ep,
                    "state_index": None if reset_state_indices is None else int(reset_state_indices[ep]),
                    "seed": seed + 10_000 + ep,
                    "return": float(ep_return),
                    "length": int(ep_len),
                    "success": success,
                    "max_abs_angle": float(ep_max_angle),
                    "terminated": info.get("termination_reason") not in {None, "time_limit"},
                    "truncated": info.get("termination_reason") == "time_limit",
                    "termination_reason": info.get("termination_reason"),
                    "final_x": float(info.get("x", np.nan)),
                    "max_cart_excursion": max_cart_excursion,
                    "time_to_first_upright": None if ttu is None else float(ttu),
                    "time_to_capture": None if ttc is None else float(ttc),
                    "capture_start_time": info.get("capture_start_time"),
                    "max_upright_streak_seconds": streak,
                    "final_upright_streak_seconds": final_streak,
                    "max_centered_upright_streak_seconds": centered_streak,
                    "max_low_momentum_upright_streak_seconds": low_momentum_streak,
                    "max_capture_quality": float(ep_max_capture_quality),
                    "low_momentum_upright": bool(ep_low_momentum_upright),
                    "lqr_switch_entry_count": int(info.get("lqr_switch_entry_count", 0)),
                    "lqr_switch_exit_count": int(info.get("lqr_switch_exit_count", 0)),
                    "lqr_switch_first_entry_time": info.get("lqr_switch_first_entry_time"),
                    "lqr_switch_lqr_steps": int(info.get("lqr_switch_lqr_steps", 0)),
                    "lqr_switch_policy_steps": int(info.get("lqr_switch_policy_steps", 0)),
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
        "time_to_first_upright_mean": None if not time_to_uprights else float(np.mean(time_to_uprights)),
        "time_to_first_upright_success_count": int(len(time_to_uprights)),
        "ever_upright_rate": float(len(time_to_uprights) / max(1, episodes)),
        "time_to_capture_mean": None if not time_to_captures else float(np.mean(time_to_captures)),
        "capture_count": int(len(time_to_captures)),
        "max_upright_streak_mean": float(np.mean(max_upright_streaks)),
        "max_upright_streak_min": float(np.min(max_upright_streaks)),
        "max_upright_streak_max": float(np.max(max_upright_streaks)),
        "final_upright_streak_mean": float(np.mean(final_upright_streaks)),
        "final_upright_streak_min": float(np.min(final_upright_streaks)),
        "max_cart_excursion_mean": float(np.mean(max_cart_excursions)),
        "max_cart_excursion_max": float(np.max(max_cart_excursions)),
        "max_centered_upright_streak_mean": float(np.mean(max_centered_upright_streaks)),
        "max_centered_upright_streak_max": float(np.max(max_centered_upright_streaks)),
        "max_low_momentum_upright_streak_mean": float(np.mean(max_low_momentum_upright_streaks)),
        "max_low_momentum_upright_streak_max": float(np.max(max_low_momentum_upright_streaks)),
        "max_capture_quality_mean": float(np.mean(max_capture_qualities)),
        "max_capture_quality_max": float(np.max(max_capture_qualities)),
        "low_momentum_upright_rate": float(np.mean(low_momentum_upright_events)),
    }
    if return_episodes:
        metrics["episode_results"] = episode_results
    return metrics


def select_evaluation_state_indices(path: str | Path, episodes: int, seed: int) -> list[int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    states = payload.get("states", payload) if isinstance(payload, dict) else payload
    if not isinstance(states, list):
        raise ValueError("curriculum evaluation state file must contain a state list")
    if episodes < 1 or episodes > len(states):
        raise ValueError(f"curriculum eval episodes must be in 1..{len(states)}")
    rng = np.random.default_rng(seed)
    return rng.permutation(len(states))[:episodes].astype(int).tolist()


def evaluate_indexed_policy_batched(
    cfg: dict[str, Any],
    model: ActorCritic,
    state_indices: list[int],
    seed: int,
    progress: float,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Evaluate exact state-list entries with batched policy inference."""
    from .env import NLinkCartPoleEnv

    if not state_indices:
        raise ValueError("state_indices must not be empty")
    batch_size = max(1, min(int(batch_size), len(state_indices)))
    reward_cfg = cfg.get("env", {}).get("reward", {})
    hinge_threshold = float(reward_cfg.get("upright_hinge_vel_threshold", 1.0))
    cart_threshold = float(reward_cfg.get("upright_cart_vel_threshold", 1.0))
    min_time = float(reward_cfg.get("low_momentum_min_time_seconds", 0.0))
    max_cart_abs = reward_cfg.get("low_momentum_max_cart_abs")
    records: list[dict[str, Any]] = []

    for batch_start in range(0, len(state_indices), batch_size):
        count = min(batch_size, len(state_indices) - batch_start)
        envs = [NLinkCartPoleEnv(cfg, progress=progress, seed=seed + batch_start + slot) for slot in range(count)]
        observations: dict[int, np.ndarray] = {}
        trackers = [
            {
                "return": 0.0,
                "length": 0,
                "max_abs_angle": 0.0,
                "max_capture_quality": 0.0,
                "low_momentum_upright": False,
            }
            for _ in range(count)
        ]
        active = set(range(count))
        try:
            for slot, env in enumerate(envs):
                observations[slot], _ = env.reset(
                    seed=seed + batch_start + slot,
                    options={"state_index": int(state_indices[batch_start + slot])},
                )
            while active:
                slots = sorted(active)
                observation_batch = np.stack([observations[slot] for slot in slots])
                actions, _, _ = sample_action(model, observation_batch, deterministic=True)
                for action_index, slot in enumerate(slots):
                    env = envs[slot]
                    obs, reward, terminated, truncated, info = env.step(actions[action_index])
                    observations[slot] = obs
                    tracker = trackers[slot]
                    tracker["return"] += float(reward)
                    tracker["length"] += 1
                    tracker["max_abs_angle"] = max(
                        float(tracker["max_abs_angle"]), float(info.get("max_abs_angle", 0.0))
                    )
                    tracker["max_capture_quality"] = max(
                        float(tracker["max_capture_quality"]), float(info.get("capture_quality", 0.0))
                    )
                    tracker["low_momentum_upright"] = bool(tracker["low_momentum_upright"]) or bool(
                        info.get("is_upright", False)
                        and float(tracker["length"]) * env.dt >= min_time
                        and float(info.get("hinge_velocity_rms", np.inf)) <= hinge_threshold
                        and abs(float(env.data.qvel[0])) <= cart_threshold
                        and (max_cart_abs is None or abs(float(env.data.qpos[0])) <= float(max_cart_abs))
                    )
                    if not (terminated or truncated):
                        continue
                    active.remove(slot)
                    episode = batch_start + slot
                    records.append(
                        {
                            "episode": episode,
                            "state_index": int(state_indices[episode]),
                            "seed": seed + episode,
                            "return": float(tracker["return"]),
                            "length": int(tracker["length"]),
                            "success": bool(info.get("success", False)),
                            "max_abs_angle": float(tracker["max_abs_angle"]),
                            "terminated": info.get("termination_reason") not in {None, "time_limit"},
                            "truncated": info.get("termination_reason") == "time_limit",
                            "termination_reason": info.get("termination_reason"),
                            "final_x": float(info.get("x", np.nan)),
                            "max_cart_excursion": float(info.get("max_cart_excursion", 0.0)),
                            "time_to_first_upright": info.get("time_to_first_upright"),
                            "time_to_capture": info.get("time_to_capture"),
                            "capture_start_time": info.get("capture_start_time"),
                            "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
                            "final_upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
                            "max_centered_upright_streak_seconds": float(
                                info.get("max_centered_upright_streak_seconds", 0.0)
                            ),
                            "max_low_momentum_upright_streak_seconds": float(
                                info.get("max_low_momentum_upright_streak_seconds", 0.0)
                            ),
                            "max_capture_quality": float(tracker["max_capture_quality"]),
                            "low_momentum_upright": bool(tracker["low_momentum_upright"]),
                            "lqr_switch_entry_count": int(info.get("lqr_switch_entry_count", 0)),
                            "lqr_switch_exit_count": int(info.get("lqr_switch_exit_count", 0)),
                            "lqr_switch_first_entry_time": info.get("lqr_switch_first_entry_time"),
                            "lqr_switch_lqr_steps": int(info.get("lqr_switch_lqr_steps", 0)),
                            "lqr_switch_policy_steps": int(info.get("lqr_switch_policy_steps", 0)),
                        }
                    )
        finally:
            for env in envs:
                env.close()

    records.sort(key=lambda row: row["episode"])
    returns = np.asarray([row["return"] for row in records], dtype=np.float64)
    lengths = np.asarray([row["length"] for row in records], dtype=np.int64)
    successes = np.asarray([row["success"] for row in records], dtype=np.float64)
    max_angles = np.asarray([row["max_abs_angle"] for row in records], dtype=np.float64)
    upright_streaks = np.asarray([row["max_upright_streak_seconds"] for row in records], dtype=np.float64)
    final_streaks = np.asarray([row["final_upright_streak_seconds"] for row in records], dtype=np.float64)
    cart_excursions = np.asarray([row["max_cart_excursion"] for row in records], dtype=np.float64)
    centered_streaks = np.asarray(
        [row["max_centered_upright_streak_seconds"] for row in records], dtype=np.float64
    )
    low_momentum_streaks = np.asarray(
        [row["max_low_momentum_upright_streak_seconds"] for row in records], dtype=np.float64
    )
    capture_qualities = np.asarray([row["max_capture_quality"] for row in records], dtype=np.float64)
    low_momentum_events = np.asarray([row["low_momentum_upright"] for row in records], dtype=np.float64)
    upright_times = [float(row["time_to_first_upright"]) for row in records if row["time_to_first_upright"] is not None]
    capture_times = [float(row["time_to_capture"]) for row in records if row["time_to_capture"] is not None]
    return {
        "episodes": len(records),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
        "length_min": int(np.min(lengths)),
        "length_max": int(np.max(lengths)),
        "success_rate": float(np.mean(successes)),
        "max_angle_mean": float(np.mean(max_angles)),
        "max_angle_max": float(np.max(max_angles)),
        "time_to_first_upright_mean": None if not upright_times else float(np.mean(upright_times)),
        "time_to_first_upright_success_count": len(upright_times),
        "ever_upright_rate": float(len(upright_times) / len(records)),
        "time_to_capture_mean": None if not capture_times else float(np.mean(capture_times)),
        "capture_count": len(capture_times),
        "max_upright_streak_mean": float(np.mean(upright_streaks)),
        "max_upright_streak_min": float(np.min(upright_streaks)),
        "max_upright_streak_max": float(np.max(upright_streaks)),
        "final_upright_streak_mean": float(np.mean(final_streaks)),
        "final_upright_streak_min": float(np.min(final_streaks)),
        "max_cart_excursion_mean": float(np.mean(cart_excursions)),
        "max_cart_excursion_max": float(np.max(cart_excursions)),
        "max_centered_upright_streak_mean": float(np.mean(centered_streaks)),
        "max_centered_upright_streak_max": float(np.max(centered_streaks)),
        "max_low_momentum_upright_streak_mean": float(np.mean(low_momentum_streaks)),
        "max_low_momentum_upright_streak_max": float(np.max(low_momentum_streaks)),
        "max_capture_quality_mean": float(np.mean(capture_qualities)),
        "max_capture_quality_max": float(np.max(capture_qualities)),
        "low_momentum_upright_rate": float(np.mean(low_momentum_events)),
        "episode_results": records,
    }


def checkpoint_score(eval_metrics: dict[str, Any]) -> tuple[float, ...]:
    """Rank checkpoints by swing-up evidence before shaped return.

    Return alone can be reward-hacked by long survival without ever reaching
    upright, so it is only a tie-breaker after success and capture metrics.
    """
    return (
        float(eval_metrics.get("success_rate", 0.0)),
        float(eval_metrics.get("low_momentum_upright_rate", 0.0)),
        float(eval_metrics.get("max_low_momentum_upright_streak_mean", 0.0)),
        float(eval_metrics.get("ever_upright_rate", 0.0)),
        float(eval_metrics.get("max_centered_upright_streak_mean", 0.0)),
        float(eval_metrics.get("max_upright_streak_mean", 0.0)),
        float(eval_metrics.get("max_upright_streak_max", 0.0)),
        float(eval_metrics.get("max_capture_quality_mean", 0.0)),
        float(eval_metrics.get("return_mean", -float("inf"))),
    )


def resolve_eval_progress(ppo: dict[str, Any], progress: float) -> float:
    """Choose the curriculum progress used for checkpoint evaluation."""
    mode = ppo.get("eval_progress", "final")
    if mode is None:
        value = 1.0
    elif isinstance(mode, str):
        normalized = mode.strip().lower()
        if normalized in {"final", "target"}:
            value = 1.0
        elif normalized in {"current", "curriculum"}:
            value = float(progress)
        else:
            try:
                value = float(normalized)
            except ValueError as exc:
                raise ValueError("ppo.eval_progress must be 'final', 'current', or a numeric progress in [0, 1]") from exc
    else:
        value = float(mode)

    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("ppo.eval_progress resolved outside [0, 1]")
    return value


def curriculum_gate_passed(ppo: dict[str, Any], eval_metrics: dict[str, Any]) -> bool:
    thresholds = {
        "success_rate": float(ppo.get("curriculum_gate_success_rate", 0.0)),
        "ever_upright_rate": float(ppo.get("curriculum_gate_ever_upright_rate", 0.6)),
        "low_momentum_upright_rate": float(ppo.get("curriculum_gate_low_momentum_upright_rate", 0.6)),
        "max_upright_streak_mean": float(ppo.get("curriculum_gate_mean_upright_streak", 0.0)),
        "max_upright_streak_max": float(ppo.get("curriculum_gate_max_upright_streak", 0.20)),
        "max_capture_quality_max": float(ppo.get("curriculum_gate_max_capture_quality", 0.50)),
    }
    return all(float(eval_metrics.get(key, 0.0)) >= threshold for key, threshold in thresholds.items())


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
    curriculum_mode = str(ppo.get("curriculum_mode", "linear")).lower()
    if curriculum_mode not in {"linear", "gated"}:
        raise ValueError("ppo.curriculum_mode must be 'linear' or 'gated'")
    curriculum_progress = float(np.clip(float(ppo.get("curriculum_start_progress", 0.0)), 0.0, 1.0))
    curriculum_step = float(ppo.get("curriculum_step", 0.025))
    curriculum_min_updates = int(ppo.get("curriculum_min_updates_per_stage", ppo.get("eval_every", 50)))
    last_curriculum_advance_update = 0
    last_env_progress: float | None = None
    current_morph_info: dict[str, Any] = {}
    eval_cfg = cfg
    eval_state_indices: list[int] | None = None
    eval_seed = int(ppo.get("curriculum_eval_seed", seed + 91_001))
    eval_states_path = ppo.get("curriculum_eval_states_path")
    if eval_states_path:
        eval_cfg = copy.deepcopy(cfg)
        eval_cfg["env"]["init_mode"] = "state_list"
        eval_cfg["env"]["init_states_path"] = str(eval_states_path)
        eval_cfg["env"]["init_state_curriculum"] = "all"
        eval_state_indices = select_evaluation_state_indices(
            eval_states_path,
            int(ppo.get("eval_episodes", 5)),
            eval_seed,
        )
    curriculum_eval_metadata = {
        "states_path": None if eval_states_path is None else str(eval_states_path),
        "seed": eval_seed,
        "state_indices": eval_state_indices,
    }
    curriculum_eval_batch_size = int(ppo.get("curriculum_eval_batch_size", 64))

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
    elif bool(ppo.get("zero_init_actor_output", False)):
        model.update(
            {
                "actor_out": {
                    "weight": mx.zeros_like(model.actor_out.weight),
                    "bias": mx.zeros_like(model.actor_out.bias),
                }
            }
        )
        mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=float(ppo["learning_rate"]))
    loss_and_grad = nn.value_and_grad(model, ppo_loss_fn)

    log_path = out_dir / "train_log.csv"
    csv_file = open(log_path, "w", newline="", encoding="utf-8")
    fieldnames = [
        "update", "global_steps", "progress", "plant_progress", "eval_progress", "steps_per_sec", "mean_ep_return", "mean_ep_len",
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
        "eval_return_mean", "eval_success_rate", "eval_length_mean", "eval_ever_upright_rate",
        "eval_max_upright_streak_mean", "eval_max_upright_streak_max",
        "eval_max_centered_upright_streak_mean", "eval_max_centered_upright_streak_max",
        "eval_max_low_momentum_upright_streak_mean", "eval_max_low_momentum_upright_streak_max",
        "eval_low_momentum_upright_rate", "eval_max_capture_quality_mean", "eval_max_capture_quality_max",
        "eval_time_to_first_upright_mean", "curriculum_advanced", "rail_limit",
        "alpha_length", "alpha_mass", "alpha_damping", "alpha_frictionloss", "init_qvel_scale",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    ep_returns = np.zeros(num_envs, dtype=np.float64)
    ep_lengths = np.zeros(num_envs, dtype=np.int64)
    recent_returns: list[float] = []
    recent_lengths: list[int] = []
    best_eval_return = -float("inf")
    best_eval_score: tuple[float, ...] | None = None
    global_steps = 0
    start_time = time.time()
    last_loss_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}

    try:
        for update in range(1, total_updates + 1):
            if curriculum_mode == "linear":
                progress = (update - 1) / max(1, total_updates - 1)
            else:
                progress = curriculum_progress

            rebuild_for_progress = last_env_progress is None or not math.isclose(progress, last_env_progress, rel_tol=0.0, abs_tol=1e-12)
            if update == 1 or rebuild_for_progress or (rebuild_every > 0 and update % rebuild_every == 0):
                obs, infos = envs.set_progress(progress)
                ep_returns[:] = 0.0
                ep_lengths[:] = 0
                current_morph_info = infos[0]
                morph_info = current_morph_info
                last_env_progress = progress
            else:
                morph_info = current_morph_info

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
            eval_progress = np.nan
            curriculum_advanced = False

            if update % int(ppo.get("eval_every", 50)) == 0 or update == total_updates:
                eval_progress = resolve_eval_progress(ppo, progress)
                if eval_state_indices is None:
                    eval_metrics = evaluate_policy(
                        eval_cfg,
                        model,
                        episodes=int(ppo.get("eval_episodes", 5)),
                        seed=seed + update * 17,
                        progress=eval_progress,
                    )
                else:
                    eval_metrics = evaluate_indexed_policy_batched(
                        eval_cfg,
                        model,
                        state_indices=eval_state_indices,
                        seed=eval_seed,
                        progress=eval_progress,
                        batch_size=curriculum_eval_batch_size,
                    )
                eval_score = checkpoint_score(eval_metrics)
                if best_eval_score is None or eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_eval_return = eval_metrics["return_mean"]
                    save_model(model, ckpt_dir / "best.safetensors")
                    dump_json(
                        {
                            "update": update,
                            "progress": progress,
                            "eval_progress": eval_progress,
                            "eval": eval_metrics,
                            "curriculum_mode": curriculum_mode,
                            "checkpoint_score": list(eval_score),
                            "checkpoint_score_order": [
                                "success_rate",
                                "low_momentum_upright_rate",
                                "max_low_momentum_upright_streak_mean",
                                "ever_upright_rate",
                                "max_centered_upright_streak_mean",
                                "max_upright_streak_mean",
                                "max_upright_streak_max",
                                "max_capture_quality_mean",
                                "return_mean",
                            ],
                            "global_steps": global_steps,
                            "curriculum_evaluation": curriculum_eval_metadata,
                        },
                        ckpt_dir / "best.meta.json",
                    )
                if (
                    curriculum_mode == "gated"
                    and math.isclose(eval_progress, progress, rel_tol=0.0, abs_tol=1e-12)
                    and progress < 1.0
                    and update - last_curriculum_advance_update >= curriculum_min_updates
                    and curriculum_gate_passed(ppo, eval_metrics)
                ):
                    next_progress = float(np.clip(progress + curriculum_step, 0.0, 1.0))
                    save_model(model, ckpt_dir / "frontier.safetensors")
                    dump_json(
                        {
                            "update": update,
                            "progress": progress,
                            "eval_progress": eval_progress,
                            "curriculum_progress_next": next_progress,
                            "curriculum_mode": curriculum_mode,
                            "eval": eval_metrics,
                            "checkpoint_score": list(eval_score),
                            "checkpoint_score_order": [
                                "success_rate",
                                "low_momentum_upright_rate",
                                "max_low_momentum_upright_streak_mean",
                                "ever_upright_rate",
                                "max_centered_upright_streak_mean",
                                "max_upright_streak_mean",
                                "max_upright_streak_max",
                                "max_capture_quality_mean",
                                "return_mean",
                            ],
                            "global_steps": global_steps,
                            "curriculum_evaluation": curriculum_eval_metadata,
                        },
                        ckpt_dir / "frontier.meta.json",
                    )
                    curriculum_progress = next_progress
                    last_curriculum_advance_update = update
                    curriculum_advanced = curriculum_progress > progress

            if update % int(ppo.get("checkpoint_every", 50)) == 0 or update == total_updates:
                save_model(model, ckpt_dir / f"update_{update:06d}.safetensors")
                save_model(model, ckpt_dir / "latest.safetensors")
                dump_json(
                    {
                        "update": update,
                        "progress": progress,
                        "curriculum_progress_next": curriculum_progress,
                        "curriculum_mode": curriculum_mode,
                        "global_steps": global_steps,
                        "elapsed_seconds": time.time() - start_time,
                    },
                    ckpt_dir / "latest.meta.json",
                )

            row = {
                "update": update,
                "global_steps": global_steps,
                "progress": progress,
                "plant_progress": morph_info.get("plant_progress", progress),
                "eval_progress": eval_progress,
                "steps_per_sec": steps_per_sec,
                "mean_ep_return": mean_ep_return,
                "mean_ep_len": mean_ep_len,
                **last_loss_metrics,
                "eval_return_mean": eval_metrics.get("return_mean", np.nan),
                "eval_success_rate": eval_metrics.get("success_rate", np.nan),
                "eval_length_mean": eval_metrics.get("length_mean", np.nan),
                "eval_ever_upright_rate": eval_metrics.get("ever_upright_rate", np.nan),
                "eval_max_upright_streak_mean": eval_metrics.get("max_upright_streak_mean", np.nan),
                "eval_max_upright_streak_max": eval_metrics.get("max_upright_streak_max", np.nan),
                "eval_max_centered_upright_streak_mean": eval_metrics.get("max_centered_upright_streak_mean", np.nan),
                "eval_max_centered_upright_streak_max": eval_metrics.get("max_centered_upright_streak_max", np.nan),
                "eval_max_low_momentum_upright_streak_mean": eval_metrics.get("max_low_momentum_upright_streak_mean", np.nan),
                "eval_max_low_momentum_upright_streak_max": eval_metrics.get("max_low_momentum_upright_streak_max", np.nan),
                "eval_low_momentum_upright_rate": eval_metrics.get("low_momentum_upright_rate", np.nan),
                "eval_max_capture_quality_mean": eval_metrics.get("max_capture_quality_mean", np.nan),
                "eval_max_capture_quality_max": eval_metrics.get("max_capture_quality_max", np.nan),
                "eval_time_to_first_upright_mean": eval_metrics.get("time_to_first_upright_mean", np.nan),
                "curriculum_advanced": curriculum_advanced,
                "rail_limit": morph_info.get("rail_limit", np.nan),
                "alpha_length": morph_info.get("alpha_length", np.nan),
                "alpha_mass": morph_info.get("alpha_mass", np.nan),
                "alpha_damping": morph_info.get("alpha_damping", np.nan),
                "alpha_frictionloss": morph_info.get("alpha_frictionloss", np.nan),
                "init_qvel_scale": morph_info.get("init_qvel_scale", np.nan),
            }
            writer.writerow(row)
            csv_file.flush()

            print(
                f"upd={update:05d}/{total_updates} steps={global_steps:,} "
                f"prog={progress:.3f} sps={steps_per_sec:,.0f} "
                f"ep_ret={mean_ep_return:.1f} ep_len={mean_ep_len:.1f} "
                f"eval_prog={eval_progress:.3f} "
                f"eval_ret={eval_metrics.get('return_mean', float('nan')):.1f} "
                f"succ={eval_metrics.get('success_rate', float('nan')):.2f} "
                f"adv={int(curriculum_advanced)}"
            )

    finally:
        csv_file.close()
        envs.close()

    return {"out_dir": str(out_dir), "best_eval_return": best_eval_return, "global_steps": global_steps}
