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
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - exercised on the CPU training host
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from .config import dump_json, save_config
from .vecenv import make_vec_env

LOG_2PI = float(math.log(2.0 * math.pi))


def require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "Could not import PyTorch. Use the system Python with the project "
            f"environment on PYTHONPATH. Error: {_TORCH_IMPORT_ERROR}"
        )


class ActorCritic(nn.Module):
    """Small tanh Gaussian actor-critic matching the MLX PPO architecture."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int],
        action_std_init: float = 0.7,
    ):
        super().__init__()
        sizes = [int(obs_dim)] + [int(size) for size in hidden_sizes]
        self.actor_layers = nn.ModuleList(
            [nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])]
        )
        self.actor_out = nn.Linear(sizes[-1], int(act_dim))
        self.critic_layers = nn.ModuleList(
            [nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])]
        )
        self.critic_out = nn.Linear(sizes[-1], 1)
        self.log_std = nn.Parameter(
            torch.full((int(act_dim),), float(math.log(action_std_init)), dtype=torch.float32)
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actor = obs
        for layer in self.actor_layers:
            actor = torch.tanh(layer(actor))
        mean = torch.tanh(self.actor_out(actor))

        critic = obs
        for layer in self.critic_layers:
            critic = torch.tanh(layer(critic))
        value = self.critic_out(critic).squeeze(-1)
        return mean, value


def clipped_log_std(model: ActorCritic) -> torch.Tensor:
    return torch.clamp(model.log_std, -5.0, 2.0)


def gaussian_log_prob(
    action: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
) -> torch.Tensor:
    return -0.5 * torch.sum(
        ((action - mean) / torch.exp(log_std)) ** 2 + 2.0 * log_std + LOG_2PI,
        dim=-1,
    )


def gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    return torch.sum(0.5 + 0.5 * LOG_2PI + log_std)


def sample_action(
    model: ActorCritic,
    obs_np: np.ndarray,
    *,
    deterministic: bool = False,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_torch()
    if device is None:
        device = next(model.parameters()).device
    with torch.no_grad():
        obs = torch.as_tensor(obs_np.astype(np.float32), device=device)
        mean, value = model(obs)
        log_std = clipped_log_std(model)
        if deterministic:
            action = mean
        else:
            action = torch.clamp(mean + torch.exp(log_std) * torch.randn_like(mean), -1.0, 1.0)
        logp = gaussian_log_prob(action, mean, log_std)
    return (
        action.detach().cpu().numpy().astype(np.float32),
        logp.detach().cpu().numpy().astype(np.float32),
        value.detach().cpu().numpy().astype(np.float32),
    )


def value_only(
    model: ActorCritic,
    obs_np: np.ndarray,
    *,
    device: torch.device | None = None,
) -> np.ndarray:
    require_torch()
    if device is None:
        device = next(model.parameters()).device
    with torch.no_grad():
        obs = torch.as_tensor(obs_np.astype(np.float32), device=device)
        _, value = model(obs)
    return value.detach().cpu().numpy().astype(np.float32)


def ppo_loss_fn(
    model: ActorCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    clip_coef: float,
    entropy_coef: float,
    value_coef: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    mean, values = model(obs)
    log_std = clipped_log_std(model)
    logp = gaussian_log_prob(actions, mean, log_std)
    entropy = gaussian_entropy(log_std)

    logratio = logp - old_logp
    ratio = torch.exp(logratio)
    pg_loss_unclipped = -advantages * ratio
    pg_loss_clipped = -advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    policy_loss = torch.mean(torch.maximum(pg_loss_unclipped, pg_loss_clipped))

    value_delta = torch.clamp(values - old_values, -clip_coef, clip_coef)
    v_clipped = old_values + value_delta
    v_loss_unclipped = (values - returns) ** 2
    v_loss_clipped = (v_clipped - returns) ** 2
    value_loss = 0.5 * torch.mean(torch.maximum(v_loss_unclipped, v_loss_clipped))

    approx_kl = torch.mean((ratio - 1.0) - logratio)
    clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip_coef).float())
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total, (policy_loss, value_loss, entropy, approx_kl, clip_fraction)


def save_model(model: ActorCritic, path: str | Path) -> None:
    require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(state, path)


def load_model(model: ActorCritic, path: str | Path) -> None:
    require_torch()
    raw = torch.load(Path(path), map_location=next(model.parameters()).device, weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    model.load_state_dict(raw)


def select_evaluation_state_indices(path: str | Path, episodes: int, seed: int) -> list[int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    states = payload.get("states", payload) if isinstance(payload, dict) else payload
    if not isinstance(states, list):
        raise ValueError("curriculum evaluation state file must contain a state list")
    if episodes < 1 or episodes > len(states):
        raise ValueError(f"curriculum eval episodes must be in 1..{len(states)}")
    rng = np.random.default_rng(seed)
    return rng.permutation(len(states))[:episodes].astype(int).tolist()


def evaluate_policy(
    cfg: dict[str, Any],
    model: ActorCritic,
    episodes: int,
    seed: int,
    progress: float = 1.0,
    return_episodes: bool = False,
    reset_state_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate the deterministic policy with the same evidence fields as MLX PPO."""
    from .env import NLinkCartPoleEnv

    require_torch()
    if reset_state_indices is not None and len(reset_state_indices) != episodes:
        raise ValueError("reset_state_indices must contain exactly one index per evaluation episode")
    device = next(model.parameters()).device
    model.eval()
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    max_angles: list[float] = []
    time_to_uprights: list[float] = []
    time_to_captures: list[float] = []
    max_upright_streaks: list[float] = []
    final_upright_streaks: list[float] = []
    max_cart_excursions: list[float] = []
    max_centered_upright_streaks: list[float] = []
    max_low_momentum_upright_streaks: list[float] = []
    max_capture_qualities: list[float] = []
    low_momentum_upright_events: list[float] = []
    episode_results: list[dict[str, Any]] = []
    reward_cfg = cfg.get("env", {}).get("reward", {})
    hinge_threshold = float(reward_cfg.get("upright_hinge_vel_threshold", 1.0))
    cart_threshold = float(reward_cfg.get("upright_cart_vel_threshold", 1.0))
    min_time = float(reward_cfg.get("low_momentum_min_time_seconds", 0.0))
    max_cart_abs = reward_cfg.get("low_momentum_max_cart_abs")

    for ep in range(int(episodes)):
        env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed + 10_000 + ep)
        options = None if reset_state_indices is None else {"state_index": int(reset_state_indices[ep])}
        obs, _ = env.reset(options=options)
        done = False
        ep_return = 0.0
        ep_len = 0
        ep_max_angle = 0.0
        ep_max_capture_quality = 0.0
        ep_low_momentum_upright = False
        info: dict[str, Any] = {}
        while not done:
            action, _, _ = sample_action(model, obs[None, :], deterministic=True, device=device)
            obs, reward, terminated, truncated, info = env.step(action[0])
            ep_return += float(reward)
            ep_len += 1
            ep_max_angle = max(ep_max_angle, float(info.get("max_abs_angle", 0.0)))
            ep_max_capture_quality = max(ep_max_capture_quality, float(info.get("capture_quality", 0.0)))
            ep_low_momentum_upright = ep_low_momentum_upright or bool(
                info.get("is_upright", False)
                and ep_len * env.dt >= min_time
                and float(info.get("hinge_velocity_rms", np.inf)) <= hinge_threshold
                and abs(float(env.data.qvel[0])) <= cart_threshold
                and (max_cart_abs is None or abs(float(env.data.qpos[0])) <= float(max_cart_abs))
            )
            done = bool(terminated or truncated)
        env.close()

        success = bool(info.get("success", False))
        ttu = info.get("time_to_first_upright")
        ttc = info.get("time_to_capture")
        max_streak = float(info.get("max_upright_streak_seconds", 0.0))
        final_streak = float(info.get("upright_streak_seconds", 0.0))
        max_cart_excursion = float(info.get("max_cart_excursion", abs(float(info.get("x", 0.0)))))
        centered_streak = float(info.get("max_centered_upright_streak_seconds", 0.0))
        low_momentum_streak = float(info.get("max_low_momentum_upright_streak_seconds", 0.0))
        returns.append(ep_return)
        lengths.append(ep_len)
        successes.append(float(success))
        max_angles.append(ep_max_angle)
        max_capture_qualities.append(ep_max_capture_quality)
        low_momentum_upright_events.append(float(ep_low_momentum_upright))
        if ttu is not None:
            time_to_uprights.append(float(ttu))
        if ttc is not None:
            time_to_captures.append(float(ttc))
        max_upright_streaks.append(max_streak)
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
                    "max_upright_streak_seconds": max_streak,
                    "final_upright_streak_seconds": final_streak,
                    "max_centered_upright_streak_seconds": centered_streak,
                    "max_low_momentum_upright_streak_seconds": low_momentum_streak,
                    "max_capture_quality": float(ep_max_capture_quality),
                    "low_momentum_upright": bool(ep_low_momentum_upright),
                }
            )

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


def checkpoint_score(eval_metrics: dict[str, Any]) -> tuple[float, ...]:
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
            value = float(normalized)
    else:
        value = float(mode)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("ppo.eval_progress must resolve inside [0, 1]")
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


CHECKPOINT_SCORE_ORDER = [
    "success_rate",
    "low_momentum_upright_rate",
    "max_low_momentum_upright_streak_mean",
    "ever_upright_rate",
    "max_centered_upright_streak_mean",
    "max_upright_streak_mean",
    "max_upright_streak_max",
    "max_capture_quality_mean",
    "return_mean",
]


def train(cfg: dict[str, Any], init_checkpoint: str | None = None) -> dict[str, Any]:
    require_torch()
    seed = int(cfg.get("experiment", {}).get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    ppo = cfg["ppo"]
    thread_count = int(ppo.get("torch_num_threads", 1))
    if thread_count > 0:
        torch.set_num_threads(thread_count)
    device = torch.device(str(ppo.get("device", "cpu")))
    if device.type != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested torch device {device}, but it is unavailable")

    out_dir = Path(cfg["experiment"]["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.resolved.yaml")

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
    curriculum_gate_evals_required = max(
        1,
        int(ppo.get("curriculum_gate_consecutive_evals", 1)),
    )
    curriculum_gate_streak = 0
    last_curriculum_advance_update = 0
    last_env_progress: float | None = None
    current_morph_info: dict[str, Any] = {}
    rollback_on_eval_regression = bool(ppo.get("rollback_on_eval_regression", False))
    stage_best_state: dict[str, Any] | None = None
    stage_best_optimizer_state: dict[str, Any] | None = None
    stage_best_score: tuple[float, ...] | None = None

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

    envs = make_vec_env(cfg, num_envs=num_envs, seed=seed, progress=0.0, backend=vec_backend)
    obs, _ = envs.reset()
    obs_dim = obs.shape[1]
    act_dim = envs.single_action_space.shape[0]
    model = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=list(ppo.get("hidden_sizes", [256, 256])),
        action_std_init=float(ppo.get("action_std_init", 0.7)),
    ).to(device)
    if init_checkpoint:
        load_model(model, init_checkpoint)
    elif bool(ppo.get("zero_init_actor_output", False)):
        with torch.no_grad():
            model.actor_out.weight.zero_()
            model.actor_out.bias.zero_()

    freeze_actor_layers = bool(ppo.get("freeze_actor_layers", False))
    freeze_log_std = bool(ppo.get("freeze_log_std", False))
    freeze_actor_until = ppo.get("freeze_actor_until_progress")
    freeze_actor_until = None if freeze_actor_until is None else float(freeze_actor_until)

    def set_actor_trainability(progress: float) -> bool:
        """Keep a proven residual teacher fixed until the first curriculum step."""
        teacher_frozen = (
            freeze_actor_until is not None
            and float(progress) <= freeze_actor_until + 1e-12
        )
        for layer in model.actor_layers:
            for parameter in layer.parameters():
                parameter.requires_grad_(not (teacher_frozen or freeze_actor_layers))
        for parameter in model.actor_out.parameters():
            parameter.requires_grad_(not teacher_frozen)
        model.log_std.requires_grad_(not (teacher_frozen or freeze_log_std))
        return teacher_frozen

    actor_teacher_frozen = set_actor_trainability(curriculum_progress)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ppo["learning_rate"]))

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
        "eval_time_to_first_upright_mean", "curriculum_advanced", "curriculum_gate_streak", "rail_limit",
        "alpha_length", "alpha_mass", "alpha_damping", "alpha_frictionloss", "init_qvel_scale",
        "curriculum_rollback", "actor_teacher_frozen",
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
            progress = (
                (update - 1) / max(1, total_updates - 1)
                if curriculum_mode == "linear"
                else curriculum_progress
            )
            actor_teacher_frozen = set_actor_trainability(progress)
            rebuild_for_progress = last_env_progress is None or not math.isclose(
                progress, last_env_progress, rel_tol=0.0, abs_tol=1e-12
            )
            if update == 1 or rebuild_for_progress or (rebuild_every > 0 and update % rebuild_every == 0):
                obs, infos = envs.set_progress(progress)
                ep_returns[:] = 0.0
                ep_lengths[:] = 0
                current_morph_info = infos[0]
                last_env_progress = progress
            morph_info = current_morph_info

            obs_buf = np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32)
            action_buf = np.zeros((rollout_steps, num_envs, act_dim), dtype=np.float32)
            logp_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
            reward_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
            done_buf = np.zeros((rollout_steps, num_envs), dtype=bool)
            value_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)

            rollout_start = time.time()
            model.eval()
            for t in range(rollout_steps):
                obs_buf[t] = obs
                actions, logp, values = sample_action(model, obs, device=device)
                action_buf[t] = actions
                logp_buf[t] = logp
                value_buf[t] = values
                next_obs, rewards, dones, infos = envs.step(actions)
                reward_buf[t] = rewards
                done_buf[t] = dones
                global_steps += num_envs
                ep_returns += rewards
                ep_lengths += 1
                for index, done in enumerate(dones):
                    if done:
                        recent_returns.append(float(ep_returns[index]))
                        recent_lengths.append(int(ep_lengths[index]))
                        recent_returns = recent_returns[-100:]
                        recent_lengths = recent_lengths[-100:]
                        ep_returns[index] = 0.0
                        ep_lengths[index] = 0
                obs = next_obs

            last_values = value_only(model, obs, device=device)
            advantages = np.zeros_like(reward_buf, dtype=np.float32)
            lastgaelam = np.zeros(num_envs, dtype=np.float32)
            gamma = float(ppo["gamma"])
            lam = float(ppo["gae_lambda"])
            for t in reversed(range(rollout_steps)):
                next_values = last_values if t == rollout_steps - 1 else value_buf[t + 1]
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
            indices = np.arange(batch_size)
            loss_sums = np.zeros(5, dtype=np.float64)
            loss_count = 0
            model.train()
            for _epoch in range(int(ppo["epochs"])):
                np.random.shuffle(indices)
                for start in range(0, batch_size, mb_size):
                    mb = indices[start : start + mb_size]
                    tensors = [
                        torch.as_tensor(b_obs[mb], device=device),
                        torch.as_tensor(b_actions[mb], device=device),
                        torch.as_tensor(b_logp[mb], device=device),
                        torch.as_tensor(b_adv[mb], device=device),
                        torch.as_tensor(b_returns[mb], device=device),
                        torch.as_tensor(b_values[mb], device=device),
                    ]
                    loss, aux = ppo_loss_fn(
                        model,
                        *tensors,
                        float(ppo["clip_coef"]),
                        float(ppo["entropy_coef"]),
                        float(ppo["value_coef"]),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    max_grad_norm = float(ppo.get("max_grad_norm", 0.0))
                    if max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    loss_sums += np.asarray([float(value.detach().cpu()) for value in aux], dtype=np.float64)
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
            eval_metrics: dict[str, Any] = {"return_mean": np.nan, "success_rate": np.nan, "length_mean": np.nan}
            eval_progress = np.nan
            curriculum_advanced = False
            curriculum_rollback = False

            if update % int(ppo.get("eval_every", 50)) == 0 or update == total_updates:
                eval_progress = resolve_eval_progress(ppo, progress)
                eval_metrics = evaluate_policy(
                    eval_cfg,
                    model,
                    episodes=int(ppo.get("eval_episodes", 5)),
                    seed=seed + update * 17,
                    progress=eval_progress,
                    reset_state_indices=eval_state_indices,
                )
                eval_score = checkpoint_score(eval_metrics)
                if rollback_on_eval_regression:
                    if stage_best_score is None or eval_score >= stage_best_score:
                        stage_best_score = eval_score
                        stage_best_state = {
                            name: value.detach().clone()
                            for name, value in model.state_dict().items()
                        }
                        stage_best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                    else:
                        if stage_best_state is not None:
                            model.load_state_dict(stage_best_state)
                        if stage_best_optimizer_state is not None:
                            optimizer.load_state_dict(stage_best_optimizer_state)
                        curriculum_rollback = True
                if not curriculum_rollback and (best_eval_score is None or eval_score > best_eval_score):
                    best_eval_score = eval_score
                    best_eval_return = float(eval_metrics["return_mean"])
                    save_model(model, ckpt_dir / "best.pt")
                    dump_json(
                        {
                            "update": update,
                            "progress": progress,
                            "eval_progress": eval_progress,
                            "eval": eval_metrics,
                            "curriculum_mode": curriculum_mode,
                            "checkpoint_score": list(eval_score),
                            "checkpoint_score_order": CHECKPOINT_SCORE_ORDER,
                            "global_steps": global_steps,
                            "curriculum_evaluation": curriculum_eval_metadata,
                        },
                        ckpt_dir / "best.meta.json",
                    )
                same_curriculum_stage = math.isclose(
                    eval_progress,
                    progress,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                if same_curriculum_stage and curriculum_gate_passed(ppo, eval_metrics):
                    curriculum_gate_streak += 1
                else:
                    curriculum_gate_streak = 0
                if (
                    curriculum_mode == "gated"
                    and same_curriculum_stage
                    and progress < 1.0
                    and update - last_curriculum_advance_update >= curriculum_min_updates
                    and curriculum_gate_streak >= curriculum_gate_evals_required
                ):
                    next_progress = float(np.clip(progress + curriculum_step, 0.0, 1.0))
                    save_model(model, ckpt_dir / "frontier.pt")
                    dump_json(
                        {
                            "update": update,
                            "progress": progress,
                            "eval_progress": eval_progress,
                            "curriculum_progress_next": next_progress,
                            "curriculum_mode": curriculum_mode,
                            "curriculum_gate_streak": curriculum_gate_streak,
                            "curriculum_gate_evals_required": curriculum_gate_evals_required,
                            "eval": eval_metrics,
                            "checkpoint_score": list(eval_score),
                            "checkpoint_score_order": CHECKPOINT_SCORE_ORDER,
                            "global_steps": global_steps,
                            "curriculum_evaluation": curriculum_eval_metadata,
                        },
                        ckpt_dir / "frontier.meta.json",
                    )
                    curriculum_progress = next_progress
                    last_curriculum_advance_update = update
                    curriculum_gate_streak = 0
                    stage_best_score = None
                    stage_best_state = None
                    stage_best_optimizer_state = None
                    curriculum_advanced = curriculum_progress > progress

            if update % int(ppo.get("checkpoint_every", 50)) == 0 or update == total_updates:
                save_model(model, ckpt_dir / f"update_{update:06d}.pt")
                save_model(model, ckpt_dir / "latest.pt")
                dump_json(
                    {
                        "update": update,
                        "progress": progress,
                        "curriculum_progress_next": curriculum_progress,
                        "curriculum_mode": curriculum_mode,
                        "curriculum_gate_streak": curriculum_gate_streak,
                        "curriculum_gate_evals_required": curriculum_gate_evals_required,
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
                "curriculum_gate_streak": curriculum_gate_streak,
                "rail_limit": morph_info.get("rail_limit", np.nan),
                "alpha_length": morph_info.get("alpha_length", np.nan),
                "alpha_mass": morph_info.get("alpha_mass", np.nan),
                "alpha_damping": morph_info.get("alpha_damping", np.nan),
                "alpha_frictionloss": morph_info.get("alpha_frictionloss", np.nan),
                "init_qvel_scale": morph_info.get("init_qvel_scale", np.nan),
                "curriculum_rollback": curriculum_rollback,
                "actor_teacher_frozen": actor_teacher_frozen,
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
