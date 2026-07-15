from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import mujoco
except Exception as exc:  # pragma: no cover - import checked by smoke_test on target machine
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None

from .mjxml import generate_nlink_cartpole_xml
from .modal import StateScales, dimensionless_absolute_transform
from .morphology import Morphology, build_morphology


_INIT_STATE_FILE_CACHE: dict[Path, list[dict[str, Any]]] = {}
_INIT_STATE_ORDER_CACHE: dict[Path, list[int]] = {}


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class NLinkCartPoleEnv(gym.Env):
    """Planar n-link cart-pole environment generated from gradient morphology parameters.

    The action space is normalized continuous force in [-1, 1]. The MuJoCo motor force is
    action * env.force_limit.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, cfg: dict[str, Any], progress: float = 0.0, seed: int | None = None, render_mode: str | None = None):
        if mujoco is None:
            raise RuntimeError(f"Could not import mujoco: {_MUJOCO_IMPORT_ERROR}")
        self.cfg = cfg
        self.env_cfg = cfg["env"]
        self.morph_cfg = cfg["morphology"]
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self.progress = float(progress)
        self.plant_progress = self._resolve_plant_progress(self.progress)
        self.n = int(self.env_cfg["n_links"])
        self.force_limit = float(self.env_cfg["force_limit"])
        self.rail_limit = self._scheduled_rail_limit(self.plant_progress)
        self.frame_skip = int(self.env_cfg.get("frame_skip", 1))
        self.max_steps = max(1, int(float(self.env_cfg["episode_seconds"]) / (float(self.env_cfg["timestep"]) * self.frame_skip)))
        self.obs_include_morphology = bool(self.env_cfg.get("obs_include_morphology", True))
        self.obs_include_frictionloss = bool(self.env_cfg.get("obs_include_frictionloss", False))
        self.obs_include_time = bool(self.env_cfg.get("obs_include_time", False))
        self.obs_include_capture_features = bool(self.env_cfg.get("obs_include_capture_features", False))
        self.step_count = 0
        self.last_action_norm = np.zeros(1, dtype=np.float32)
        self.last_policy_action_norm = np.zeros(1, dtype=np.float32)
        self.last_action_bias_norm = 0.0
        self.last_residual_scale = 1.0
        self.last_lqr_cart_target = 0.0
        self.lqr_switch_active = False
        self.last_controller_mode = "policy"
        self.lqr_switch_entry_count = 0
        self.lqr_switch_exit_count = 0
        self.lqr_switch_first_entry_step: int | None = None
        self.lqr_switch_lqr_steps = 0
        self.lqr_switch_policy_steps = 0
        self.upright_streak_steps = 0
        self.max_upright_streak_steps = 0
        self.centered_upright_streak_steps = 0
        self.max_centered_upright_streak_steps = 0
        self.low_momentum_upright_streak_steps = 0
        self.max_low_momentum_upright_streak_steps = 0
        self.first_upright_step: int | None = None
        self.capture_step: int | None = None
        self.capture_start_step: int | None = None
        self.max_cart_excursion = 0.0
        self.last_init_state_index: int | None = None
        self._init_state_cache: list[dict[str, Any]] | None = None
        self._init_state_cache_key: Path | None = None
        self._init_state_order_cache: list[int] | None = None
        self.renderer = None
        self._build_model(self.progress)

        obs_dim = self._get_obs().shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    @property
    def dt(self) -> float:
        return float(self.env_cfg["timestep"]) * self.frame_skip

    def _scheduled_rail_limit(self, progress: float) -> float:
        target = float(self.env_cfg["rail_limit"])
        start = float(self.env_cfg.get("rail_limit_start", target))
        end = float(self.env_cfg.get("rail_limit_end", target))
        t = float(np.clip(progress, 0.0, 1.0))
        return start + (end - start) * t

    def _resolve_plant_progress(self, progress: float) -> float:
        override = self.env_cfg.get("plant_progress")
        if override is None:
            return float(progress)
        return float(np.clip(float(override), 0.0, 1.0))

    def _build_model(self, progress: float) -> None:
        self.progress = float(progress)
        self.plant_progress = self._resolve_plant_progress(self.progress)
        self.rail_limit = self._scheduled_rail_limit(self.plant_progress)
        self.morphology: Morphology = build_morphology(self.env_cfg, self.morph_cfg, self.plant_progress)
        xml = generate_nlink_cartpole_xml(
            self.morphology,
            cart_mass=float(self.env_cfg["cart_mass"]),
            rail_limit=float(self.rail_limit),
            force_limit=float(self.env_cfg["force_limit"]),
            timestep=float(self.env_cfg["timestep"]),
            cart_damping=float(self.env_cfg.get("cart_damping", 0.0)),
            cart_frictionloss=float(self.env_cfg.get("cart_frictionloss", 0.0)),
            joint_armature=float(self.env_cfg.get("joint_armature", 0.0)),
            link_radius=float(self.env_cfg.get("link_radius", 0.025)),
        )
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        mujoco.mj_forward(self.model, self.data)

    def set_progress(self, progress: float) -> None:
        progress = float(progress)
        plant_progress = self._resolve_plant_progress(progress)
        rail_limit = self._scheduled_rail_limit(plant_progress)
        if np.isclose(plant_progress, self.plant_progress) and np.isclose(rail_limit, self.rail_limit):
            self.progress = progress
        else:
            self._build_model(progress)
        self.reset()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.upright_streak_steps = 0
        self.max_upright_streak_steps = 0
        self.centered_upright_streak_steps = 0
        self.max_centered_upright_streak_steps = 0
        self.low_momentum_upright_streak_steps = 0
        self.max_low_momentum_upright_streak_steps = 0
        self.first_upright_step = None
        self.capture_step = None
        self.capture_start_step = None
        self.max_cart_excursion = 0.0
        self.last_init_state_index = None
        self.lqr_switch_active = False
        self.last_controller_mode = "policy"
        self.lqr_switch_entry_count = 0
        self.lqr_switch_exit_count = 0
        self.lqr_switch_first_entry_step = None
        self.lqr_switch_lqr_steps = 0
        self.lqr_switch_policy_steps = 0
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        angle_noise = self._progress_value(self.env_cfg, "init_angle_noise", 0.02)
        vel_noise = self._progress_value(self.env_cfg, "init_vel_noise", 0.01)
        if options is not None and ("qpos" in options or "qvel" in options):
            if "qpos" not in options or "qvel" not in options:
                raise ValueError("reset options must provide both qpos and qvel")
            init_qpos = np.asarray(options["qpos"], dtype=np.float64)
            init_qvel = np.asarray(options["qvel"], dtype=np.float64)
            return self._reset_to_state(init_qpos, init_qvel, angle_noise, vel_noise)
        init_mode = str(self.env_cfg.get("init_mode", "upright"))
        if init_mode == "upright":
            base_angles = np.zeros(self.n, dtype=np.float64)
        elif init_mode == "hanging":
            # Relative-angle convention: [pi, 0, ..., 0] makes the whole
            # serial chain hang downward below the cart.
            base_angles = np.zeros(self.n, dtype=np.float64)
            base_angles[0] = np.pi
        elif init_mode == "hanging_curriculum":
            # Training curriculum: progress 0 starts upright, progress 1 starts
            # fully hanging. Evaluation at progress=1.0 is the true swing-up task.
            base_angles = np.zeros(self.n, dtype=np.float64)
            progress = float(np.clip(self.progress, 0.0, 1.0))
            progress = progress ** max(1e-6, float(self.env_cfg.get("hanging_curriculum_power", 1.0)))
            base_angles[0] = np.pi * progress
        elif init_mode == "folded":
            # A compact alternating folded start, useful for stress tests.
            base_angles = np.asarray([np.pi if i % 2 == 0 else -np.pi for i in range(self.n)], dtype=np.float64)
        elif init_mode == "fixed_state":
            init_qpos = np.asarray(self.env_cfg.get("init_qpos", []), dtype=np.float64)
            init_qvel = np.asarray(self.env_cfg.get("init_qvel", []), dtype=np.float64)
            return self._reset_to_state(init_qpos, init_qvel, angle_noise, vel_noise)
        elif init_mode == "state_list":
            states = self._load_init_states()
            if not states:
                raise ValueError("env.init_mode=state_list requires at least one init state")
            requested_index = None if options is None else options.get("state_index")
            idx = self._select_init_state_index(states, requested_index=requested_index)
            self.last_init_state_index = idx
            state = states[idx]
            init_qpos = np.asarray(state.get("qpos", []), dtype=np.float64)
            init_qvel = np.asarray(state.get("qvel", []), dtype=np.float64)
            return self._reset_to_state(init_qpos, init_qvel, angle_noise, vel_noise)
        else:
            raise ValueError(f"Unknown env.init_mode: {init_mode}")
        self.data.qpos[0] += self.rng.normal(0.0, self._progress_value(self.env_cfg, "init_cart_noise", 0.0))
        self.data.qpos[1 : 1 + self.n] = base_angles + self.rng.normal(0.0, angle_noise, size=self.n)
        self.data.qvel[:] = self.rng.normal(0.0, vel_noise, size=self.n + 1)
        self.data.qvel[0] = self.rng.normal(0.0, self._progress_value(self.env_cfg, "init_cart_vel_noise", vel_noise))
        self.data.ctrl[:] = 0.0
        self.last_action_norm[:] = 0.0
        self.last_policy_action_norm[:] = 0.0
        self.last_action_bias_norm = 0.0
        self.last_residual_scale = 1.0
        self.last_lqr_cart_target = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.max_cart_excursion = abs(float(self.data.qpos[0]))
        self._update_upright_tracking()
        return self._get_obs(), self._info()

    def _load_init_states(self) -> list[dict[str, Any]]:
        if self._init_state_cache is not None:
            return self._init_state_cache
        states = self.env_cfg.get("init_states")
        if states is None:
            path = self.env_cfg.get("init_states_path")
            if not path:
                raise ValueError("env.init_mode=state_list requires env.init_states or env.init_states_path")
            cache_key = Path(path).resolve()
            self._init_state_cache_key = cache_key
            states = _INIT_STATE_FILE_CACHE.get(cache_key)
            if states is None:
                with open(cache_key, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                states = payload.get("states", payload) if isinstance(payload, dict) else payload
                if isinstance(states, list):
                    _INIT_STATE_FILE_CACHE[cache_key] = states
        if not isinstance(states, list):
            raise ValueError("state_list initial states must be a list")
        self._init_state_cache = states
        return self._init_state_cache

    def _state_list_quality(self, state: dict[str, Any]) -> tuple[float, float, float, float]:
        qpos = np.asarray(state.get("qpos", []), dtype=np.float64)
        qvel = np.asarray(state.get("qvel", []), dtype=np.float64)
        if qpos.shape != (self.n + 1,) or qvel.shape != (self.n + 1,):
            return (float("inf"), float("inf"), float("inf"), float("inf"))
        abs_angles = wrap_angle(np.cumsum(qpos[1 : 1 + self.n]))
        hinge_rms = float(np.sqrt(np.mean(qvel[1 : 1 + self.n] ** 2)))
        return (
            float(np.max(np.abs(abs_angles))),
            hinge_rms,
            abs(float(qpos[0])),
            abs(float(qvel[0])),
        )

    def _select_init_state_index(self, states: list[dict[str, Any]], requested_index: int | None = None) -> int:
        if requested_index is not None:
            index = int(requested_index)
            if index < 0 or index >= len(states):
                raise IndexError(f"state_index {index} is outside [0, {len(states)})")
            return index
        mode = str(self.env_cfg.get("init_state_curriculum", "all")).strip().lower()
        if mode in {"", "all", "none"}:
            return int(self.rng.integers(0, len(states)))
        if mode != "quality_prefix":
            raise ValueError("env.init_state_curriculum must be 'all' or 'quality_prefix'")
        if self._init_state_cache_key is not None:
            order = _INIT_STATE_ORDER_CACHE.get(self._init_state_cache_key)
            if order is None:
                order = sorted(range(len(states)), key=lambda idx: self._state_list_quality(states[idx]))
                _INIT_STATE_ORDER_CACHE[self._init_state_cache_key] = order
        else:
            if self._init_state_order_cache is None:
                self._init_state_order_cache = sorted(
                    range(len(states)), key=lambda idx: self._state_list_quality(states[idx])
                )
            order = self._init_state_order_cache
        min_count = int(np.clip(int(self.env_cfg.get("init_state_curriculum_min_count", 1)), 1, len(states)))
        power = max(1e-6, float(self.env_cfg.get("init_state_curriculum_power", 1.0)))
        t = float(np.clip(self.progress, 0.0, 1.0)) ** power
        count = int(np.ceil(min_count + (len(states) - min_count) * t))
        count = int(np.clip(count, min_count, len(states)))
        return int(order[int(self.rng.integers(0, count))])

    def _init_qvel_scale(self) -> float:
        return self._progress_value(self.env_cfg, "init_qvel_scale", 1.0)

    def _reset_to_state(self, init_qpos: np.ndarray, init_qvel: np.ndarray, angle_noise: float, vel_noise: float):
        expected = self.n + 1
        if init_qpos.shape != (expected,) or init_qvel.shape != (expected,):
            raise ValueError(
                "fixed state reset requires qpos and qvel "
                f"with length {expected}; got {init_qpos.shape} and {init_qvel.shape}"
            )
        self.data.qpos[:] = init_qpos
        qpos_scale = self._progress_value(self.env_cfg, "init_qpos_scale", 1.0)
        if not np.isclose(qpos_scale, 1.0) or "init_qpos_scale_start" in self.env_cfg or "init_qpos_scale_end" in self.env_cfg:
            qpos_target = np.zeros(expected, dtype=np.float64)
            qpos_target[0] = float(self.env_cfg.get("init_qpos_cart_target", self.env_cfg.get("init_state_cart_target", 0.0)))
            self.data.qpos[:] = qpos_target + qpos_scale * (init_qpos - qpos_target)
            self.data.qpos[1 : 1 + self.n] = qpos_scale * wrap_angle(init_qpos[1 : 1 + self.n])
        self.data.qvel[:] = init_qvel * self._init_qvel_scale()
        self.data.qpos[0] += self.rng.normal(0.0, self._progress_value(self.env_cfg, "init_cart_noise", 0.0))
        self.data.qvel[0] += self.rng.normal(0.0, self._progress_value(self.env_cfg, "init_cart_vel_noise", vel_noise))
        self.data.qpos[1 : 1 + self.n] += self.rng.normal(0.0, angle_noise, size=self.n)
        self.data.qvel[1 : 1 + self.n] += self.rng.normal(0.0, vel_noise, size=self.n)
        self.data.ctrl[:] = 0.0
        self.last_action_norm[:] = 0.0
        self.last_policy_action_norm[:] = 0.0
        self.last_action_bias_norm = 0.0
        self.last_residual_scale = 1.0
        self.last_lqr_cart_target = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.max_cart_excursion = abs(float(self.data.qpos[0]))
        self._update_upright_tracking()
        return self._get_obs(), self._info()

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        policy_action_norm = float(np.clip(action_arr[0], -1.0, 1.0))
        action_norm = self._applied_action_norm(policy_action_norm)
        self.last_policy_action_norm[0] = policy_action_norm
        self.last_action_norm[0] = action_norm
        self.data.ctrl[0] = action_norm * self.force_limit
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        self.max_cart_excursion = max(self.max_cart_excursion, abs(float(self.data.qpos[0])))
        self._update_upright_tracking()

        obs = self._get_obs()
        reward = self._reward(action_norm)
        termination_reason = self._termination_reason()
        terminated = termination_reason is not None
        truncated = self.step_count >= self.max_steps
        if terminated and not truncated:
            reward += float(self.env_cfg.get("reward", {}).get("terminal_penalty", 0.0))
        info = self._info()
        info["termination_reason"] = termination_reason if terminated else ("time_limit" if truncated else None)
        info["success"] = bool(truncated and not terminated and self._success())
        return obs, reward, terminated, truncated, info

    def _progress_value(self, cfg: dict[str, Any], key: str, default: float) -> float:
        start = cfg.get(f"{key}_start")
        end = cfg.get(f"{key}_end")
        if start is None and end is None:
            return float(cfg.get(key, default))
        base = float(cfg.get(key, default))
        start = base if start is None else float(start)
        end = base if end is None else float(end)
        t = float(np.clip(self.progress, 0.0, 1.0))
        t = t ** max(1e-9, float(cfg.get(f"{key}_power", 1.0)))
        return float(start + (end - start) * t)

    def _lqr_action_bias(self, cfg: dict[str, Any]) -> tuple[float, float]:
        gain = np.asarray(cfg.get("state_gain", []), dtype=np.float64)
        expected = 2 * (self.n + 1)
        if gain.shape != (expected,):
            raise ValueError(f"LQR action state_gain must have length {expected}; got {gain.shape}")
        cart_target = self._progress_value(cfg, "cart_target", 0.0)
        scale = self._progress_value(cfg, "scale", 1.0)
        state = np.zeros(expected, dtype=np.float64)
        state[0] = float(self.data.qpos[0]) - cart_target
        state[1 : 1 + self.n] = wrap_angle(np.asarray(self.data.qpos[1 : 1 + self.n], dtype=np.float64))
        state[self.n + 1 :] = np.asarray(self.data.qvel, dtype=np.float64)
        raw_action = -scale * float(gain @ state)
        action_squash = str(cfg.get("action_squash", "clip")).lower()
        if action_squash == "tanh":
            action = float(np.tanh(raw_action))
        elif action_squash == "clip":
            action = float(np.clip(raw_action, -1.0, 1.0))
        else:
            raise ValueError("LQR action_squash must be 'clip' or 'tanh'")
        return action, float(cart_target)

    def _within_lqr_switch_bounds(self, cfg: dict[str, Any], prefix: str) -> bool:
        _, abs_angles = self._angles()
        qpos = self.data.qpos
        qvel = self.data.qvel
        hinge_velocity_rms = float(np.sqrt(np.mean(qvel[1 : 1 + self.n] ** 2)))

        def limit(name: str) -> float:
            enter_key = f"enter_{name}"
            return float(cfg.get(f"{prefix}_{name}", cfg[enter_key]))

        within_box = bool(
            float(np.max(np.abs(abs_angles))) <= limit("max_abs_angle")
            and hinge_velocity_rms <= limit("hinge_velocity_rms")
            and abs(float(qpos[0])) <= limit("cart_abs")
            and abs(float(qvel[0])) <= limit("cart_velocity_abs")
        )
        lyapunov_limit = cfg.get(
            f"{prefix}_lyapunov_max",
            cfg.get("enter_lyapunov_max"),
        )
        if not within_box or lyapunov_limit is None:
            return within_box
        return self._lqr_switch_lyapunov_value(cfg) <= float(lyapunov_limit)

    def _lqr_switch_lyapunov_value(self, cfg: dict[str, Any]) -> float:
        state_size = 2 * (self.n + 1)
        if "state_transform" in cfg:
            transform = np.asarray(cfg["state_transform"], dtype=np.float64)
        else:
            scales = cfg.get("state_scales", {})
            try:
                transform = dimensionless_absolute_transform(
                    self.n,
                    StateScales(
                        float(scales["cart_position"]),
                        float(scales["absolute_angle"]),
                        float(scales["cart_velocity"]),
                        float(scales["hinge_velocity"]),
                    ),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Lyapunov-gated LQR switch requires state_transform or four positive state_scales"
                ) from exc
        lyapunov = np.asarray(cfg.get("lyapunov_matrix", []), dtype=np.float64)
        expected = (state_size, state_size)
        if transform.shape != expected or lyapunov.shape != expected:
            raise ValueError(
                "Lyapunov-gated LQR switch requires state_transform and "
                f"lyapunov_matrix with shape {expected}"
            )
        state = np.r_[np.asarray(self.data.qpos), np.asarray(self.data.qvel)].astype(np.float64)
        state[1 : 1 + self.n] = wrap_angle(state[1 : 1 + self.n])
        dimensionless_state = transform @ state
        value = float(dimensionless_state @ lyapunov @ dimensionless_state)
        return value if np.isfinite(value) else float("inf")

    def _applied_action_norm(self, policy_action_norm: float) -> float:
        residual_cfg = self.env_cfg.get("action_lqr_residual", {})
        switch_cfg = self.env_cfg.get("action_lqr_switch", {})
        residual_enabled = bool(residual_cfg.get("enabled", False))
        switch_enabled = bool(switch_cfg.get("enabled", False))
        if residual_enabled and switch_enabled:
            raise ValueError("action_lqr_residual and action_lqr_switch are mutually exclusive")
        if switch_enabled:
            was_active = self.lqr_switch_active
            if self.lqr_switch_active:
                self.lqr_switch_active = self._within_lqr_switch_bounds(switch_cfg, "exit")
            elif self._within_lqr_switch_bounds(switch_cfg, "enter"):
                self.lqr_switch_active = True
            if self.lqr_switch_active and not was_active:
                self.lqr_switch_entry_count += 1
                if self.lqr_switch_first_entry_step is None:
                    self.lqr_switch_first_entry_step = self.step_count
            elif was_active and not self.lqr_switch_active:
                self.lqr_switch_exit_count += 1
            if self.lqr_switch_active:
                bias, cart_target = self._lqr_action_bias(switch_cfg)
                self.last_action_bias_norm = float(bias)
                self.last_residual_scale = 0.0
                self.last_lqr_cart_target = float(cart_target)
                self.last_controller_mode = "lqr"
                self.lqr_switch_lqr_steps += 1
                return float(bias)
            self.last_action_bias_norm = 0.0
            self.last_residual_scale = 1.0
            self.last_lqr_cart_target = 0.0
            self.last_controller_mode = "policy"
            self.lqr_switch_policy_steps += 1
            return float(policy_action_norm)
        if not residual_enabled:
            self.last_action_bias_norm = 0.0
            self.last_residual_scale = 1.0
            self.last_lqr_cart_target = 0.0
            self.last_controller_mode = "policy"
            return float(policy_action_norm)
        bias, cart_target = self._lqr_action_bias(residual_cfg)
        residual_scale = self._progress_value(residual_cfg, "residual_scale", 1.0)
        self.last_action_bias_norm = float(bias)
        self.last_residual_scale = float(residual_scale)
        self.last_lqr_cart_target = float(cart_target)
        self.last_controller_mode = "lqr_residual"
        return float(np.clip(bias + residual_scale * policy_action_norm, -1.0, 1.0))

    def _angles(self) -> tuple[np.ndarray, np.ndarray]:
        rel = wrap_angle(np.array(self.data.qpos[1 : 1 + self.n], dtype=np.float64))
        abs_angles = wrap_angle(np.cumsum(np.array(self.data.qpos[1 : 1 + self.n], dtype=np.float64)))
        return rel, abs_angles

    def _get_obs(self) -> np.ndarray:
        qpos = np.array(self.data.qpos, dtype=np.float64)
        qvel = np.array(self.data.qvel, dtype=np.float64)
        rel, abs_angles = self._angles()
        obs_parts = [
            np.array([qpos[0] / self.rail_limit, qvel[0]], dtype=np.float64),
            np.sin(abs_angles),
            np.cos(abs_angles),
            rel,
            qvel[1 : 1 + self.n],
        ]
        if self.obs_include_morphology:
            obs_parts.append(self.morphology.fingerprint())
        if self.obs_include_frictionloss:
            obs_parts.append(self.morphology.frictionloss_fingerprint())
        if self.obs_include_capture_features:
            qpos_scale = max(1e-9, abs(self._progress_value(self.env_cfg, "init_qpos_scale", 1.0)))
            qvel_scale = max(1e-9, abs(self._init_qvel_scale()))
            cart_position_bound = max(1e-9, float(self.env_cfg.get("capture_feature_cart_position_bound", 1.25)))
            angle_bound = max(1e-9, float(self.env_cfg.get("capture_feature_angle_bound", 0.15)))
            cart_velocity_bound = max(1e-9, float(self.env_cfg.get("capture_feature_cart_velocity_bound", 0.50)))
            hinge_velocity_bound = max(1e-9, float(self.env_cfg.get("capture_feature_hinge_velocity_bound", 0.75)))
            residual_cfg = self.env_cfg.get("action_lqr_residual", {})
            switch_cfg = self.env_cfg.get("action_lqr_switch", {})
            lqr_bias = 0.0
            if bool(residual_cfg.get("enabled", False)):
                lqr_bias, _ = self._lqr_action_bias(residual_cfg)
            elif bool(switch_cfg.get("enabled", False)):
                lqr_bias, _ = self._lqr_action_bias(switch_cfg)
            capture_features = np.r_[
                qpos[0] / (cart_position_bound * qpos_scale),
                qvel[0] / (cart_velocity_bound * qvel_scale),
                abs_angles / (angle_bound * qpos_scale),
                qvel[1 : 1 + self.n] / (hinge_velocity_bound * qvel_scale),
                lqr_bias,
            ]
            obs_parts.append(np.clip(capture_features, -10.0, 10.0))
        if self.obs_include_time:
            time_scale = max(1e-9, float(self.env_cfg.get("obs_time_scale_seconds", self.env_cfg["episode_seconds"])))
            phase = float(self.step_count * self.dt) / time_scale
            time_parts = [phase]
            for freq in self.env_cfg.get("obs_time_frequencies", []):
                angle = 2.0 * np.pi * float(freq) * phase
                time_parts.extend([float(np.sin(angle)), float(np.cos(angle))])
            obs_parts.append(np.asarray(time_parts, dtype=np.float64))
        obs = np.concatenate(obs_parts).astype(np.float32)
        # Keep pathological MuJoCo explosions from poisoning PPO batches.
        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    def _reward(self, action_norm: float) -> float:
        reward_cfg = self.env_cfg.get("reward", {})
        rel, abs_angles = self._angles()
        qpos = self.data.qpos
        qvel = self.data.qvel

        hinge_vel_rms = float(np.sqrt(np.mean(qvel[1 : 1 + self.n] ** 2)))
        max_abs_angle = float(np.max(np.abs(abs_angles)))
        angle_abs_cost = float(np.mean(1.0 - np.cos(abs_angles)))
        angle_cos_mean = float(np.mean(np.cos(abs_angles)))
        angle_rel_cost = float(np.mean(rel * rel))
        tip_cost = float(1.0 - np.cos(abs_angles[-1])) if abs_angles.size else 0.0
        cart_pos_cost = float((qpos[0] / self.rail_limit) ** 2)
        cart_vel_cost = float(qvel[0] ** 2)
        hinge_vel_cost = float(hinge_vel_rms * hinge_vel_rms)
        control_cost = float(action_norm * action_norm)
        policy_control_cost = float(self.last_policy_action_norm[0] ** 2)
        upright = float(self._is_upright(abs_angles))
        capture_quality = self._capture_quality(max_abs_angle=max_abs_angle, hinge_vel_rms=hinge_vel_rms)
        rail_margin_start = float(reward_cfg.get("rail_margin_start", 1.0))
        rail_margin = max(0.0, abs(float(qpos[0])) / self.rail_limit - rail_margin_start)
        low_momentum_reward_min_time = float(
            reward_cfg.get(
                "low_momentum_reward_min_time_seconds",
                reward_cfg.get("low_momentum_min_time_seconds", 0.0),
            )
        )
        max_handoff_cart_abs = reward_cfg.get("low_momentum_max_cart_abs")
        upright_low_velocity = float(
            bool(upright)
            and float(self.step_count * self.dt) >= low_momentum_reward_min_time
            and hinge_vel_rms <= float(reward_cfg.get("upright_hinge_vel_threshold", 1.0))
            and abs(float(qvel[0])) <= float(reward_cfg.get("upright_cart_vel_threshold", 1.0))
            and (
                max_handoff_cart_abs is None
                or abs(float(qpos[0])) <= float(max_handoff_cart_abs)
            )
        )
        upright_streak_seconds = min(
            float(self.upright_streak_steps * self.dt),
            float(reward_cfg.get("upright_streak_cap_seconds", self.env_cfg.get("success_sustain_seconds", 0.0))),
        )
        centered_upright_streak_seconds = min(
            float(self.centered_upright_streak_steps * self.dt),
            float(
                reward_cfg.get(
                    "centered_upright_streak_cap_seconds",
                    reward_cfg.get("upright_streak_cap_seconds", self.env_cfg.get("success_sustain_seconds", 0.0)),
                )
            ),
        )
        low_momentum_upright_streak_seconds = min(
            float(self.low_momentum_upright_streak_steps * self.dt),
            float(
                reward_cfg.get(
                    "low_momentum_upright_streak_cap_seconds",
                    reward_cfg.get("centered_upright_streak_cap_seconds", self.env_cfg.get("success_sustain_seconds", 0.0)),
                )
            ),
        )

        reward = float(reward_cfg.get("alive", 1.0))
        reward += float(reward_cfg.get("angle_cos", 0.0)) * angle_cos_mean
        reward += float(reward_cfg.get("upright_bonus", 0.0)) * upright
        reward += float(reward_cfg.get("centered_upright_bonus", 0.0)) * float(self.centered_upright_streak_steps > 0)
        reward += float(reward_cfg.get("sustained_upright_bonus", 0.0)) * float(self.upright_streak_steps > 0)
        reward += float(reward_cfg.get("capture_quality", 0.0)) * capture_quality
        reward += float(reward_cfg.get("upright_low_velocity_bonus", 0.0)) * upright_low_velocity
        reward += float(reward_cfg.get("upright_streak", 0.0)) * upright_streak_seconds
        reward += float(reward_cfg.get("centered_upright_streak", 0.0)) * centered_upright_streak_seconds
        reward += float(reward_cfg.get("low_momentum_upright_streak", 0.0)) * low_momentum_upright_streak_seconds
        reward -= float(reward_cfg.get("angle_abs", 2.5)) * angle_abs_cost
        reward -= float(reward_cfg.get("angle_rel", 0.15)) * angle_rel_cost
        reward -= float(reward_cfg.get("tip", 0.0)) * tip_cost
        reward -= float(reward_cfg.get("cart_pos", 0.10)) * cart_pos_cost
        reward -= float(reward_cfg.get("cart_vel", 0.005)) * cart_vel_cost
        reward -= float(reward_cfg.get("hinge_vel", 0.001)) * hinge_vel_cost
        reward -= float(reward_cfg.get("rail_margin", 0.0)) * rail_margin * rail_margin
        reward -= float(reward_cfg.get("control", 0.0003)) * control_cost
        reward -= float(reward_cfg.get("policy_control", 0.0)) * policy_control_cost
        return float(np.clip(reward, -100.0, 100.0))

    def _capture_quality(self, *, max_abs_angle: float | None = None, hinge_vel_rms: float | None = None) -> float:
        reward_cfg = self.env_cfg.get("reward", {})
        _, abs_angles = self._angles()
        qpos = self.data.qpos
        qvel = self.data.qvel
        angle_scale = max(1e-6, float(reward_cfg.get("capture_angle_scale", 0.30)))
        hinge_vel_scale = max(1e-6, float(reward_cfg.get("capture_hinge_vel_scale", 1.25)))
        cart_pos_scale = max(1e-6, float(reward_cfg.get("capture_cart_pos_scale", 0.60)))
        cart_vel_scale = max(1e-6, float(reward_cfg.get("capture_cart_vel_scale", 1.00)))
        if max_abs_angle is None:
            max_abs_angle = float(np.max(np.abs(abs_angles)))
        if hinge_vel_rms is None:
            hinge_vel_rms = float(np.sqrt(np.mean(qvel[1 : 1 + self.n] ** 2)))
        cart_pos_norm = abs(float(qpos[0])) / self.rail_limit
        cart_pos_abs_scale = reward_cfg.get("capture_cart_pos_abs_scale")
        if cart_pos_abs_scale is None:
            cart_pos_cost = cart_pos_norm / cart_pos_scale
        else:
            cart_pos_cost = abs(float(qpos[0])) / max(1e-6, float(cart_pos_abs_scale))
        cost = (
            (float(max_abs_angle) / angle_scale) ** 2
            + (float(hinge_vel_rms) / hinge_vel_scale) ** 2
            + cart_pos_cost ** 2
            + (abs(float(qvel[0])) / cart_vel_scale) ** 2
        )
        return float(np.exp(-min(50.0, cost)))

    def _termination_reason(self) -> str | None:
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            return "non_finite_state"
        if abs(float(self.data.qpos[0])) > self.rail_limit:
            return "rail_violation"
        _, abs_angles = self._angles()
        terminate_abs_angle = self.env_cfg.get("terminate_abs_angle", 1.25)
        if terminate_abs_angle is not None and float(np.max(np.abs(abs_angles))) > float(terminate_abs_angle):
            return "angle_limit"
        return None

    def _terminated(self) -> bool:
        return self._termination_reason() is not None

    def _is_upright(self, abs_angles: np.ndarray | None = None) -> bool:
        if abs_angles is None:
            _, abs_angles = self._angles()
        threshold = float(self.env_cfg.get("success_upright_threshold", self.env_cfg.get("reward", {}).get("upright_threshold", 0.10)))
        return bool(float(np.max(np.abs(abs_angles))) < threshold)

    def _update_upright_tracking(self) -> None:
        _, abs_angles = self._angles()
        reward_cfg = self.env_cfg.get("reward", {})
        qpos = self.data.qpos
        qvel = self.data.qvel
        hinge_vel_rms = float(np.sqrt(np.mean(qvel[1 : 1 + self.n] ** 2)))
        upright = self._is_upright(abs_angles)
        centered_max_cart_abs = reward_cfg.get("centered_upright_max_cart_abs", reward_cfg.get("low_momentum_max_cart_abs"))
        centered_upright = bool(
            upright
            and (
                centered_max_cart_abs is None
                or abs(float(qpos[0])) <= float(centered_max_cart_abs)
            )
        )
        low_momentum_upright = bool(
            centered_upright
            and hinge_vel_rms <= float(
                reward_cfg.get("low_momentum_streak_hinge_vel_threshold", reward_cfg.get("upright_hinge_vel_threshold", 1.0))
            )
            and abs(float(qvel[0])) <= float(
                reward_cfg.get("low_momentum_streak_cart_vel_threshold", reward_cfg.get("upright_cart_vel_threshold", 1.0))
            )
        )

        if upright:
            self.upright_streak_steps += 1
            if self.first_upright_step is None:
                self.first_upright_step = self.step_count
        else:
            self.upright_streak_steps = 0
        if centered_upright:
            self.centered_upright_streak_steps += 1
        else:
            self.centered_upright_streak_steps = 0
        if low_momentum_upright:
            self.low_momentum_upright_streak_steps += 1
        else:
            self.low_momentum_upright_streak_steps = 0
        self.max_upright_streak_steps = max(self.max_upright_streak_steps, self.upright_streak_steps)
        self.max_centered_upright_streak_steps = max(
            self.max_centered_upright_streak_steps,
            self.centered_upright_streak_steps,
        )
        self.max_low_momentum_upright_streak_steps = max(
            self.max_low_momentum_upright_streak_steps,
            self.low_momentum_upright_streak_steps,
        )
        sustain_steps = int(np.ceil(float(self.env_cfg.get("success_sustain_seconds", 0.0)) / self.dt))
        if self.capture_step is None and self.upright_streak_steps >= max(1, sustain_steps):
            self.capture_step = self.step_count
            self.capture_start_step = self.step_count - self.upright_streak_steps + 1

    def _success(self) -> bool:
        sustain_seconds = float(self.env_cfg.get("success_sustain_seconds", 0.0))
        sustain_steps = int(np.ceil(sustain_seconds / self.dt))
        return self.max_upright_streak_steps >= sustain_steps

    def _info(self) -> dict[str, Any]:
        rel, abs_angles = self._angles()
        first_upright_time = None if self.first_upright_step is None else float(self.first_upright_step * self.dt)
        capture_time = None if self.capture_step is None else float(self.capture_step * self.dt)
        capture_start_time = None if self.capture_start_step is None else float(max(0, self.capture_start_step) * self.dt)
        hinge_vel_rms = float(np.sqrt(np.mean(self.data.qvel[1 : 1 + self.n] ** 2)))
        max_abs_angle = float(np.max(np.abs(abs_angles)))
        return {
            "x": float(self.data.qpos[0]),
            "max_abs_angle": max_abs_angle,
            "mean_abs_angle": float(np.mean(np.abs(abs_angles))),
            "hinge_velocity_rms": hinge_vel_rms,
            "capture_quality": self._capture_quality(max_abs_angle=max_abs_angle, hinge_vel_rms=hinge_vel_rms),
            "is_upright": bool(self._is_upright(abs_angles)),
            "upright_streak_seconds": float(self.upright_streak_steps * self.dt),
            "max_upright_streak_seconds": float(self.max_upright_streak_steps * self.dt),
            "centered_upright_streak_seconds": float(self.centered_upright_streak_steps * self.dt),
            "max_centered_upright_streak_seconds": float(self.max_centered_upright_streak_steps * self.dt),
            "low_momentum_upright_streak_seconds": float(self.low_momentum_upright_streak_steps * self.dt),
            "max_low_momentum_upright_streak_seconds": float(self.max_low_momentum_upright_streak_steps * self.dt),
            "time_to_first_upright": first_upright_time,
            "time_to_capture": capture_time,
            "capture_start_time": capture_start_time,
            "max_cart_excursion": float(self.max_cart_excursion),
            "termination_reason": None,
            "progress": float(self.progress),
            "plant_progress": float(self.plant_progress),
            "rail_limit": float(self.rail_limit),
            "alpha_length": float(self.morphology.alpha_length),
            "alpha_mass": float(self.morphology.alpha_mass),
            "alpha_damping": float(self.morphology.alpha_damping),
            "alpha_frictionloss": float(self.morphology.alpha_frictionloss),
            "init_qvel_scale": float(self._init_qvel_scale()),
            "policy_action_norm": float(self.last_policy_action_norm[0]),
            "applied_action_norm": float(self.last_action_norm[0]),
            "action_bias_norm": float(self.last_action_bias_norm),
            "residual_scale": float(self.last_residual_scale),
            "lqr_cart_target": float(self.last_lqr_cart_target),
            "controller_mode": self.last_controller_mode,
            "lqr_switch_active": bool(self.lqr_switch_active),
            "lqr_switch_lyapunov_value": (
                self._lqr_switch_lyapunov_value(self.env_cfg["action_lqr_switch"])
                if self.env_cfg.get("action_lqr_switch", {}).get("enter_lyapunov_max") is not None
                else None
            ),
            "lqr_switch_entry_count": int(self.lqr_switch_entry_count),
            "lqr_switch_exit_count": int(self.lqr_switch_exit_count),
            "lqr_switch_first_entry_step": self.lqr_switch_first_entry_step,
            "lqr_switch_first_entry_time": (
                None
                if self.lqr_switch_first_entry_step is None
                else float(self.lqr_switch_first_entry_step * self.dt)
            ),
            "lqr_switch_lqr_steps": int(self.lqr_switch_lqr_steps),
            "lqr_switch_policy_steps": int(self.lqr_switch_policy_steps),
            "lengths": self.morphology.lengths.astype(float).tolist(),
            "masses": self.morphology.masses.astype(float).tolist(),
            "damping": self.morphology.damping.astype(float).tolist(),
            "frictionloss": self.morphology.frictionloss.astype(float).tolist(),
            "init_state_index": self.last_init_state_index,
        }

    def render(self):
        return self.render_rgb()

    def render_rgb(self, width: int = 1280, height: int = 720, camera: str = "side") -> np.ndarray:
        if self.renderer is None or self.renderer.width != width or self.renderer.height != height:
            self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
