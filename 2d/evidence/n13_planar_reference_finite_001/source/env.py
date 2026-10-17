from __future__ import annotations

from dataclasses import asdict, dataclass
import gymnasium as gym
from gymnasium import spaces
import mujoco as mj
import numpy as np


@dataclass(frozen=True)
class Config:
    n_links: int = 1
    total_length: float = 3.0
    total_mass: float = 1.0
    cart_mass: float = 1.0
    cart_armature: float = 0.0
    plane_limit: float = 3.0
    force_limit: float = 80.0  # Euclidean force budget, not 80 N independently per axis
    timestep: float = 0.005
    frame_skip: int = 4
    episode_seconds: float = 30.0
    angle_threshold: float = 0.15
    sustain_seconds: float = 5.0
    joint_damping: float = 0.0075  # total, divided equally among links
    armature: float = 0.0005
    link_radius: float = 0.025

    def __post_init__(self):
        if not isinstance(self.n_links, int) or self.n_links < 1:
            raise ValueError("n_links must be a positive integer")
        if not isinstance(self.frame_skip, int) or self.frame_skip < 1:
            raise ValueError("frame_skip must be a positive integer")
        for key in ("total_length", "total_mass", "cart_mass", "plane_limit", "force_limit", "timestep", "episode_seconds", "angle_threshold", "sustain_seconds", "link_radius"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ('joint_damping','armature','cart_armature'):
            if not np.isfinite(getattr(self,key)) or getattr(self,key)<0:
                raise ValueError(f'{key} must be finite and nonnegative')

    @property
    def dt(self):
        return self.timestep * self.frame_skip

    @property
    def horizon(self):
        return round(self.episode_seconds / self.dt)


def model_xml(c: Config) -> str:
    length, mass = c.total_length / c.n_links, c.total_mass / c.n_links
    lines = [f'''<mujoco model="spatial_cartpole_{c.n_links}">
      <compiler angle="radian"/>
      <option timestep="{c.timestep}" gravity="0 0 -9.81" integrator="RK4"/>
      <visual><global offwidth="1280" offheight="720"/><headlight ambient="0.4 0.4 0.4"/></visual>
      <default><geom contype="0" conaffinity="0"/><joint limited="false"/></default>
      <asset><texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.11 0.16 0.23" rgb2="0.15 0.21 0.29"/>
      <material name="floor" texture="grid" texrepeat="12 12" texuniform="true" reflectance="0.1"/></asset>
      <worldbody><light pos="0 -4 8" dir="0 0 -1"/>
      <geom name="floor" type="plane" pos="0 0 -0.09" size="6 6 0.1" material="floor"/>
      <body name="cart"><joint name="cart_x" type="slide" axis="1 0 0" damping="0.02" armature="{c.cart_armature}"/>
      <joint name="cart_y" type="slide" axis="0 1 0" damping="0.02" armature="{c.cart_armature}"/>
      <geom type="box" size="0.18 0.18 0.07" mass="{c.cart_mass}" rgba="0.12 0.6 0.95 1"/>''']
    # The floor is visual only; the chain may swing through z < 0, as in the
    # original no-contact cartpole. No joint limits, equalities, springs or motors.
    for i in range(c.n_links):
        color = "1 0.4 0.18 1" if i % 2 == 0 else "1 0.78 0.15 1"
        lines.append(f'''<body name="link_{i}" pos="0 0 {0.07 if i == 0 else length}">
          <joint name="ball_{i}" type="ball" damping="{c.joint_damping / c.n_links}" armature="{c.armature}"/>
          <geom name="rod_{i}" type="capsule" fromto="0 0 0 0 0 {length}" size="{c.link_radius}" mass="{mass}" rgba="{color}"/>
          <site name="tip_{i}" pos="0 0 {length}" size="0.035" rgba="0.8 0.9 1 1"/>''')
    lines += ["</body>" * (c.n_links + 1), "</worldbody>",
              f'''<actuator><motor name="force_x" joint="cart_x" gear="1" ctrllimited="true" ctrlrange="-{c.force_limit} {c.force_limit}"/>
              <motor name="force_y" joint="cart_y" gear="1" ctrllimited="true" ctrlrange="-{c.force_limit} {c.force_limit}"/></actuator></mujoco>''']
    return "\n".join(lines)


class SpatialCartpole(gym.Env):
    """Full-state Gymnasium environment. Actions are normalized XY forces.

    Upright is measured by each rod's WORLD direction, not local joint angles.
    Joint twist is free and is not an upright error. Boundary exits terminate;
    joint/ground contacts do not enforce a hidden support or balancing torque.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, config: Config = Config(), render_mode=None):
        self.config, self.render_mode = config, render_mode
        self.xml = model_xml(config)
        self.model = mj.MjModel.from_xml_string(self.xml)
        self.data = mj.MjData(self.model)
        self.action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.model.nq + self.model.nv,), dtype=np.float64)
        self.link_ids = np.array([self.model.body(f"link_{i}").id for i in range(config.n_links)])
        self.steps = self.streak = self.best_streak = self.native_steps = 0
        self.renderer = None
        self.min_plane_margin = config.plane_limit
        self.warning_seen = False

    def state(self):
        return np.r_[self.data.qpos, self.data.qvel].copy()

    def set_state(self, state):
        # Reset all integration scratch state, preventing dependence on preceding
        # optimization queries. No contacts, activation, or external forces exist.
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] = state[:self.model.nq]
        self.data.qvel[:] = state[self.model.nq:]
        mj.mj_forward(self.model, self.data)

    def difference(self, state, reference):
        dq = np.empty(self.model.nv)
        mj.mj_differentiatePos(self.model, dq, 1., reference[:self.model.nq], state[:self.model.nq])
        return np.r_[dq, state[self.model.nq:] - reference[self.model.nq:]]

    def integrate(self, state, delta):
        result = state.copy()
        mj.mj_integratePos(self.model, result[:self.model.nq], delta[:self.model.nv], 1.)
        result[self.model.nq:] += delta[self.model.nv:]
        return result

    def upright_state(self):
        return np.r_[self.model.qpos0, np.zeros(self.model.nv)]

    def deliver(self, force):
        force = np.asarray(force, dtype=np.float64)
        if force.shape != (2,) or not np.isfinite(force).all():
            raise ValueError("force must contain two finite components")
        force = force * min(1., self.config.force_limit / max(np.linalg.norm(force), 1e-30))
        delivered = force.astype(np.float32)
        # Avoid a rounded vector exceeding the Euclidean limit.
        while np.linalg.norm(delivered.astype(float)) > self.config.force_limit:
            delivered = np.nextafter(delivered, np.float32(0))
        return delivered

    def transition(self, state, force, quantize=True):
        self.set_state(state)
        self.data.ctrl[:] = self.deliver(force) if quantize else force
        for _ in range(self.config.frame_skip):
            mj.mj_step(self.model, self.data)
            self.native_steps += 1
            self.min_plane_margin = min(self.min_plane_margin, self.config.plane_limit - float(np.max(np.abs(self.data.qpos[:2]))))
            self.warning_seen |= any(w.number > 0 for w in self.data.warning)
        # RK4 Cartesian body transforms can refer to an intermediate stage.
        mj.mj_forward(self.model, self.data)
        return self.state()

    def tilts(self):
        directions = self.data.xmat[self.link_ids].reshape(-1, 3, 3)[:, :, 2]
        return np.arccos(np.clip(directions[:, 2], -1., 1.))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        state = self.upright_state()
        mode = options.get("mode", "upright")
        if mode == "hanging":
            state[2:6] = [0., 0., 1., 0.]
        elif mode != "upright":
            raise ValueError("mode must be upright or hanging")
        noise = float(options.get("angle_noise", 0.02))
        delta = np.zeros(2 * self.model.nv)
        delta[:2] = self.np_random.uniform(-0.05, 0.05, 2) if noise else 0
        delta[2:self.model.nv] = self.np_random.uniform(-noise, noise, self.model.nv - 2)
        delta[self.model.nv:] = self.np_random.uniform(-2 * noise, 2 * noise, self.model.nv)
        self.set_state(self.integrate(state, delta))
        self.steps = self.streak = self.best_streak = self.native_steps = 0
        self.min_plane_margin, self.warning_seen = self.config.plane_limit, False
        return self.state(), self.info()

    def info(self):
        finite = np.isfinite(self.state()).all()
        return {"max_tilt": float(np.max(self.tilts())), "best_hold_seconds": self.best_streak * self.config.dt,
                "plane_margin": self.min_plane_margin, "physics_warning": self.warning_seen,
                "success": bool(self.steps == self.config.horizon and self.best_streak * self.config.dt >= self.config.sustain_seconds and self.min_plane_margin >= 0 and finite and not self.warning_seen)}

    def step_force(self, force):
        state = self.transition(self.state(), force)
        self.steps += 1
        tilt = float(np.max(self.tilts()))
        self.streak = self.streak + 1 if tilt < self.config.angle_threshold else 0
        self.best_streak = max(self.best_streak, self.streak)
        terminated = self.min_plane_margin < 0 or not np.isfinite(state).all() or self.warning_seen
        truncated = self.steps >= self.config.horizon
        reward = float(np.mean(np.cos(self.tilts())) - 0.03 * np.sum(state[:2] ** 2) - 0.001 * np.sum(self.data.ctrl ** 2))
        return state, reward, bool(terminated), bool(truncated), self.info()

    def step(self, action):
        return self.step_force(np.asarray(action, dtype=float) * self.config.force_limit)

    def render(self):
        if self.renderer is None:
            self.renderer = mj.Renderer(self.model, height=720, width=1280)
        cam = mj.MjvCamera()
        cam.lookat[:] = [0, 0, 0.5]
        cam.distance, cam.azimuth, cam.elevation = 11., 130., -18.
        self.renderer.update_scene(self.data, camera=cam)
        floor_id = self.model.geom("floor").id
        for geom in self.renderer.scene.geoms[:self.renderer.scene.ngeom]:
            if geom.objtype == mj.mjtObj.mjOBJ_GEOM and geom.objid == floor_id:
                geom.rgba[3] = 0.12  # hanging links must remain visible below XY
        return self.renderer.render()

    def close(self):
        if hasattr(self,'_fd_rollout'):
            self._fd_rollout.close()
            del self._fd_rollout
            del self._fd_data
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
