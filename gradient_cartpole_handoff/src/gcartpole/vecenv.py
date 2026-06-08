from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np

from .env import NLinkCartPoleEnv


class SerialVectorEnv:
    def __init__(self, cfg: dict[str, Any], num_envs: int, seed: int = 0, progress: float = 0.0):
        self.envs = [NLinkCartPoleEnv(cfg, progress=progress, seed=seed + i) for i in range(num_envs)]
        self.num_envs = num_envs
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

    def reset(self):
        obs = []
        infos = []
        for env in self.envs:
            o, info = env.reset()
            obs.append(o)
            infos.append(info)
        return np.stack(obs).astype(np.float32), infos

    def step(self, actions: np.ndarray):
        obs, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            o, r, term, trunc, info = env.step(action)
            done = bool(term or trunc)
            if done:
                final_obs = o.copy()
                final_info = dict(info)
                o, reset_info = env.reset()
                info = dict(reset_info)
                info["final_observation"] = final_obs
                info["final_info"] = final_info
            obs.append(o)
            rewards.append(r)
            dones.append(done)
            infos.append(info)
        return (
            np.stack(obs).astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def set_progress(self, progress: float):
        for env in self.envs:
            env.set_progress(progress)
        return self.reset()

    def close(self):
        for env in self.envs:
            env.close()


def _worker(remote, cfg: dict[str, Any], seed: int, progress: float):
    env = None
    try:
        env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs, info = env.reset(seed=data)
                remote.send((obs, info))
            elif cmd == "step":
                obs, reward, term, trunc, info = env.step(data)
                done = bool(term or trunc)
                if done:
                    final_obs = obs.copy()
                    final_info = dict(info)
                    obs, reset_info = env.reset()
                    info = dict(reset_info)
                    info["final_observation"] = final_obs
                    info["final_info"] = final_info
                remote.send((obs, float(reward), done, info))
            elif cmd == "set_progress":
                env.set_progress(float(data))
                obs, info = env.reset()
                remote.send((obs, info))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown worker command: {cmd}")
    except KeyboardInterrupt:
        pass
    finally:
        if env is not None:
            env.close()


class SubprocessVectorEnv:
    def __init__(self, cfg: dict[str, Any], num_envs: int, seed: int = 0, progress: float = 0.0):
        self.num_envs = int(num_envs)
        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for i, work_remote in enumerate(self.work_remotes):
            proc = ctx.Process(target=_worker, args=(work_remote, cfg, seed + i, progress), daemon=True)
            proc.start()
            self.processes.append(proc)
            work_remote.close()
        probe = NLinkCartPoleEnv(cfg, progress=progress, seed=seed + 999_000)
        self.single_observation_space = probe.observation_space
        self.single_action_space = probe.action_space
        probe.close()

    def reset(self):
        for i, remote in enumerate(self.remotes):
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs).astype(np.float32), list(infos)

    def step(self, actions: np.ndarray):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", np.asarray(action, dtype=np.float32)))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return (
            np.stack(obs).astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            list(infos),
        )

    def set_progress(self, progress: float):
        for remote in self.remotes:
            remote.send(("set_progress", float(progress)))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs).astype(np.float32), list(infos)

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for proc in self.processes:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()


def make_vec_env(cfg: dict[str, Any], num_envs: int, seed: int, progress: float = 0.0, backend: str = "serial"):
    if backend == "serial":
        return SerialVectorEnv(cfg, num_envs, seed=seed, progress=progress)
    if backend == "subprocess":
        return SubprocessVectorEnv(cfg, num_envs, seed=seed, progress=progress)
    raise ValueError(f"Unknown vec backend: {backend}")
