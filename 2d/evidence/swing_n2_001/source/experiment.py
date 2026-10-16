from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .env import Config, SpatialCartpole
from .control import BalancePolicy, WorldBalancePolicy, RoutePolicy, balance_design, optimize_route


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def freeze_inputs(out, args):
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    shutil.copytree(source, out / "source", ignore=shutil.ignore_patterns("__pycache__"))
    memo = Path("research/plan.md")
    if memo.exists():
        shutil.copy(memo, out / "plan.md")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((out / "source").glob("*.py"))}
    write_json(out / "inputs.json", {"args": vars(args), "source_sha256": hashes,
               "utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
               "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "mujoco": mj.__version__}})


def episode(config, policy_factory, seed, mode="upright", noise=0.02, tape=None):
    env = SpatialCartpole(config)
    state, _ = env.reset(seed=seed, options={"mode": mode, "angle_noise": noise})
    policy = policy_factory(env)
    # Design/setup queries must not pollute actual integration counters/margins.
    env.set_state(state)
    env.native_steps, env.min_plane_margin, env.warning_seen = 0, config.plane_limit, False
    states, actions, raw_actions, tilts = [state], [], [], []
    start = time.perf_counter()
    for step in range(config.horizon):
        force = policy(state, step)
        delivered = env.deliver(force)
        state, _, terminated, truncated, info = env.step_force(delivered)
        states.append(state)
        raw_actions.append(force)
        actions.append(delivered)
        tilts.append(env.tilts())
        if terminated or truncated:
            break
    result = {"seed": seed, "initialization": mode, "angle_noise": noise, **info,
              "steps": env.steps, "native_steps": env.native_steps,
              "wall_seconds": time.perf_counter() - start,
              "max_xy": np.max(np.abs(np.asarray(states)[:, :2]), axis=0).tolist(),
              "max_force_norm": float(np.max(np.linalg.norm(actions, axis=1))),
              "final_tilt": float(np.max(tilts[-1]))}
    # Independent open-loop replay from every exact delivered command. This is
    # an integrity/reproducibility check, not another independent success sample.
    proof = SpatialCartpole(config)
    replay = [np.asarray(states)[0]]
    for action in actions:
        replay.append(proof.transition(replay[-1], action))
    result["replay_exact"] = bool(np.array_equal(np.asarray(states), np.array(replay)))
    result["replay_max_error"] = float(np.max(np.abs(np.asarray(states) - replay)))
    result["proof_native_steps"] = proof.native_steps
    if tape:
        np.savez_compressed(tape, states=states, actions=np.asarray(actions, np.float32), raw_actions=raw_actions,
                            tilts=tilts, config=json.dumps(asdict(config)))
    env.close()
    proof.close()
    return result


def render_tape(tape, output):
    data = np.load(tape)
    config = Config(**json.loads(str(data["config"])))
    env = SpatialCartpole(config)
    frames = data["states"]
    tilts = data["tilts"]
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 23)
        title_font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 34)
    except OSError:
        font = title_font = ImageFont.load_default()
    streak = best = 0
    writer = imageio.get_writer(str(output), fps=round(1/config.dt), codec="libx264", quality=7, macro_block_size=16)
    try:
        for i, state in enumerate(frames):
            env.set_state(state)
            img = Image.fromarray(env.render())
            d = ImageDraw.Draw(img)
            d.rectangle((0, 0, 1280, 114), fill=(10, 18, 29))
            d.text((30, 13), f"SPATIAL CARTPOLE  /  {config.n_links} {'LINK' if config.n_links == 1 else 'LINKS'}", font=title_font, fill=(235, 244, 255))
            max_tilt = float(np.max(env.tilts()))
            if i:
                streak = streak + 1 if np.max(tilts[i-1]) < config.angle_threshold else 0
                best = max(best, streak)
            text = f"t {i*config.dt:5.2f}s    tilt {max_tilt:5.3f} rad    best hold {best*config.dt:5.2f}s    XY cart + passive ball joints"
            d.text((30, 65), text, font=font, fill=(150, 220, 245))
            # Top-down inset proves both dimensions are physically free, and
            # shows the complete cart path rather than a camera-tracked illusion.
            x0, y0, side = 1030, 430, 210
            d.rectangle((x0, y0, x0+side, y0+side), fill=(15, 25, 39), outline=(110, 135, 155), width=2)
            path = frames[:i+1, :2]
            pts = [(x0+side/2+p[0]/config.plane_limit*side/2, y0+side/2-p[1]/config.plane_limit*side/2) for p in path]
            if len(pts)>1:
                d.line(pts, fill=(65, 170, 235), width=2)
            px, py = pts[-1]
            d.ellipse((px-4, py-4, px+4, py+4), fill=(255, 180, 75))
            d.text((x0, y0-31), "XY plane / +/-3m", font=font, fill=(210, 225, 240))
            writer.append_data(np.asarray(img))
            if i % 250 == 0:
                print(f"render {i}/{len(frames)}", flush=True)
    finally:
        writer.close()
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["balance", "swing", "video"])
    p.add_argument("--links", type=int, default=1)
    p.add_argument("--max-links", type=int, default=8)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--route-seconds", type=float, default=8.)
    p.add_argument("--amplitude", type=float, default=12.)
    p.add_argument("--frequency", type=float, default=0.45)
    p.add_argument("--out", required=True)
    p.add_argument("--route")
    p.add_argument("--tape")
    p.add_argument("--world-feedback", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    if a.task == "video":
        render_tape(a.tape, out)
        return
    freeze_inputs(out, a)
    start = time.perf_counter()
    if a.task == "balance":
        summary = []
        for n in range(a.links, a.max_links + 1):
            config = Config(n_links=n)
            env = SpatialCartpole(config)
            try:
                design = balance_design(env)
            except Exception as exc:
                write_json(out / f"n{n}_design_failure.json", {"type": type(exc).__name__, "error": str(exc)})
                print(f"n={n} design failed: {exc}", flush=True)
                break
            np.savez_compressed(out / f"n{n}_controller.npz", **design, config=json.dumps(asdict(config)))
            rows = []
            for k in range(a.episodes):
                cls = WorldBalancePolicy if a.world_feedback else BalancePolicy
                result = episode(config, lambda e: cls(e, design), a.seed+k, noise=a.noise,
                                 tape=out / f"n{n}_seed{a.seed+k}.npz")
                rows.append(result)
            row = {"n_links": n, "passes": sum(r["success"] and r["replay_exact"] for r in rows),
                   "episodes": a.episodes, "closed_loop_radius": design["closed_loop_radius"], "results": rows,
                   "design_native_steps": env.native_steps}
            summary.append(row)
            write_json(out / "summary.json", summary)
            print(json.dumps({k:v for k,v in row.items() if k != "results"}), flush=True)
            if row["passes"] != a.episodes:
                break
    else:
        config = Config(n_links=a.links)
        env = SpatialCartpole(config)
        initial, _ = env.reset(seed=a.seed, options={"mode":"hanging", "angle_noise":0.})
        H = round(a.route_seconds / config.dt)
        if a.route:
            saved = np.load(a.route)
            actions = saved["actions"].copy()
        else:
            t = np.arange(H) * config.dt
            actions = np.zeros((H, 2))
            actions[:, 0] = a.amplitude * np.sin(2*np.pi*a.frequency*t)
        def log(row):
            with (out / "optimizer.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row)+"\n")
            if row["iteration"] % 5 == 0:
                print(json.dumps(row), flush=True)
        states, actions, gains, design, history = optimize_route(env, initial, actions, a.iterations, log)
        np.savez_compressed(out / "route.npz", states=states, actions=actions, gains=gains,
                            **design, config=json.dumps(asdict(config)))
        native_design = env.native_steps
        factory = lambda e: RoutePolicy(e, states, actions, gains, design)
        rows = [episode(config, factory, a.seed, "hanging", 0., out / "nominal.npz")]
        print("nominal: " + json.dumps(rows[0]), flush=True)
        for k in range(a.episodes):
            rows.append(episode(config, factory, a.seed+100+k, "hanging", a.noise, out / f"validation_{k}.npz"))
        write_json(out / "summary.json", {"n_links":a.links, "design_native_steps":native_design,
                   "optimizer_iterations":len(history), "results":rows,
                   "passes":sum(r["success"] and r["replay_exact"] for r in rows), "wall_seconds":time.perf_counter()-start})
    write_json(out / "completed.json", {"wall_seconds":time.perf_counter()-start, "status":"completed"})


if __name__ == "__main__":
    main()
