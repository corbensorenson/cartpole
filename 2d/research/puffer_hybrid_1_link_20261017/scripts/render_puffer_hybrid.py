"""Full first fresh hybrid episode, with the two controller roles visible."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spatialpole import Config, SpatialCartpole
from spatialpole.experiment import write_json


ap = argparse.ArgumentParser()
ap.add_argument('--qualification', required=True)
ap.add_argument('--audit', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
run, out = Path(a.qualification), Path(a.out)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit = json.loads(Path(a.audit).read_text())
assert audit['passed'] and audit['qualified_hybrid_swing_up']
assert audit['qualification_summary_sha256'] == sha(run/'summary.json')
summary = json.loads((run/'summary.json').read_text())
assert summary['passes'] == len(summary['results']) == 32
assert not out.exists() and not out.with_suffix('.json').exists()
out.parent.mkdir(parents=True, exist_ok=True)
tape = run/'seed20330116.npz'
with np.load(tape) as z:
    states, tilts, actions = z['states'], z['tilts'], z['actions']
    config = Config(**json.loads(str(z['config'])))
    metadata = json.loads(str(z['policy_metadata']))
assert config == Config(n_links=1) and states.shape == (1501, 11) and actions.shape == (1500, 2)
assert metadata['hybrid_handoff_step'] == 400 and metadata['handoff_actual_seconds'] == 8.
font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 22)
title = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 32)
env = SpatialCartpole(config)
writer = imageio.get_writer(str(out), fps=50, codec='libx264', quality=7, macro_block_size=16)
best = streak = 0
try:
    for i, state in enumerate(states):
        env.set_state(state)
        frame = Image.fromarray(env.render())
        d = ImageDraw.Draw(frame)
        d.rectangle((0, 0, 1280, 145), fill=(10, 18, 29))
        d.text((24, 9), '2D CARTPOLE / 1 LINK / QUALIFIED HYBRID', font=title, fill=(235, 244, 255))
        d.text((24, 56), 'CLASSICAL SWING-UP + LEARNED PUFFER CAPTURE / 32/32 FRESH STARTS', font=font, fill=(255, 195, 90))
        if i:
            streak = streak+1 if np.max(tilts[i-1]) < .15 else 0
            best = max(best, streak)
        phase = 'CLASSICAL SWING' if i < 400 else 'LEARNED CAPTURE / BALANCE'
        d.text((24, 98), f't {i*.02:.2f}s   tilt {env.tilts().max():.3f} rad   best hold {best*.02:.2f}s   {phase}',
               font=font, fill=(150, 220, 245))
        x0, y0, side = 1030, 430, 210
        d.rectangle((x0, y0, x0+side, y0+side), fill=(15, 25, 39), outline=(110, 135, 155), width=2)
        points = [(x0+side/2+x/3*side/2, y0+side/2-y/3*side/2) for x, y in states[:i+1, :2]]
        if len(points) > 1:
            d.line(points, fill=(65, 170, 235), width=2)
        x, y = points[-1]
        d.ellipse((x-4, y-4, x+4, y+4), fill=(255, 180, 75))
        d.text((x0, y0-31), 'XY plane / +/-3m', font=font, fill=(210, 225, 240))
        writer.append_data(np.asarray(frame))
        if i % 250 == 0:
            print(f'render {i}/1501', flush=True)
finally:
    writer.close()
    env.close()
reader = imageio.get_reader(str(out))
assert reader.count_frames() == 1501 and reader.get_meta_data()['fps'] == 50
for i in (0, 400, 1500):
    imageio.imwrite(out.with_name(out.stem+f'-frame{i:04d}.png'), reader.get_data(i))
reader.close()
write_json(out.with_suffix('.json'), dict(frames=1501, fps=50, simulated_seconds=30., encoded_seconds=30.02,
           first_fresh_seed=20330116, qualified_hybrid=True, pure_RL_swing_up=False,
           native_steps=0, no_trimming=True, video_sha256=sha(out), tape_sha256=sha(tape),
           controller_checkpoint_sha256=metadata['checkpoint_sha256'], audit_sha256=sha(a.audit),
           renderer_sha256=sha(__file__)))
print(json.dumps(dict(video=str(out), sha256=sha(out))), flush=True)
