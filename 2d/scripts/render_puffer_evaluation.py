"""Render a complete imported native evaluation without implying qualification."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from spatialpole import Config, SpatialCartpole
from spatialpole.experiment import write_json

p = argparse.ArgumentParser(); p.add_argument('--tape',required=True); p.add_argument('--out',required=True)
a = p.parse_args(); tape = Path(a.tape); out = Path(a.out)
assert not out.exists() and not out.with_suffix('.json').exists()
out.parent.mkdir(parents=True,exist_ok=True)
with np.load(tape) as z:
    states=z['states']; tilts=z['tilts']; actions=z['actions']
    config=Config(**json.loads(str(z['config']))); meta=json.loads(str(z['policy_metadata']))
assert states.shape[0]==len(actions)+1 and tilts.shape==(len(actions),config.n_links)
diagnostic=bool(meta.get('diagnostic',False))
has_checkpoint=bool(meta.get('checkpoint_sha256'))
status='DIAGNOSTIC CAPTURE' if diagnostic else 'DEVELOPMENT EVALUATION'
notice='NOT A TRAINED POLICY / NOT A QUALIFIED RESULT' if diagnostic or not has_checkpoint else 'RECORDED POLICY EXECUTION / DEVELOPMENT COHORT'
try:
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',22)
    title=ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf',32)
except OSError:
    font=title=ImageFont.load_default()
env=SpatialCartpole(config); writer=imageio.get_writer(str(out),fps=50,codec='libx264',quality=7,macro_block_size=16)
streak=best=0
try:
    for i,state in enumerate(states):
        env.set_state(state)
        frame=Image.fromarray(env.render());d=ImageDraw.Draw(frame)
        d.rectangle((0,0,1280,145),fill=(10,18,29))
        d.text((24,9),f'2D CARTPOLE / {config.n_links} LINKS / {status}',font=title,fill=(235,244,255))
        d.text((24,56),notice,font=font,fill=(255,195,90))
        if i:
            streak=streak+1 if np.max(tilts[i-1])<config.angle_threshold else 0
            best=max(best,streak)
        theta=float(np.max(env.tilts()))
        d.text((24,98),f't {i*config.dt:.2f}s   world tilt {theta:.3f} rad   best hold {best*config.dt:.2f}s   XY forces / free ball joints',font=font,fill=(150,220,245))
        x0,y0,side=1030,430,210
        d.rectangle((x0,y0,x0+side,y0+side),fill=(15,25,39),outline=(110,135,155),width=2)
        pts=[(x0+side/2+x/config.plane_limit*side/2,y0+side/2-y/config.plane_limit*side/2) for x,y in states[:i+1,:2]]
        if len(pts)>1:d.line(pts,fill=(65,170,235),width=2)
        x,y=pts[-1];d.ellipse((x-4,y-4,x+4,y+4),fill=(255,180,75))
        d.text((x0,y0-31),'XY plane / +/-3m',font=font,fill=(210,225,240))
        writer.append_data(np.asarray(frame))
        if i%250==0:print(f'render {i}/{len(states)}',flush=True)
finally:
    writer.close();env.close()
reader=imageio.get_reader(str(out)); metadata=reader.get_meta_data()
assert reader.count_frames()==len(states) and metadata['fps']==50
for index in sorted({0,min(75,len(states)-1),len(states)-1}):
    imageio.imwrite(out.with_name(out.stem+f'-frame{index:04d}.png'),reader.get_data(index))
reader.close()
report=dict(frames=len(states),fps=50,encoded_seconds=len(states)/50,
            simulated_seconds=(len(states)-1)*config.dt,links=config.n_links,
            full_tape_no_trimming=True,diagnostic=diagnostic,qualified=False,
            checkpoint_sha256=meta.get('checkpoint_sha256'),
            tape_sha256=hashlib.sha256(tape.read_bytes()).hexdigest(),
            video_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
write_json(out.with_suffix('.json'),report)
print(json.dumps(report),flush=True)
