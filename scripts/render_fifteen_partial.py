"""Render every saved state of a failed nominal15 attempt, without simulation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont

FONT=Path('/System/Library/Fonts/Supplemental/Arial.ttf')
def font(size):
 try:return ImageFont.truetype(str(FONT),size)
 except OSError:return ImageFont.load_default(size=size)
def frame(state,action,index,streak,best):
 w,h=1280,900;im=Image.new('RGB',(w,h),(10,17,30));d=ImageDraw.Draw(im);scale=82.;origin=(w/2,440)
 def xy(x,y):return (origin[0]+x*scale,origin[1]-y*scale)
 angles=np.cumsum(state[1:16]);wrapped=(angles+np.pi)%(2*np.pi)-np.pi;maxangle=float(np.max(abs(wrapped)));failed=abs(state[0])>3
 d.text((44,24),'15 LINKS  /  PARTIAL RESULT',font=font(31),fill=(240,244,250));d.text((44,67),'Best recorded upright hold: 3.00 s     |     UNSOLVED',font=font(22),fill=(255,170,100))
 d.text((44,104),'Exact nominal hanging start | Recorded simulation states | 50 Hz commands',font=font(17),fill=(151,170,195))
 d.text((950,28),f't = {index/50:5.2f} s',font=font(25),fill='white')
 for x in range(-6,7):
  xx=xy(x,0)[0];d.line((xx,160,xx,725),fill=(22,35,53),width=1)
 rail_y=xy(0,0)[1]+21;d.line((xy(-3,0)[0],rail_y,xy(3,0)[0],rail_y),fill=(107,126,151),width=6)
 for x in [-3,3]:
  xx=xy(x,0)[0];d.line((xx,rail_y-14,xx,rail_y+18),fill=(255,91,101),width=4);d.text((xx-22,rail_y+23),f'{x:+d} m',font=font(17),fill=(255,140,145))
 cart=xy(float(state[0]),0);d.rounded_rectangle((cart[0]-24,cart[1]-12,cart[0]+24,cart[1]+20),radius=5,fill=(227,237,247))
 x,y=float(state[0]),0.;colors=[(81+int(120*i/14),194-int(50*i/14),242-int(100*i/14)) for i in range(15)]
 for j,a in enumerate(angles):
  nx=x+.2*np.sin(a);ny=y+.2*np.cos(a);p=xy(x,y);q=xy(nx,ny);d.line((*p,*q),fill=colors[j],width=4);d.ellipse((p[0]-2.5,p[1]-2.5,p[0]+2.5,p[1]+2.5),fill='white');x,y=nx,ny
 q=xy(x,y);d.ellipse((q[0]-3,q[1]-3,q[0]+3,q[1]+3),fill=(255,223,105))
 d.rectangle((28,756,w-28,h-24),fill=(18,30,48));d.text((48,771),f'Cart: {state[0]:+.4f} m    Applied force: {action*80:+.2f} N    Max absolute link angle: {maxangle:.4f} rad',font=font(18),fill=(223,234,247))
 d.text((48,804),f'Current upright streak: {streak/50:.2f} s    Best so far: {best/50:.2f} s    Required: 5.00 s + full 30 s episode',font=font(18),fill=(223,234,247))
 status='RAIL LIMIT EXCEEDED — FAILED ATTEMPT' if failed else 'Development example; no accepted 15-link record'
 d.text((48,839),status,font=font(20),fill=(255,105,115) if failed else (255,177,104));return np.asarray(im)

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkout',default=str(Path(__file__).resolve().parents[1]));a=p.parse_args();root=Path(a.checkout);out=root/'runs/fifteen_link_partial_20261015';tape=out/'actual_minimax.npz'
 with np.load(tape) as z:states=z['states'];actions=z['actions']
 video=out/'fifteen_link_partial_full_attempt.mp4';streak=best=0;frames=[]
 with imageio.get_writer(video,fps=50,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-pix_fmt','yuv420p']) as writer:
  for i,s in enumerate(states):
   if i:
    angles=np.cumsum(s[1:16]);angles=np.where((angles>=-np.pi)&(angles<np.pi),angles,(angles+np.pi)%(2*np.pi)-np.pi);streak=streak+1 if np.max(abs(angles))<.15 else 0;best=max(best,streak)
   f=frame(s,0. if not i else actions[i-1],i,streak,best);writer.append_data(f)
   if i in [0,441,len(states)-1]:Image.fromarray(f).save(out/f'video_frame_{i:04d}.png')
   if i%2==0:frames.append(Image.fromarray(f).resize((640,450)))
 assets=root/'assets';assets.mkdir(exist_ok=True);frames[0].save(assets/'fifteen-link-partial.gif',save_all=True,append_images=frames[1:],duration=40,loop=0,optimize=False)
 meta=dict(status='partial_unsolved',state_tape_sha256=hashlib.sha256(tape.read_bytes()).hexdigest(),video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),states_rendered=len(states),fps=50,first_state_time_seconds=0.,last_state_time_seconds=(len(states)-1)/50,video_duration_seconds=len(states)/50,all_states_in_original_order=True,skipped_states=0,new_native_transition_calls=0,source='Complete recorded minimax failed episode, including final out-of-rail state. Schematic geometry follows recorded cumulative joint angles; no new physics, interpolated state, resets or teleports.')
 (out/'video_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
if __name__=='__main__':main()
