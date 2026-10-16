"""Make a full-duration GitHub preview and a video inspection contact sheet."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image,ImageDraw

p = argparse.ArgumentParser()
p.add_argument('--video',required=True)
p.add_argument('--out',required=True)
p.add_argument('--qa-out',required=True)
a = p.parse_args()
video,out,qa = Path(a.video),Path(a.out),Path(a.qa_out)
assert video.is_file() and not out.exists() and not qa.exists()
reader = imageio.get_reader(video)
metadata = reader.get_meta_data()
count = reader.count_frames()
fps = metadata['fps']
assert count==1501 and fps==50 and tuple(metadata['size'])==(1280,720)
sheet = Image.new('RGB',(960,3*292),(10,18,29))
for index,frame in enumerate((0,400,900,1140,1250,1500)):
    im = Image.fromarray(reader.get_data(frame)).resize((480,270),Image.Resampling.LANCZOS)
    x,y = (index%2)*480,(index//2)*292
    sheet.paste(im,(x,y))
    ImageDraw.Draw(sheet).text((x+10,y+273),f'{frame/fps:.2f}s',fill=(240,240,240))
sheet.save(qa)
reader.close()
filters = 'fps=10,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer'
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error',
    '-i',str(video),'-filter_complex',filters,'-loop','0',str(out)],check=True)
with Image.open(out) as gif:
    duration = 0
    for i in range(gif.n_frames):
        gif.seek(i)
        duration += gif.info['duration']
    assert 29900<=duration<=30200
    gif_frames = gif.n_frames
result = {'video_frames':count,'fps':fps,'encoded_duration':count/fps,
          'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
          'preview_frames':gif_frames,'preview_duration_ms':duration,
          'preview_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
qa.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
