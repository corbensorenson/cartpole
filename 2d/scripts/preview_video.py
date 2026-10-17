"""Create a normal-speed GIF and inspectable contact sheets from a full video."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('--video',required=True)
p.add_argument('--prefix',required=True);a=p.parse_args()
video,prefix=Path(a.video),Path(a.prefix)
outputs=[Path(str(prefix)+suffix) for suffix in ('-preview.gif','-contact.png','-swing-contact.png','-contact.json')]
assert all(not path.exists() for path in outputs),'video QA outputs are create-only'
full_ids=[0,250,500,750,950,960,1000,1100,1200,1242,1350,1500]
swing_ids=[960,1000,1050,1100,1150,1200,1242,1300,1500]
selected={};preview=[];count=0
reader=imageio.get_reader(video);meta=reader.get_meta_data()
try:
    for i,frame in enumerate(reader):
        assert frame.shape==(720,1280,3)
        img=Image.fromarray(frame)
        if i in full_ids or i in swing_ids:
            selected[i]=img.resize((426,240),Image.Resampling.LANCZOS)
        if i<1500 and i%5==0:
            preview.append(img.resize((640,360),Image.Resampling.LANCZOS).quantize(colors=256))
        count+=1
finally:
    reader.close()
assert count==1501 and meta['fps']==50.
assert len(preview)==300
preview[0].save(outputs[0],save_all=True,append_images=preview[1:],duration=100,loop=0,disposal=2)
for ids,target in ((full_ids,outputs[1]),(swing_ids,outputs[2])):
    sheet=Image.new('RGB',(1278,240*((len(ids)+2)//3)),(10,18,29))
    for index,key in enumerate(ids):
        sheet.paste(selected[key],((index%3)*426,(index//3)*240))
    sheet.save(target)
with Image.open(outputs[0]) as gif:
    duration=0
    for i in range(gif.n_frames):
        gif.seek(i);duration+=gif.info['duration']
    assert gif.n_frames==300 and duration==30000
    gif_frames=gif.n_frames
report=dict(video_frames=count,fps=meta['fps'],encoded_duration=count/meta['fps'],
    video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),preview_frames=gif_frames,
    preview_duration_ms=duration,preview_sha256=hashlib.sha256(outputs[0].read_bytes()).hexdigest(),
    contact_frame_indices=full_ids,swing_contact_frame_indices=swing_ids)
outputs[3].write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
