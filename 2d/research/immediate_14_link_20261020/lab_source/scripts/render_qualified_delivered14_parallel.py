"""Full first-reserved delivered14 video, admitted only after fresh32 qualification."""
import argparse
import itertools
import json
from pathlib import Path
import shutil
from qualified_delivered_source14_parallel import first_qualified,read,sha

DISTANCE=12.4


def render(env):
    if env.renderer is None:env.renderer=mj.Renderer(env.model,height=720,width=1280)
    camera=mj.MjvCamera();camera.lookat[:]=[0,0,.5]
    camera.distance,camera.azimuth,camera.elevation=DISTANCE,130.,-18.
    env.renderer.update_scene(env.data,camera=camera)
    floor=env.model.geom('floor').id
    for geom in env.renderer.scene.geoms[:env.renderer.scene.ngeom]:
        if geom.objtype==mj.mjtObj.mjOBJ_GEOM and geom.objid==floor:geom.rgba[3]=.12
    return env.renderer.render()


def main():
    p=argparse.ArgumentParser();p.add_argument('--proof',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();admission=first_qualified(a.proof);out=Path(a.out);s=admission['witness'];run,audit=Path(s['run']),Path(s['audit'])
    global mj
    import imageio.v2 as imageio
    import mujoco as mj
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from spatialpole import Config,SpatialCartpole
    from spatialpole.experiment import write_json
    assert sha(run/'summary.json')==s['summary_sha256'] and sha(audit/'summary.json')==s['audit_summary_sha256']
    assert read(audit/'summary.json')['task_success'] and sha(run/'actual.npz')==s['actual_sha256']
    assert not out.parent.exists();out.parent.mkdir(parents=True);shutil.copyfile(Path(__file__),out.parent/'render_driver.py')
    with np.load(run/'actual.npz') as z:tape={k:z[k] for k in z.files}
    config=Config(**json.loads(str(tape['config'])));assert config==Config(n_links=14)
    assert tape['states'].shape==(1501,102) and tape['actions'].shape==(1500,2)
    assert read(run/'frozen_manifest.json')['protocol']=='common-original-delivery14-fresh-canonical-witness-v1'
    mode=read(run/'frozen_manifest.json')['mode'];assert mode in ('fixed','native_control')
    write_json(out.parent/'qualified_source_admission.json',admission)
    env=SpatialCartpole(config)
    try:
        env.set_state(tape['states'][0]);render(env)
        camera=mj.mjv_averageCamera(env.renderer.scene.camera[0],env.renderer.scene.camera[1])
        eye,up,forward=map(lambda x:np.asarray(x,float),(camera.pos,camera.up,camera.forward));right=np.cross(forward,up)
        extent=camera.frustum_top-camera.frustum_bottom;near=camera.frustum_near
        cart=int(env.model.body('cart').id);cart_geom=int(np.flatnonzero(env.model.geom_bodyid==cart)[0]);rows=[]
        corners=np.asarray(list(itertools.product((-1,1),repeat=3)))
        for state in tape['states']:
            env.set_state(state);roots=env.data.xpos[env.link_ids].copy();R=env.data.xmat[env.link_ids[-1]].reshape(3,3)
            rods=np.vstack([roots,roots[-1]+R[:,2]*(3/14)])
            box=env.data.geom_xpos[cart_geom]+(corners*env.model.geom_size[cart_geom])@env.data.geom_xmat[cart_geom].reshape(3,3).T
            points=np.vstack([rods,box]);radii=np.r_[np.full(len(rods),.035),np.zeros(8)]
            q=points-eye;depth=q@forward;assert np.all(depth>radii)
            h,v=q@right,q@up;px=640+720*near*h/depth/extent;py=720*(camera.frustum_top-near*v/depth)/extent
            padx=720*near*radii*(depth+abs(h))/(depth*(depth-radii))/extent
            pady=720*near*radii*(depth+abs(v))/(depth*(depth-radii))/extent
            rows.append([np.min(px-padx),np.max(px+padx),np.min(py-pady),np.max(py+pady)])
        bounds=np.asarray(rows);np.savez_compressed(out.parent/'projected_body_bounds.npz',bounds=bounds)
        assert bounds[:,0].min()>=25 and bounds[:,1].max()<=1015
        assert bounds[:,2].min()>=153 and bounds[:,3].max()<=629,'Keep this failed framing proof and select a new render directory'
        highest,lowest=int(bounds[:,2].argmin()),int(bounds[:,3].argmax())
        write_json(out.parent/'framing_proof.json',dict(passed=True,native_steps=0,camera_distance=DISTANCE,frames=1501,
            minimum_projected_y=float(bounds[:,2].min()),maximum_projected_y=float(bounds[:,3].max()),highest_frame=highest,lowest_frame=lowest,
            capsule_and_joint_radius_bound_m=.035,cart_all8vertices_included=True,overlay_clearance_pixels=8))
        font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',23);small=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',20)
        title=ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf',30)
        writer=imageio.get_writer(str(out),fps=50,codec='libx264',quality=7,macro_block_size=16)
        streak=best=0
        try:
            for i,state in enumerate(tape['states']):
                env.set_state(state);img=Image.fromarray(render(env));d=ImageDraw.Draw(img);tilt=float(env.tilts().max())
                if i:streak=streak+1 if tape['tilts'][i-1].max()<.15 else 0;best=max(best,streak)
                d.rectangle((0,0,1280,145),fill=(10,18,29))
                d.text((25,9),'2D CARTPOLE / 14 LINKS / IMMEDIATE SWING-UP',font=title,fill=(235,244,255))
                phase='SWING-UP' if tilt>=.15 else 'UPRIGHT FEEDBACK'
                d.text((25,53),f'{i*.02:5.2f} s    {phase}    world tilt {tilt:.3f} rad',font=font,fill=(150,220,245))
                d.text((25,84),f'Best hold {best*.02:.2f} s / zero hanging preparation',font=font,fill=(210,225,240))
                d.text((25,114),'FRESH QUALIFICATION / FIRST RESERVED START / 32 OF 32 PASS',font=small,fill=(125,235,160))
                d.text((925,53),'STATE FEEDBACK' if mode=='fixed' else 'NATIVE FORCE RESPONSE',font=small,fill=(210,225,240))
                d.text((925,84),'Refreshed every command',font=small,fill=(150,220,245))
                d.text((925,114),'Offline / sim paused',font=small,fill=(210,225,240))
                d.rectangle((25,637,925,704),fill=(10,18,29))
                d.text((39,643),'Full original 30 s / initial noise .02 / no added gust',font=font,fill=(235,244,255))
                d.text((39,674),'Free XY cart / passive ball joints / original 80 N force limit',font=small,fill=(150,220,245))
                x0,y0,side=1035,445,205;d.rectangle((x0,y0,x0+side,y0+side),fill=(15,25,39),outline=(110,135,155),width=2)
                pts=[(x0+side/2+x/3*side/2,y0+side/2-y/3*side/2) for x,y in tape['states'][:i+1,:2]]
                if len(pts)>1:d.line(pts,fill=(65,170,235),width=2)
                x,y=pts[-1];d.ellipse((x-4,y-4,x+4,y+4),fill=(255,180,75));d.text((x0,y0-31),'XY / +/-3 m',font=small,fill=(210,225,240))
                writer.append_data(np.asarray(img))
                if i%250==0:print(f'render {i}/1501',flush=True)
        finally:writer.close()
        assert env.native_steps==0 and best*.02==s['result']['best_hold_seconds']
        selected=sorted(set((0,100,150,256,257,292,350,750,1500,highest,lowest)));images={};reader=imageio.get_reader(str(out))
        try:
            count=0
            for i,frame in enumerate(reader):
                assert frame.shape==(720,1280,3)
                if i in selected:
                    im=Image.fromarray(frame);im.save(out.with_name(out.stem+f'-frame{i:04d}.png'));images[i]=im.copy()
                count+=1
            assert count==1501 and reader.get_meta_data()['fps']==50
        finally:reader.close()
        sheet=Image.new('RGB',(1280,360*((len(selected)+1)//2)),(255,255,255))
        for j,i in enumerate(selected):sheet.paste(images[i].resize((640,360)),((j%2)*640,(j//2)*360))
        sheet.save(out.parent/'contact-sheet.png')
        write_json(out.with_suffix('.json'),dict(video=str(out.resolve()),sha256=sha(out),frames=1501,fps=50,simulation_seconds=30,
            encoded_seconds=30.02,full_video=True,development_preview=False,qualified=True,links=14,seed=s['result']['seed'],native_steps=0,
            park_seconds=0,native_feedback_from_command=0,offline_guide_preparation=True,generic_swing_up_qualified=True,extra_gust_qualified=False,
            best_hold_seconds=best*.02,source_witness_inventory_sha256=sha(Path(admission['cohort'])/'cases'/str(admission['first_seed'])/'witness_inventory.json'),source_actual_sha256=s['actual_sha256'],
            source_audit_summary_sha256=s['audit_summary_sha256'],predictor_native_steps=s['predictor_native_steps'],render_driver_sha256=sha(out.parent/'render_driver.py'),
            framing_proof_sha256=sha(out.parent/'framing_proof.json'),all1501_encoded_frames_decoded=True,manual_visual_qa_pending=True,keyframes=selected,
            source_cohort_audit_sha256=admission['proof_summary_sha256'],source_cohort_manifest_sha256=admission['cohort_manifest_sha256'],controller_identity=admission['controller_identity'],feedback_mode=mode))
        print(dict(rendered=str(out),decoded_frames=1501,manual_visual_qa_pending=True),flush=True)
    finally:env.close()


if __name__=='__main__':main()
