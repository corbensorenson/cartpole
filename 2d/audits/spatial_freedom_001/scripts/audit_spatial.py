"""Audit physical freedoms independently of planar controller references."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import mujoco as mj
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from spatialpole import Config,SpatialCartpole
from spatialpole.experiment import freeze_inputs,write_json

p = argparse.ArgumentParser()
p.add_argument('--out',required=True)
p.add_argument('--evidence-root',default='output')
p.add_argument('--additional-evidence-root')
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
physics_steps = forward_queries = 0

def directions(env,states):
    global forward_queries
    result = []
    for x in states:
        env.set_state(x)
        result.append(env.data.xmat[env.link_ids].reshape(-1,3,3)[:,:,2].copy())
        forward_queries += 1
    return np.array(result)

def path_ratio(xy):
    singular = np.linalg.svd(xy-np.mean(xy,axis=0),compute_uv=False)
    return float(singular[-1]/max(singular[0],1e-30))

def direct_rollout(env,initial,forces):
    global physics_steps
    data = mj.MjData(env.model)
    data.qpos[:],data.qvel[:] = initial[:env.model.nq],initial[env.model.nq:]
    mj.mj_forward(env.model,data)
    result = [initial.copy()]
    for u in forces:
        data.ctrl[:] = u
        for _ in range(env.config.frame_skip):
            mj.mj_step(env.model,data)
            physics_steps += 1
        mj.mj_forward(env.model,data)
        result.append(np.r_[data.qpos,data.qvel].copy())
    assert not any(w.number for w in data.warning)
    return np.array(result)

inventory,probes,cohorts,paths = [],[],[],{}
for n in (1,3,7):
    env = SpatialCartpole(Config(n_links=n))
    model = env.model
    row = dict(n_links=n,nq=model.nq,nv=model.nv,nu=model.nu,neq=model.neq,
        joint_types=model.jnt_type.tolist(),joint_limits=model.jnt_limited.tolist(),
        slide_axes=model.jnt_axis[:2].tolist(),actuator_joint_ids=model.actuator_trnid[:,0].tolist(),
        actuator_gear=model.actuator_gear.tolist(),dof_damping=model.dof_damping.tolist(),
        dof_armature=model.dof_armature.tolist())
    assert model.nq==2+4*n and model.nv==2+3*n and model.nu==2 and model.neq==0
    assert np.array_equal(model.jnt_type[2:],np.full(n,int(mj.mjtJoint.mjJNT_BALL)))
    assert not model.jnt_limited.any() and np.array_equal(model.actuator_trnid[:,0],[0,1])
    np.testing.assert_array_equal(model.jnt_axis[:2],[[1,0,0],[0,1,0]])
    (out/f'n{n}_model.xml').write_text(env.xml,encoding='utf-8')
    inventory.append(row)
    initial,_ = env.reset(seed=0,options={'mode':'hanging','angle_noise':0.})
    xforce = np.c_[np.r_[np.full(20,10.),np.zeros(10)],np.zeros(30)].astype(np.float32)
    yforce = xforce[:,::-1].copy()
    xs,ys = direct_rollout(env,initial,xforce),direct_rollout(env,initial,yforce)
    dx,dy = directions(env,xs),directions(env,ys)
    rotated = np.stack([-dx[:,:,1],dx[:,:,0],dx[:,:,2]],axis=-1)
    mirror_error = max(float(np.max(np.abs(xs[:,0]-ys[:,1]))),float(np.max(np.abs(dy-rotated))))
    assert mirror_error<1e-9
    assert xs[-1,0]>.1 and ys[-1,1]>.1 and np.max(np.abs(dy[:,:,1]))>.02
    joint_responses = []
    for joint in range(n):
        for axis in (0,1):
            delta = np.zeros(2*model.nv)
            delta[2+3*joint+axis] = .12
            perturbed = env.integrate(initial,delta)
            moved = direct_rollout(env,perturbed,np.zeros((1,2),np.float32))[-1]
            value = float(np.linalg.norm(directions(env,moved[None])[0,joint,:2]))
            assert value>.05
            joint_responses.append(dict(joint=joint,local_axis=axis,horizontal_rod_component=value))
    spin = initial.copy()
    spin[model.nq+4+3*(n-1)] = .5
    spun = direct_rollout(env,spin,np.zeros((1,2),np.float32))[-1]
    spin_increment = float(env.difference(spun,spin)[4+3*(n-1)])
    assert abs(spin_increment)>.001
    t = np.arange(150)*env.config.dt
    forces = np.c_[1.2*np.sin(2*np.pi*t/1.4),2.*np.cos(2*np.pi*t/2.3)].astype(np.float32)
    states = direct_rollout(env,initial,forces)
    rods = directions(env,states)
    replay = [initial]
    for u in forces:
        replay.append(env.transition(replay[-1],u))
    physics_steps += env.native_steps
    np.testing.assert_array_equal(states,np.array(replay))
    cart_rank,rod_rank = path_ratio(states[:,:2]),path_ratio(rods[:,:,:2].reshape(-1,2))
    assert cart_rank>.02 and rod_rank>.02
    probes.append(dict(n_links=n,axis_rotation_error=mirror_error,cart_path_minor_major_ratio=cart_rank,
        rod_horizontal_minor_major_ratio=rod_rank,max_cart_xy=np.max(np.abs(states[:,:2]),axis=0).tolist(),
        max_rod_xy=np.max(np.abs(rods[:,:,:2]),axis=(0,1)).tolist(),
        each_joint_transverse_response=joint_responses,passive_spin_increment=spin_increment,
        persistent_native_vs_environment_replay_exact=True))
    np.savez_compressed(out/f'n{n}_nonplanar_probe.npz',states=states,actions=forces,
        world_directions=rods,config=json.dumps(asdict(env.config)),scope='3s open-loop physical diagnostic; not a swing-up solve')
    paths[n] = {'probe':states[:,:2]}
    env.close()

for n in range(1,8):
    name = f'qualify_n{n}_002' if n==3 else f'qualify_n{n}_001'
    folder = Path(a.evidence_root)/name
    if not (folder/'summary.json').exists() and a.additional_evidence_root:
        folder = Path(a.additional_evidence_root)/name
    if not (folder/'summary.json').exists():
        continue
    summary = json.loads((folder/'summary.json').read_text())
    frozen = np.load(folder/'frozen_route.npz')
    config = Config(**json.loads(str(frozen['config'])))
    assert config==Config(n_links=n)
    env = SpatialCartpole(config)
    ref,actions = frozen['states'],frozen['actions']
    reference_rods = directions(env,ref)
    ycart,yforce,yrod = 0.,0.,0.
    for result in summary['results']:
        tape = np.load(folder/f"seed{result['seed']}.npz")
        ycart = max(ycart,float(np.max(np.abs(tape['states'][:,1]))))
        yforce = max(yforce,float(np.max(np.abs(tape['actions'][:,1]))))
        yrod = max(yrod,float(np.max(np.abs(directions(env,tape['states'])[:,:,1]))))
    cohorts.append(dict(n_links=n,qualification_passes=summary['passes'],episodes=len(summary['results']),
        controller_sha256=hashlib.sha256((folder/'frozen_route.npz').read_bytes()).hexdigest(),
        reference_cart_minor_major_ratio=path_ratio(ref[:,:2]),reference_max_abs_y=float(np.max(np.abs(ref[:,1]))),
        reference_max_abs_force_y=float(np.max(np.abs(actions[:,1]))),
        reference_max_abs_rod_direction_y=float(np.max(np.abs(reference_rods[:,:,1]))),
        actual_max_abs_y=ycart,actual_max_abs_force_y=yforce,actual_max_abs_rod_direction_y=yrod))
    if n in paths:
        paths[n]['reference'] = ref[:,:2]
    env.close()

report = dict(compiled_models=inventory,independent_native_probes=probes,recorded_qualification_motion=cohorts,
    physics_steps=physics_steps,kinematic_forward_queries=forward_queries,
    scope='Plant freedom audit; references may be planar and full 3D noisy success is a separate finite-sample claim')
assert physics_steps<30000
write_json(out/'summary.json',report)
write_json(out/'completed.json',{'status':'completed'})

font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',20)
canvas = Image.new('RGB',(1200,1080),(12,20,32))
draw = ImageDraw.Draw(canvas)
for row,n in enumerate((1,3,7)):
    for col,key in enumerate(('reference','probe')):
        if key not in paths[n]:
            draw.text((col*600+65,row*360+90),'Reference not included in selected evidence',font=font,fill=(235,245,255))
            continue
        xy = paths[n][key]
        x0,y0,w,h = col*600+65,row*360+50,480,265
        center = (np.max(xy,axis=0)+np.min(xy,axis=0))/2
        span = max(float(np.max(np.ptp(xy,axis=0)))*1.15,.1)
        draw.rectangle((x0,y0,x0+w,y0+h),outline=(100,130,155))
        scale = min(w,h)/span
        points = [(x0+w/2+(v[0]-center[0])*scale,y0+h/2-(v[1]-center[1])*scale) for v in xy]
        draw.line(points,fill=(70,200,245),width=3)
        px,py = points[-1]
        draw.ellipse((px-4,py-4,px+4,py+4),fill=(255,180,65))
        draw.text((x0,y0-32),f'{n} link(s): '+('frozen swing reference' if key=='reference' else 'changing-direction XY force probe'),font=font,fill=(235,245,255))
        draw.text((x0,y0+h+5),f'Equal X/Y scale; minor/major ratio {path_ratio(xy):.3f}',font=font,fill=(170,205,225))
canvas.save(out/'xy_motion_audit.png')
print(json.dumps(report,indent=2))
