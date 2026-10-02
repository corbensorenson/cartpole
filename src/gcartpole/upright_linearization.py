"""Structured upright linearization of the compiled planar serial-chain plant.

Compiled model scalars are promoted exactly; mass/gravity sums and the RK4
map are formed at the requested precision. This improves identification of
the equilibrium derivative, not the precision of the nonlinear MuJoCo plant.
Only the unconstrained planar serial geometry is supported here.
"""
from __future__ import annotations

import mpmath
import mujoco
import numpy as np


def planar_upright_rk4(model, normalized_force, frame_skip, *, decimal_digits=80):
    if decimal_digits < 30 or int(frame_skip) != frame_skip or frame_skip < 1:
        raise ValueError('>=30 digits and a positive integer frame skip required')
    n = model.nv - 1
    if (n < 1 or model.nq != model.nv or model.njnt != n+1
            or model.nbody != n+2 or model.nu != 1 or model.neq != 0
            or model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4):
        raise ValueError('unconstrained single-input planar serial RK4 chain required')
    if (not np.isfinite(normalized_force) or normalized_force <= 0
            or not np.array_equal(model.jnt_type, [mujoco.mjtJoint.mjJNT_SLIDE]+[mujoco.mjtJoint.mjJNT_HINGE]*n)
            or not np.array_equal(model.jnt_axis, [[1.,0.,0.]]+[[0.,1.,0.]]*n)
            or not np.array_equal(model.jnt_pos, np.zeros((n+1,3)))
            or not np.array_equal(model.qpos0, np.zeros(n+1))
            or np.any(model.dof_frictionloss) or np.any(model.jnt_stiffness)
            or np.any(model.geom_contype) or np.any(model.geom_conaffinity)
            or model.opt.density != 0 or model.opt.viscosity != 0
            or np.any(model.opt.wind)
            or int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)):
        raise ValueError('unsupported axes, reference, passive forces, contacts or model options')
    if (not np.array_equal(model.jnt_bodyid, np.arange(1,n+2))
            or not np.array_equal(model.body_parentid[1:], np.arange(n+1))
            or not np.array_equal(model.body_quat[1:], np.tile([1.,0.,0.,0.],(n+1,1)))
            or np.any(model.body_pos[1:,:2]) or np.any(model.body_ipos[1:,:2])
            or np.any(model.jnt_limited[1:])
            or not model.jnt_range[0,0] < 0 < model.jnt_range[0,1]
            or not np.array_equal(model.actuator_trnid[0], [0,-1])
            or model.actuator_trntype[0] != mujoco.mjtTrn.mjTRN_JOINT
            or model.actuator_dyntype[0] != mujoco.mjtDyn.mjDYN_NONE
            or model.actuator_gaintype[0] != mujoco.mjtGain.mjGAIN_FIXED
            or model.actuator_biastype[0] != mujoco.mjtBias.mjBIAS_NONE
            or model.actuator_gainprm[0,0] != 1
            or not np.array_equal(model.actuator_gear[0], [1.,0.,0.,0.,0.,0.])
            or model.opt.gravity[0] != 0 or model.opt.gravity[1] != 0 or model.opt.gravity[2] >= 0):
        raise ValueError('unsupported body tree, joint limits, actuation or gravity')
    ctx=mpmath.mp.clone();ctx.dps=decimal_digits;mp=lambda x:ctx.mpf(float(x))
    lengths=[];centers=[];masses=[];inertias=[]
    for i in range(n):
        body=i+2;sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,f'tip_{i+1}')
        if (sid < 0 or model.site_bodyid[sid] != body or np.any(model.site_pos[sid,:2])
                or model.site_pos[sid,2] <= 0 or model.body_mass[body] <= 0
                or not 0 < model.body_ipos[body,2] <= model.site_pos[sid,2]
                or (i>0 and model.body_pos[body,2] != model.site_pos[
                    mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,f'tip_{i}'),2])):
            raise ValueError('serial link lengths, centers and tip sites must agree')
        lengths.append(mp(model.site_pos[sid,2]));centers.append(mp(model.body_ipos[body,2]));masses.append(mp(model.body_mass[body]))
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,model.body_iquat[body]);row=rotation.reshape(3,3)[1]
        inertias.append(sum(mp(row[k])**2*mp(model.body_inertia[body,k]) for k in range(3)))
    d=n+1;mass=ctx.zeros(d);gravity=ctx.zeros(d);jacobians=[]
    for i in range(n):
        jacobians.append([ctx.mpf(1)]+[sum(lengths[j:i])+centers[i] if j<=i else ctx.mpf(0) for j in range(n)])
    mass[0,0]=mp(model.body_mass[1])
    for i in range(n):
        for j in range(d):
            for k in range(d):
                mass[j,k]+=masses[i]*jacobians[i][j]*jacobians[i][k]
                if 0<j<=i+1 and 0<k<=i+1:mass[j,k]+=inertias[i]
        moment=masses[i]*centers[i]+lengths[i]*sum(masses[i+1:])
        for j in range(i+1):
            for k in range(i+1):gravity[j+1,k+1]+=-mp(model.opt.gravity[2])*moment
    for j in range(d):mass[j,j]+=mp(model.dof_armature[j])
    damping=ctx.diag([mp(x) for x in model.dof_damping]);ctx.cholesky(mass)
    continuous=ctx.zeros(2*d);continuous_input=ctx.zeros(2*d,1)
    inverse=ctx.inverse(mass);acc_q=inverse*gravity;acc_v=-inverse*damping
    for j in range(d):
        continuous[j,d+j]=1
        continuous_input[d+j,0]=inverse[j,0]*mp(normalized_force)
        for k in range(d):continuous[d+j,k]=acc_q[j,k];continuous[d+j,d+k]=acc_v[j,k]
    # RK4's linear map with a zero-order-held cart force; composition retains
    # the exact physics substep grid and the declared action hold duration.
    h=mp(model.opt.timestep);identity=ctx.eye(2*d);power=identity.copy();single=identity.copy();single_b=ctx.zeros(2*d,1)
    for order in range(1,5):
        single_b+=h**order/ctx.factorial(order)*(power*continuous_input)
        power=power*continuous;single+=h**order/ctx.factorial(order)*power
    discrete=identity.copy();discrete_b=ctx.zeros(2*d,1)
    for _ in range(int(frame_skip)):discrete_b=single*discrete_b+single_b;discrete=single*discrete
    return discrete,discrete_b,dict(method='compiled_planar_chain_structured_upright_rk4',decimal_digits=decimal_digits,n_links=n,physics_timestep=float(model.opt.timestep),frame_skip=int(frame_skip),nonlinear_certified=False,input_precision='compiled binary64 model scalars promoted exactly; structured mass/gravity/RK4 operations retain requested precision',mass_matrix=mass,gravity_matrix=gravity,damping_matrix=damping)
