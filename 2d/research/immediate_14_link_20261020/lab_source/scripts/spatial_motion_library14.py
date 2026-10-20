"""Common-yaw fitting and physical means over up to six existing motions."""
import itertools
import numpy as np
from spatialpole.circular import rotate_world_azimuth
from spatialpole import Config
from spatialpole.circular import state_from_physical
from spatialpole.world_coordinates import native_geometry,world_coordinate_error
from prepare_spatial_motion_blend14 import tilt_vectors,SCALE


def simplex_fit(C,target):
    """Solve all faces of the finite-variable convex QP; independently check KKT."""
    n=C.shape[1];assert 2<=n<=6
    G,b=C.T@C,C.T@target;answers=[]
    for k in range(1,n+1):
        for active in itertools.combinations(range(n),k):
            active=list(active);K=np.block([[G[np.ix_(active,active)],np.ones((k,1))],[np.ones((1,k)),np.zeros((1,1))]])
            try:x=np.linalg.solve(K,np.r_[b[active],1.])
            except np.linalg.LinAlgError:continue
            w=np.zeros(n);w[active]=x[:k]
            if np.min(w)<-1e-12:continue
            g=G@w-b;inactive=[j for j in range(n) if j not in active]
            if inactive and np.min(g[inactive]-np.mean(g[active])) < -1e-10:continue
            answers.append((float(np.sum((C@w-target)**2)),w))
    assert answers;cost,w=min(answers,key=lambda row:row[0])
    gradient=C.T@(C@w-target);active=w>1e-10;lagrange=float(np.mean(gradient[active]))
    assert abs(np.sum(w)-1.)<1e-12 and np.min(w)>=-1e-12
    assert np.max(abs(gradient[active]-lagrange))<1e-9
    assert np.all(gradient[~active]-lagrange>=-1e-9)
    return w,cost,dict(sum_error=abs(float(w.sum())-1.),active_stationarity=float(np.max(abs(gradient[active]-lagrange))),
        minimum_inactive_dual=float(np.min(gradient[~active]-lagrange)) if np.any(~active) else None)


def blend_reference(geometry,initial,tapes):
    assert geometry.config==Config(n_links=14) and 2<=len(tapes)<=6
    assert initial.shape==(102,) and np.isfinite(initial).all()
    for tape in tapes:assert tape['states'].shape==(1501,102) and tape['actions'].shape==(1500,2)
    nominal,_=geometry.reset(seed=20261016,options=dict(mode='hanging',angle_noise=0.))
    R0,w0=native_geometry(geometry,nominal)
    C=np.asarray([world_coordinate_error(geometry,t['states'][0],nominal,R0,w0)/SCALE for t in tapes]).T
    target=world_coordinate_error(geometry,initial,nominal,R0,w0)/SCALE
    weights,cost,kkt=simplex_fit(C,target);single=np.sum((C-target[:,None])**2,axis=0)
    states=[];directions=[];omegas=[];tilts=[];direction_errors=[];rate_errors=[]
    for phase in range(1501):
        pairs=[native_geometry(geometry,t['states'][phase]) for t in tapes]
        direction=np.einsum('k,kij->ij',weights,np.asarray([r[:,:,2] for r,w in pairs]))
        direction/=np.linalg.norm(direction,axis=1)[:,None]
        omega=np.einsum('k,kij->ij',weights,np.asarray([w for r,w in pairs]))
        xy=sum(weight*t['states'][phase,:2] for weight,t in zip(weights,tapes))
        velocity=sum(weight*t['states'][phase,58:60] for weight,t in zip(weights,tapes))
        vectors=tilt_vectors(direction);x=state_from_physical(geometry,xy,vectors,velocity,omega)
        R,w=native_geometry(geometry,x)
        direction_errors.append(float(np.max(abs(R[:,:,2]-direction))));rate_errors.append(float(np.max(abs(w-omega))))
        states.append(x);directions.append(direction);omegas.append(omega);tilts.append(np.linalg.norm(vectors,axis=1))
    states=np.asarray(states);assert max(direction_errors)<1e-12 and max(rate_errors)<1e-10
    rank_checks=[]
    for phase in (0,160,325,1000,1500):
        columns=[];base=states[phase];vectors=tilt_vectors(directions[phase]);eps=1e-5
        for j in range(44):
            values=[]
            for sign in (-1,1):
                velocity=base[58:60].copy();omega=omegas[phase].copy()
                if j<2:velocity[j]+=sign*eps
                else:omega.flat[j-2]+=sign*eps
                values.append(state_from_physical(geometry,base[:2],vectors,velocity,omega)[58:])
            columns.append((values[1]-values[0])/(2*eps))
        rank=int(np.linalg.matrix_rank(np.asarray(columns).T));assert rank==44;rank_checks.append(dict(phase=phase,rank=rank))
    actions=np.asarray([geometry.deliver(sum(weight*t['actions'][phase].astype(float) for weight,t in zip(weights,tapes))) for phase in range(1500)])
    states[0]=initial.copy();margin=float(3.-np.max(abs(states[:,:2])));tail=float(np.max(tilts[-251:]))
    valid=bool(margin>=0. and tail<.15 and np.max(np.linalg.norm(actions.astype(float),axis=1))<=80.)
    assert geometry.native_steps==0
    return dict(states=states,actions=actions),dict(geometry_passed=valid,weights=weights.tolist(),initial_scaled_squared_error=cost,
        single_source_scaled_squared_errors=single.tolist(),simplex_kkt=kkt,virtual_plane_margin=margin,last5_virtual_max_tilt=tail,
        all_physical_rates=44,rate_chart_checks=rank_checks,maximum_direction_roundtrip_error=max(direction_errors),
        maximum_world_rate_roundtrip_error=max(rate_errors),nontrivial_fit=bool(np.count_nonzero(weights>1e-10)>=2),
        initial_fit_improves20_percent=bool(cost<=.8*min(single)),native_steps=0,
        rule='unique convex minimum over fixed successful motions, including vertex solutions; initial fitting improvement is diagnostic, original physical gates are unchanged')


def initial_columns(geometry,initial_states,nominal,R,w,angle):
    return np.asarray([world_coordinate_error(geometry,rotate_world_azimuth(geometry,x,angle),nominal,R,w)/SCALE
        for x in initial_states]).T


def fit(C0,Cpi,Cquarter,target):
    """The rotation is common to all motions, keeping the blend coherent."""
    n=C0.shape[1];assert 2<=n<=6
    assert C0.shape==Cpi.shape==Cquarter.shape==(74,n) and target.shape==(74,)
    G=C0.T@C0;b0=C0.T@target;bp=Cpi.T@target;bq=Cquarter.T@target
    bz=(b0+bp)/2;bc=(b0-bp)/2;bs=bq-bz
    candidates=[];rows=[]
    for count in range(1,n+1):
        for ids in itertools.combinations(range(n),count):
            ids=list(ids);K=np.block([[G[np.ix_(ids,ids)],np.ones((count,1))],
                [np.ones((1,count)),np.zeros((1,1))]])
            Kinv=np.linalg.inv(K);rz=np.r_[bz[ids],1.];rc=np.r_[bc[ids],0.];rs=np.r_[bs[ids],0.]
            c1=-2*rz@Kinv@rc;s1=-2*rz@Kinv@rs
            c2=-.5*(rc@Kinv@rc-rs@Kinv@rs);s2=-rc@Kinv@rs
            polynomial=np.asarray([-s1+2*s2,-2*c1+8*c2,-12*s2,-2*c1-8*c2,s1+2*s2])
            roots=np.roots(np.trim_zeros(polynomial,'f')) if np.any(polynomial) else np.array([])
            angles=[0.,np.pi]+[float(2*np.arctan(z.real)) for z in roots if abs(z.imag)<1e-8]
            for angle in angles:
                b=bz+bc*np.cos(angle)+bs*np.sin(angle);solution=Kinv@np.r_[b[ids],1.]
                weights=np.zeros(n);weights[ids]=solution[:count]
                if weights.min() < -1e-10:continue
                gradient=G@weights-b;lagrange=float(np.mean(gradient[ids]));inactive=[k for k in range(n) if k not in ids]
                if inactive and np.min(gradient[inactive]-lagrange)<-1e-8:continue
                value=float(target@target+weights@G@weights-2*b@weights)
                derivative=float(-c1*np.sin(angle)+s1*np.cos(angle)-2*c2*np.sin(2*angle)+2*s2*np.cos(2*angle))
                row=dict(angle=angle,weights=weights.tolist(),cost=value,active=ids,angle_derivative=derivative)
                rows.append(row);candidates.append((value,angle,weights))
    assert candidates
    value,angle,weights=min(candidates,key=lambda x:(x[0],abs(x[1])))
    return angle,weights,value,dict(faces=2**n-1,candidates=rows,all_stationary_real_roots_enumerated=True,
        b_constant=bz,b_cosine=bc,b_sine=bs,G=G)


def observed_fit(geometry,initial,tapes):
    n=len(tapes);assert 2<=n<=6
    nominal,_=geometry.reset(seed=20261016,options=dict(mode='hanging',angle_noise=0.))
    R,w=native_geometry(geometry,nominal);initial_states=[t['states'][0] for t in tapes]
    target=world_coordinate_error(geometry,initial,nominal,R,w)/SCALE
    columns=[initial_columns(geometry,initial_states,nominal,R,w,a) for a in (0.,np.pi,np.pi/2)]
    angle,weights,cost,details=fit(*columns,target)
    C=initial_columns(geometry,initial_states,nominal,R,w,angle)
    exact_weights,exact_cost,kkt=simplex_fit(C,target)
    assert np.max(abs(exact_weights-weights))<1e-7 and abs(exact_cost-cost)<1e-7
    G=columns[0].T@columns[0]
    checks=[]
    for a in (.37,1.13,-2.11):
        Ctest=initial_columns(geometry,initial_states,nominal,R,w,a)
        predicted_b=details['b_constant']+details['b_cosine']*np.cos(a)+details['b_sine']*np.sin(a)
        gram=float(np.max(abs(Ctest.T@Ctest-G)));rhs=float(np.max(abs(Ctest.T@target-predicted_b)))
        assert gram<1e-8 and rhs<1e-8;checks.append(dict(angle=a,gram_error=gram,rhs_error=rhs))
    _,old_cost,_=simplex_fit(columns[0],target)
    assert geometry.native_steps==0
    return angle,exact_weights,dict(cost=exact_cost,original_cost=old_cost,cost_fraction=exact_cost/old_cost,
        angle=angle,weights=exact_weights.tolist(),kkt=kkt,rotation_identity_checks=checks,
        stationary_candidates=details['candidates'],faces=2**n-1),dict(C0=columns[0],Cpi=columns[1],Cquarter=columns[2],target=target)


def reference(geometry,initial,tapes):
    assert geometry.config==Config(n_links=14) and geometry.native_steps==0
    angle,weights,fit,_=observed_fit(geometry,initial,tapes)
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    transformed=[]
    for tape in tapes:
        states=np.asarray([rotate_world_azimuth(geometry,x,angle) for x in tape['states']])
        actions=np.asarray([geometry.deliver(rotation@u.astype(float)) for u in tape['actions']])
        transformed.append(dict(states=states,actions=actions))
    v,info=blend_reference(geometry,initial,transformed)
    assert np.max(abs(np.asarray(info['weights'])-weights))<1e-7 and geometry.native_steps==0
    return v,dict(geometry_passed=info['geometry_passed'],azimuth_fit=fit,physical_geometry=info,
        native_steps=0,all_physical_rates=44,
        rule='global minimum initial-state fit over common azimuth and finite-source simplex; fit improvement diagnostic, original physical task gates unchanged')
