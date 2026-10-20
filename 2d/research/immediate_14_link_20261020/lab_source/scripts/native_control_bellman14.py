"""Full-state native control Jacobians and a bounded two-force Bellman step."""
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.aligned_value import relative_value_cost
from spatial_synthesis_geometry import CorrectorPolicy,error

EPS=.001
R=.001


def inward_directions(force):
    if np.linalg.norm(force)<=80.-4*EPS:return np.eye(2)
    radial=force/np.linalg.norm(force);perp=np.array([-radial[1],radial[0]])
    return np.column_stack([-radial,(-radial+perp)/np.sqrt(2.)])


def disk_quadratic(U,residual,center):
    qr=np.linalg.qr(np.column_stack([U,residual]),mode='reduced')[1]
    C,q=qr[:2,:2],qr[:2,2]
    proposed=center-np.linalg.solve(C,q);multiplier=0.
    if np.linalg.norm(proposed)>80.:
        def regularized(lam):
            L=np.vstack([C,np.eye(2)*np.sqrt(lam)])
            target=np.r_[C@center-q,np.zeros(2)]
            rr=np.linalg.qr(np.column_stack([L,target]),mode='reduced')[1]
            return np.linalg.solve(rr[:2,:2],rr[:2,2])
        low=0.;high=max(float(np.linalg.norm(C.T@(C@center-q)))/80.,1e-30)
        for _ in range(128):
            proposed=regularized(high)
            if np.linalg.norm(proposed)<=80.:break
            high*=2.
        else:raise RuntimeError('Bounded force quadratic multiplier failed to bracket')
        for _ in range(80):
            mid=(low+high)/2.;trial=regularized(mid)
            if np.linalg.norm(trial)>80.:low=mid
            else:high=mid;proposed=trial
        multiplier=high
    delta=proposed-center;gradient=U.T@(U@delta+residual)+multiplier*proposed
    scale=max(1.,float(np.linalg.norm(U.T@residual)),float(np.linalg.norm(U.T@(U@delta))),multiplier*float(np.linalg.norm(proposed)))
    stationarity=float(np.linalg.norm(gradient))/scale
    feasibility=max(0.,float(np.linalg.norm(proposed))-80.)
    complementarity=abs(multiplier*(float(proposed@proposed)-80.**2))/max(1.,multiplier*80.**2)
    return proposed,dict(multiplier=multiplier,stationarity=stationarity,feasibility=feasibility,complementarity=complementarity,
        passed=bool(max(stationarity,feasibility,complementarity)<1e-6))


class NativeControlBellman:
    def __init__(self,config,v,m,c,w):
        assert config==Config(n_links=14) and not np.any(w)
        self.geometry=SpatialCartpole(config);self.predictor=SpatialCartpole(config)
        self.base=CorrectorPolicy(self.geometry,v,m['F'],c,w)
        self.v,self.m,self.c=v,m,c;self.queries=[];self.calls=[];self.selected=None

    def __call__(self,state,phase):
        raw=self.base(state,phase);baseline=self.geometry.deliver(raw);start=len(self.queries)
        root,s=self.c['S'][phase+1],self.c['value_residual'][phase+1]
        reference_force=self.v['actions'][phase].astype(float)
        def query(force,kind):
            force=np.asarray(force,float);assert np.linalg.norm(force)<=80.+1e-12
            p=self.predictor;p.min_plane_margin=3.;p.warning_seen=False
            successor=p.transition(state,force,quantize=False)
            finite=bool(np.isfinite(successor).all());valid=bool(finite and p.min_plane_margin>=0 and not p.warning_seen)
            e=error(self.geometry,self.v,self.m['F'],successor,phase+1) if finite else np.zeros(74)
            index=len(self.queries);row=dict(phase=phase,kind=kind,force=force.copy(),successor=successor.copy(),
                error=e,margin=p.min_plane_margin,warning=p.warning_seen,valid=valid,relative_cost=0.)
            self.queries.append(row);return index
        base_id=query(baseline,'baseline');base=self.queries[base_id];e0=base['error'];delta0=baseline.astype(float)-reference_force
        value=root@e0+s;base_cost=float(value@value+R*(delta0@delta0));D=inward_directions(baseline.astype(float));probes=[]
        for j in range(2):
            for k in range(1,5):probes.append(query(baseline.astype(float)+k*EPS*D[:,j],'derivative'))
        derivative_valid=all(self.queries[i]['valid'] for i in probes)
        J=np.zeros((74,2));proposed=baseline.astype(float);proof=dict(passed=False,multiplier=0.,stationarity=0.,feasibility=0.,complementarity=0.)
        best=base_id;seen={baseline.tobytes()};trials=[]
        if derivative_valid:
            columns=[]
            for j in range(2):
                ys=[self.queries[i]['error'] for i in probes[4*j:4*j+4]]
                columns.append((-25*e0+48*ys[0]-36*ys[1]+16*ys[2]-3*ys[3])/(12*EPS))
            J=np.linalg.solve(D.T,np.column_stack(columns).T).T
            U=np.vstack([np.eye(2)*np.sqrt(R),root@J]);residual=np.r_[np.sqrt(R)*delta0,value]
            proposed,proof=disk_quadratic(U,residual,baseline.astype(float))
            if proof['passed']:
                for alpha in (1.,.5,.25):
                    force=self.geometry.deliver(baseline.astype(float)+alpha*(proposed-baseline.astype(float)))
                    if force.tobytes() in seen:continue
                    seen.add(force.tobytes());i=query(force,'trial');trials.append(i);row=self.queries[i]
                    row['relative_cost']=relative_value_cost(root,s,e0,row['error'],np.eye(2)*R,delta0,force.astype(float)-reference_force)
                    if row['valid'] and (not self.queries[best]['valid'] or row['relative_cost']<self.queries[best]['relative_cost']):best=i
        assert len(self.queries)-start<=12 and self.predictor.native_steps<=72000
        self.calls.append(dict(phase=phase,baseline=base_id,selected=best,probes=probes,trials=trials,directions=D,control_jacobian=J,
            proposed_force=proposed,proof=proof,baseline_raw=raw,baseline_cost=base_cost,derivative_valid=derivative_valid))
        self.selected=self.queries[best]
        return self.selected['force'].copy()

    def save(self,path):
        q=self.queries;c=self.calls
        np.savez_compressed(path,phase=np.asarray([r['phase'] for r in q],int),kind=np.asarray([r['kind'] for r in q]),
            force=np.asarray([r['force'] for r in q]),successor=np.asarray([r['successor'] for r in q]),error=np.asarray([r['error'] for r in q]),
            margin=np.asarray([r['margin'] for r in q]),warning=np.asarray([r['warning'] for r in q]),valid=np.asarray([r['valid'] for r in q]),relative_cost=np.asarray([r['relative_cost'] for r in q]),
            call_phase=np.asarray([r['phase'] for r in c],int),baseline=np.asarray([r['baseline'] for r in c],int),selected=np.asarray([r['selected'] for r in c],int),
            directions=np.asarray([r['directions'] for r in c]),control_jacobian=np.asarray([r['control_jacobian'] for r in c]),
            proposed_force=np.asarray([r['proposed_force'] for r in c]),baseline_raw=np.asarray([r['baseline_raw'] for r in c]),
            baseline_cost=np.asarray([r['baseline_cost'] for r in c]),derivative_valid=np.asarray([r['derivative_valid'] for r in c]),
            qp_passed=np.asarray([r['proof']['passed'] for r in c]),multiplier=np.asarray([r['proof']['multiplier'] for r in c]),
            stationarity=np.asarray([r['proof']['stationarity'] for r in c]),feasibility=np.asarray([r['proof']['feasibility'] for r in c]),complementarity=np.asarray([r['proof']['complementarity'] for r in c]))

    def close(self):
        assert self.geometry.native_steps==self.base.geometry.native_steps==0
        self.base.close();self.geometry.close();self.predictor.close()
