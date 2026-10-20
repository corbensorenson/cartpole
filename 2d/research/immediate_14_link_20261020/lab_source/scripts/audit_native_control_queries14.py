"""Independent native-control query/Jacobian/KKT checks shared by new case audits."""
import hashlib
import math
import numpy as np
from spatialpole.native_mpc import exact_bytes
from spatialpole.experiment import write_json
from spatial_synthesis_geometry import CorrectorPolicy,error


def verify(trace,v,r,c,plan,tape,geometry,native,cold,out):
    screen=False;observations={t:x for t,x in enumerate(tape['states'][:-1])}
    assert trace['call_phase'].tolist()==list(range(len(tape['actions'])))
    native_before=native.native_steps;base=CorrectorPolicy(geometry,v,r['F'],c,plan['w'])
    try:
        offset=0;query_digest=hashlib.sha256();changes=0;selected_costs=[]
        for j,phase in enumerate(trace['call_phase']):
            phase=int(phase);state=observations[phase];ids=np.flatnonzero(trace['phase']==phase);assert ids.tolist()==list(range(offset,offset+len(ids))) and 9<=len(ids)<=12
            offset+=len(ids);baseline_id=int(trace['baseline'][j]);selected=int(trace['selected'][j]);assert baseline_id==ids[0] and selected in ids
            assert trace['kind'][baseline_id]=='baseline' and np.all(trace['kind'][ids[1:9]]=='derivative') and np.all(trace['kind'][ids[9:]]=='trial')
            raw=base(state,phase);exact_bytes(raw,trace['baseline_raw'][j]);force0=geometry.deliver(raw).astype(float);exact_bytes(force0,trace['force'][baseline_id])
            for i in ids:
                force=trace['force'][i];assert np.linalg.norm(force)<=80.
                native.min_plane_margin=3.;native.warning_seen=False;successor=native.transition(state,force,quantize=False)
                exact_bytes(successor,trace['successor'][i]);assert native.min_plane_margin==trace['margin'][i] and native.warning_seen==trace['warning'][i]
                valid=bool(np.isfinite(successor).all() and native.min_plane_margin>=0 and not native.warning_seen);assert valid==bool(trace['valid'][i])
                e=error(geometry,v,r['F'],successor,phase+1) if np.isfinite(successor).all() else np.zeros(74);exact_bytes(e,trace['error'][i])
                query_digest.update(state.tobytes());query_digest.update(force.tobytes());query_digest.update(successor.tobytes())
            if np.linalg.norm(force0)<=80.-.004:D=np.eye(2)
            else:
                radial=force0/np.linalg.norm(force0);perp=np.array([-radial[1],radial[0]])
                D=np.column_stack([-radial,(-radial+perp)/np.sqrt(2.)])
            exact_bytes(D,trace['directions'][j])
            for axis in range(2):
                for k in range(1,5):exact_bytes(force0+k*.001*D[:,axis],trace['force'][ids[1+axis*4+k-1]])
            derivative_valid=bool(np.all(trace['valid'][ids[1:9]]));assert derivative_valid==bool(trace['derivative_valid'][j])
            root,residual=c['S'][phase+1],c['value_residual'][phase+1];e0=trace['error'][baseline_id]
            delta0=force0-v['actions'][phase].astype(float);value=root@e0+residual
            baseline_value=float(value@value+.001*(delta0@delta0));assert baseline_value==trace['baseline_cost'][j]
            J=np.zeros((74,2));expected_trials=[]
            if derivative_valid:
                columns=[]
                for axis in range(2):
                    y=[trace['error'][i] for i in ids[1+axis*4:5+axis*4]]
                    columns.append((-25*e0+48*y[0]-36*y[1]+16*y[2]-3*y[3])/.012)
                J=np.linalg.solve(D.T,np.column_stack(columns).T).T
                U=np.vstack([np.eye(2)*np.sqrt(.001),root@J]);rr=np.r_[np.sqrt(.001)*delta0,value]
                u=trace['proposed_force'][j];lam=float(trace['multiplier'][j]);assert lam>=0
                delta=u-force0;gradient=U.T@(U@delta+rr)+lam*u
                scale=max(1.,float(np.linalg.norm(U.T@rr)),float(np.linalg.norm(U.T@(U@delta))),lam*float(np.linalg.norm(u)))
                stationarity=float(np.linalg.norm(gradient))/scale;feasibility=max(0.,float(np.linalg.norm(u))-80.)
                complementarity=abs(lam*(float(u@u)-80.**2))/max(1.,lam*80.**2)
                assert stationarity==trace['stationarity'][j] and feasibility==trace['feasibility'][j] and complementarity==trace['complementarity'][j]
                qp_passed=max(stationarity,feasibility,complementarity)<1e-6;assert qp_passed==bool(trace['qp_passed'][j])
                if qp_passed:
                    seen={force0.astype(np.float32).tobytes()}
                    for alpha in (1.,.5,.25):
                        force=geometry.deliver(force0+alpha*(u-force0));key=force.tobytes()
                        if key in seen:continue
                        seen.add(key);expected_trials.append(force.astype(float))
            else:assert not trace['qp_passed'][j]
            exact_bytes(J,trace['control_jacobian'][j]);assert len(expected_trials)==len(ids[9:])
            best=baseline_id
            for force,i in zip(expected_trials,ids[9:]):
                exact_bytes(force,trace['force'][i]);delta=force-v['actions'][phase].astype(float);dv=root@(trace['error'][i]-e0)
                cost=math.fsum(((2*value+dv)*dv).tolist())+float((delta-delta0)@(np.eye(2)*.001)@(delta+delta0))
                assert cost==trace['relative_cost'][i]
                if trace['valid'][i] and (not trace['valid'][best] or cost<trace['relative_cost'][best]):best=int(i)
            assert selected==best;changes+=int(selected!=baseline_id);selected_costs.append(float(trace['relative_cost'][selected]))
            if not screen:
                exact_bytes(trace['force'][selected],tape['raw_actions'][phase]);exact_bytes(geometry.deliver(trace['force'][selected]),tape['actions'][phase])
                state,_,done,truncated,info=cold.step_force(tape['actions'][phase]);exact_bytes(state,tape['states'][phase+1]);exact_bytes(cold.tilts(),tape['tilts'][phase])
                exact_bytes(state,trace['successor'][selected]);assert phase==len(tape['actions'])-1 or not (done or truncated)
            if j%100==0:write_json(out/'progress.json',dict(checked_calls=j+1,native_steps=native.native_steps+cold.native_steps))
        assert offset==len(trace['phase']) and native.native_steps-native_before==4*len(trace['phase']) and geometry.native_steps==0
        return dict(passed=True,new_predictor_native_steps=native.native_steps-native_before,query_intervals=len(trace['phase']),
            calls=len(trace['call_phase']),changed_actions=changes,query_input_output_sha256=query_digest.hexdigest(),
            all_native_queries_byte_exact=True,all_derivative_kkt_relative_cost_order_and_selected_actions_checked=True,
            all_actual_controls_and_selected_next_states_byte_exact=True)
    finally:base.close()
