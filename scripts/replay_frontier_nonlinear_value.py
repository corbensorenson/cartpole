#!/usr/bin/env python
"""Physically test native next-step value corrections along a saved route.

Forecast states stay in a separate predictor. An exported static initializer
requires its own replay and every canonical gate before any release claim.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from gcartpole.config import dump_json,load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import file_metadata,data_sha256,utc_timestamp,runtime_metadata
from gcartpole.ilqr import MujocoTransition
from gcartpole.nonlinear_value_control import select_nonlinear_value_action
from gcartpole.simulation import SimulationError
from scripts.search_capture_sequence import fixed_state_cfg
from scripts.search_ilqr_capture import execute_controller
from scripts.search_fddp_capture import warm_start_diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--source',required=True)
    p.add_argument('--value-factors',required=True);p.add_argument('--capture-cache',required=True)
    p.add_argument('--directory',required=True);p.add_argument('--handoff-seconds',type=float,default=9.)
    p.add_argument('--start-seconds',type=float,default=3.)
    p.add_argument('--decimal-digits',type=int,default=80)
    p.add_argument('--grid-points',type=int,default=17);p.add_argument('--max-iterations',type=int,default=40)
    args=p.parse_args();out=Path(args.directory)
    if (out/'result.json').exists() or (out/'static_initializer.json').exists():raise FileExistsError(out)
    source=json.loads(Path(args.source).read_text());c=source['controller']
    values=json.loads(Path(args.value_factors).read_text());cache=json.loads(Path(args.capture_cache).read_text())
    cfg=load_config(args.config);dt=cfg['env']['timestep']*cfg['env']['frame_skip'];steps=round(args.handoff_seconds/dt);start=round(args.start_seconds/dt)
    if (not c['continuous_angles'] or c['phase_adaptive'] or not 0<=start<steps<len(c['controls'])
            or not np.isclose(start*dt,args.start_seconds) or not np.isclose(steps*dt,args.handoff_seconds)):
        raise ValueError('Fixed-time continuous route and aligned interior correction/handoff times required')
    source_meta=file_metadata(args.source);config_meta=file_metadata(args.config)
    if (values['source_controller']['sha256']!=source_meta['sha256'] or cache['source']['sha256']!=source_meta['sha256']
            or values['config']['sha256']!=config_meta['sha256'] or cache['config']['sha256']!=config_meta['sha256']):
        raise ValueError('Value/capture inputs do not match source and config')
    controls=np.asarray(c['controls']);nodes=np.asarray(source['search']['nominal_coordinate_states']);gains=np.asarray(c['feedback_gains']);transform=np.asarray(c['coordinate_transform']);factors=np.asarray(values['value_factors'])
    if (data_sha256(controls.tolist())!=values['controls_sha256'] or data_sha256(nodes.tolist())!=values['nominal_states_sha256']
            or not np.array_equal(transform,values['coordinate_transform']) or factors.shape!=(len(controls)+1,nodes.shape[1],nodes.shape[1])
            or not np.all(np.isfinite(factors))):raise ValueError('Value roots do not match route nodes/controls/basis')
    gain=np.asarray(cache['gain']);factor=np.asarray(source['lyapunov']['factor'])
    if data_sha256(gain.tolist())!=cache['gain_sha256'] or data_sha256(factor.tolist())!=source['lyapunov']['factor_sha256']:
        raise ValueError('Capture gain or value factor hash mismatch')
    cfg=fixed_state_cfg(cfg,source['selected_state'],30.)
    env=NLinkCartPoleEnv(cfg,progress=1.,seed=source['seed']);env.reset(seed=source['seed'])
    predictor=MujocoTransition(env,transform,continuous_angles=True);evaluations=0;fallbacks=0
    begun=time.monotonic()
    def selector(step,state,baseline):
        nonlocal evaluations,fallbacks
        if step<start:return baseline,None
        try:
            action,diag=select_nonlinear_value_action(predictor,state,factors[step+1],baseline,
                error_function=lambda predicted:predicted-nodes[step+1],rail_limit=env.rail_limit,
                grid_points=args.grid_points,max_iterations=args.max_iterations,
                control_cost=values['control_cost']+values['regularization'],reference_action=float(controls[step]),
                decimal_digits=args.decimal_digits,candidate_actions=[controls[step]])
            evaluations+=diag['predictor_evaluations'];diag['baseline_action']=baseline
        except SimulationError as error:
            # Keep an executed failure record; the live environment enforces
            # rail/simulation termination. Never substitute a forecast state.
            action=float(np.float32(np.clip(baseline,-1,1)));fallbacks+=1
            diag=dict(fallback=True,error=str(error))
        if step%25==0:print(json.dumps(dict(step=step,time_seconds=step*dt,action=action,diagnostic=diag)),flush=True)
        return action,diag
    try:
        result=execute_controller(cfg,progress=1.,seed=source['seed'],controls=controls[:steps],nominal_states=nodes[:steps+1],feedback_gains=gains[:steps],
            gain=gain,lqr_scale=c['lqr_scale'],transform=transform,lyapunov=factor.T@factor,lyapunov_factor=factor,
            handoff_lyapunov=c['handoff_lyapunov'],handoff_cart_abs=c['handoff_cart_abs'],handoff_angle_abs=c['handoff_angle_abs'],
            handoff_cart_velocity_abs=c['handoff_cart_velocity_abs'],handoff_hinge_velocity_rms=c['handoff_hinge_velocity_rms'],
            tracking_mode='native_next_step_value_tracking',defer_handoff_until_horizon=True,continuous_angles=True,
            tracking_action_selector=selector)
        length=min(steps,result['length']);rows=result['trajectory'][:length]
        initial=np.r_[source['selected_state']['qpos'],source['selected_state']['qvel']]
        actual=np.vstack([initial,[r['qpos']+r['qvel'] for r in rows]])@transform.T
        delivered=np.asarray([r['action'] for r in rows],dtype=np.float32).astype(float)
        diagnostics=warm_start_diagnostics(predictor,actual[0],delivered,actual)
        if not diagnostics['is_exactly_feasible']:raise ValueError('Recorded physical prefix does not independently reproduce exactly')
        full_pass=bool(result['success'] and result['trajectory_integrity'] and result['termination_reason']=='time_limit' and result['length']==1500 and result['max_upright_streak_seconds']>=5 and result['max_cart_excursion']<=3)
        dump_json(dict(not_solution=True,benchmark_evidence=False,generated_at=utc_timestamp(),runtime=runtime_metadata(),parameters=vars(args),
            source=source_meta,config=config_meta,value_factors=file_metadata(args.value_factors),capture_cache=file_metadata(args.capture_cache),
            result=result,full_development_episode_passed=full_pass,physical_prefix_diagnostics=diagnostics,
            predictor_evaluations=evaluations,total_predictor_transition_calls=predictor.evaluations,fallbacks=fallbacks,wall_time_seconds=time.monotonic()-begun,
            executed_policy=dict(type='native_next_step_factored_value_tracking_then_lqr',frequency_hz=1/dt,action_precision='float32',
                cost_arithmetic_decimal_digits=args.decimal_digits,live_state_projection=False,static_initializer_is_not_executed_policy=True),
            note='One exact-start development episode only. The emitted physical tape with inherited feedback is an offline initializer, not the policy that produced this episode. It requires independent static replay, exact rebuild and all noisy/reserved release gates.'),out/'result.json')
        initializer=dict(c);initializer.update(type='physical_nonlinear_preview_tape_static_feedback_initializer',controls=delivered.tolist(),feedback_gains=gains[:length].tolist(),horizon_steps=length,horizon_seconds=length*dt,final_trajectory_diagnostics=diagnostics,initialization_only=True,feedback_gains_are_distillation_initializer=True)
        initializer.pop('solver_feedback_gains',None)
        dump_json(dict(not_solution=True,initialization_only=True,seed=source['seed'],selected_state=source['selected_state'],controller=initializer,
            search=dict(nominal_coordinate_states=actual.tolist()),source=file_metadata(out/'result.json'),
            note='Exact physical prefix tape; inherited feedback must be independently tested about these new nodes.'),out/'static_initializer.json')
        print(json.dumps(dict(full_episode_passed=full_pass,length=result['length'],hold=result['max_upright_streak_seconds'],termination=result['termination_reason'],fallbacks=fallbacks)),flush=True)
    finally:env.close()
if __name__=='__main__':main()
