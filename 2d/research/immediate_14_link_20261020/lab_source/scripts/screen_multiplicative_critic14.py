"""One saved-data uncertainty-aware critic screen; never advances the plant."""
import argparse
from pathlib import Path
import shutil
import numpy as np
from scipy.linalg import solve_triangular
from spatialpole import Config, SpatialCartpole
from spatialpole.experiment import freeze_inputs, write_json
from spatialpole.native_mpc import exact_bytes
from audit_spatial_synthesis_models import load
from eleven_synthesis_protocol import read, sha
from eleven_value_sequence import right
from multiplicative_critic14 import design, fit, dense_toy_check, DesignRangeError
from stage_six_source_synthesis14 import helper_closure


def envelope(m, v, c, C, g):
    H = len(m['A'])
    z, W = np.zeros(74), np.zeros((74,74))
    mean = [z.copy()]
    forces, raws, cart_upper, force_upper, tilt_upper, weights = [], [], [], [], [], []
    for t in range(H):
        S,Sn,G,f = c['S'][t],c['S'][t+1],c['gain'][t],c['feed'][t]
        K = right(G,S)
        raw = v['actions'][t].astype(float)+f-K@z
        force = g.deliver(raw)
        du = force.astype(float)-v['actions'][t]
        LW = right(m['factors'][t],S)
        energy = float(np.sum((LW@z)**2)+np.sum((LW@W)**2)+.001*(du@du+np.sum((K@W)**2)))
        SZ = Sn@np.linalg.solve(m['factors'][t+1],C)
        A = right(Sn@m['A'][t]-(Sn@m['B'][t])@G,S)
        zn = A@z+Sn@(m['B'][t]@f+m['d'][t])+(Sn@m['B'][t])@(force.astype(float)-raw)
        Wn = np.linalg.qr(np.c_[A@W,SZ*np.sqrt(energy)].T,mode='reduced')[1].T
        if not np.isfinite(zn).all() or not np.isfinite(Wn).all() or max(np.max(abs(zn)),np.max(abs(Wn)))>1e100:
            return dict(passed=False,stage='envelope_range_rejected',commands=t),dict(white_mean=np.array(mean))
        ee = solve_triangular(Sn,zn)
        x = g.integrate(v['states'][t+1],m['E'][t+1]@ee)
        g.set_state(x)
        eW = solve_triangular(Sn,Wn)
        cart = v['states'][t+1,:2]+m['cart'][t+1]@ee
        cart_unc = np.linalg.norm(m['cart'][t+1]@eW,axis=1)
        physical_unc = m['factors'][t+1]@eW
        # The 28 tilt rows in the weighted physical chart have weight40.
        polar_unc = physical_unc[2:30].reshape(14,2,74)/np.sqrt(40.)
        angle_unc = np.sqrt(np.sum(polar_unc*polar_unc,axis=(1,2)))
        tilt_upper.append(g.tilts()+3.*angle_unc)
        cart_upper.append(abs(cart)+3.*cart_unc)
        force_upper.append(float(np.linalg.norm(force.astype(float))+3.*np.linalg.norm(K@W)))
        weights.append(energy)
        forces.append(force)
        raws.append(raw)
        z,W = zn,Wn
        mean.append(z.copy())
    cart_upper,force_upper,tilt_upper = map(np.array,(cart_upper,force_upper,tilt_upper))
    passed = bool(np.max(cart_upper)<=3. and np.max(force_upper)<=80. and np.max(tilt_upper[-251:])<.15)
    return dict(passed=passed,stage='envelope_admitted' if passed else 'envelope_rejected',commands=H,
        model_cart_upper_max=float(np.max(cart_upper)),model_force_upper_max=float(np.max(force_upper)),
        model_last251_tilt_upper_max=float(np.max(tilt_upper[-251:]))),dict(white_mean=np.array(mean),
        mean_actions=np.array(forces),mean_raw_actions=np.array(raws),cart_upper=cart_upper,force_upper=force_upper,
        tilt_upper=tilt_upper,energy=np.array(weights),final_white_covariance_factor=W)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--diagnostic-inventory')
    p.add_argument('--ledger-inventory')
    p.add_argument('--run')
    p.add_argument('--out',required=True)
    a = p.parse_args()
    actor = not a.run
    args = vars(a) if actor else read(Path(a.run)/'inputs.json')['args']
    inventory,ledger,out = Path(args['diagnostic_inventory']),Path(args['ledger_inventory']),Path(a.out)
    parent,lp = read(inventory),read(ledger)
    diagnostic = Path(parent['run'])
    assert parent['passed'] and lp['passed'] and not parent['task_success']
    assert sha(diagnostic/'summary.json')==parent['summary_sha256'] and sha(diagnostic/'diagnostic_arrays.npz')==parent['arrays_sha256']
    actor_inventory = Path(read(diagnostic/'inputs.json')['args']['inventory'])
    assert sha(actor_inventory)==parent['parent_inventory_sha256']
    ap = read(actor_inventory)
    run = Path(ap['run'])
    assert ap['passed'] and not ap['component_plan_found'] and sha(run/'summary.json')==ap['summary_sha256']
    assert sha(Path(ap['audit'])/'summary.json')==ap['audit_summary_sha256']
    assert sha(run/'adaptive_forecast.npz')==ap['forecast_sha256']
    manifest = read(run/'frozen_manifest.json')
    assert manifest['config']==Config(n_links=14).__dict__
    m,v,old,tape,x = [load(path) for path in (run/'native_models.npz',run/'virtual_reference.npz',run/'coordinates.npz',run/'adaptive_forecast.npz',diagnostic/'diagnostic_arrays.npz')]
    assert len(m['A'])==1130
    freeze_inputs(out,a)
    assert read(out/'inputs.json')['source_sha256']==manifest['source_sha256']
    helpers = helper_closure({'screen_multiplicative_critic14.py','multiplicative_critic14.py'})
    for name in helpers:
        shutil.copyfile(Path(__file__).with_name(name),out/name)
    proof = dict(protocol='empirical-multiplicative-critic14-screen-v1',config=manifest['config'],
        diagnostic_inventory_sha256=sha(inventory),ledger_inventory_sha256=sha(ledger),max_native_steps=0,
        max_wall_seconds=120,helper_sha256={name:sha(out/name) for name in helpers},
        input_sha256={str(path):sha(path) for path in (run/'native_models.npz',run/'virtual_reference.npz',run/'coordinates.npz',run/'adaptive_forecast.npz',diagnostic/'diagnostic_arrays.npz')},
        fit_relative_phases=list(range(11)),uncertainty_multiplier=1.,actual_added_noise=False,
        source_sha256=manifest['source_sha256'],scope='one empirical multiplicative design/model envelope, no statistical robustness or physical-task claim')
    write_json(out/'frozen_manifest.json',proof)
    geometry = SpatialCartpole(Config(n_links=14))
    try:
        toy = dense_toy_check()
        zero = design(m,np.zeros((74,11)))
        for key in ('S','value_residual','feed'):
            exact_bytes(zero[key],old[key])
        difference = zero['gain']-np.einsum('tij,tjk->tik',old['K'],old['S'][:-1])
        relative = float(np.linalg.norm(difference)/max(1.,np.linalg.norm(zero['gain'])))
        assert relative<1e-12
        print(dict(stage='dense_and_full_zero_uncertainty_checks_passed',gain_reconstruction_relative=relative),flush=True)
        C,fit_proof = fit(m,x,tape['actions'])
        np.savez_compressed(out/'fitted_directions.npz',directions=C,**fit_proof)
        result = None
        def checkpoint(t,c):
            write_json(out/'progress.json',dict(phase=t,computed=int(c['computed'].sum()),native_steps=0,uncertainty_weight=float(c['uncertainty_weight'][t])))
        try:
            c = design(m,C,checkpoint)
        except DesignRangeError as exc:
            c = exc.partial
            result = dict(passed=False,stage='design_range_rejected',phase=exc.phase,reason=str(exc))
        np.savez_compressed(out/'uncertain_critic.npz',**c)
        if result is None:
            result,arrays = envelope(m,v,c,C,geometry)
            np.savez_compressed(out/'model_envelope.npz',**arrays)
        assert geometry.native_steps==0
        metrics = dict(passed=True,links=14,qualified=False,task_success=False,ready_for_native_forecast=result['passed'],
            native_steps=0,new_noisy_samples=0,reserved_cases_consumed=0,original_reset_episodes=0,
            toy_check=toy,full_zero_uncertainty_roots_residuals_feed_byte_exact=True,zero_gain_reconstruction_relative=relative,
            fitted_direction_norm=float(np.linalg.norm(C)),fitted_direction_rank=int(np.linalg.matrix_rank(C)),
            computed_backward_steps=int(c['computed'].sum()),result=result,
            scope='saved-data expected-quadratic design and approximate mean+3sigma model admission only; no simulator execution or qualified noise-arrest claim')
        if actor:
            write_json(out/'summary.json',metrics)
        else:
            original = Path(a.run)
            source_manifest = read(original/'frozen_manifest.json')
            assert source_manifest==proof
            assert (original/'completed.json').exists()
            assert read(original/'summary.json')==metrics
            for name in ('fitted_directions.npz','uncertain_critic.npz')+(() if not (out/'model_envelope.npz').exists() else ('model_envelope.npz',)):
                actual,saved = load(out/name),load(original/name)
                assert set(actual)==set(saved)
                for key in actual:
                    exact_bytes(actual[key],saved[key])
            write_json(out/'summary.json',dict(**metrics,source_summary_sha256=sha(original/'summary.json'),source_manifest_sha256=sha(original/'frozen_manifest.json'),all_screen_arrays_byte_exact=True))
        write_json(out/'completed.json',dict(status='completed'))
        print(read(out/'summary.json'),flush=True)
    except BaseException as exc:
        write_json(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),native_steps=geometry.native_steps,accounting_complete=True))
        raise
    finally:
        geometry.close()


if __name__=='__main__':
    main()
