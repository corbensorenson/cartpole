"""Additional read-only evidence for administratively interrupted native work."""
from pathlib import Path
import numpy as np
from eleven_synthesis_protocol import read,sha
from spatialpole import Config
from spatial_synthesis_geometry import Dimensions


def validate_resume_manifest(case):
    case=Path(case);a=read(case/'administrative_resume_manifest.json');old=Path(a['original_case'])
    m=read(case/'frozen_manifest.json');config=Config(**m['config'])
    assert not (old/'completed.json').exists() and not (old/'actual.npz').exists()
    assert sha(old/'frozen_manifest.json')==sha(case/'frozen_manifest.json')==a['original_frozen_manifest_sha256']
    i=a['start_iteration'];assert 0<=i<4 and a['completed_models_reused']==i
    assert a['completed_actual_episodes_repeated']==a['new_noise_labels']==0
    assert a['interrupted_native_steps_lower_bound']==0 and a['interrupted_native_steps_upper_bound']==Dimensions.from_config(config).model_native_steps
    assert a['uncheckpointed_work_exact_count_unknown']
    partial=old/'planning'/f'iteration{i}';assert {p.name for p in partial.iterdir()}=={'virtual_reference.npz'}
    assert sha(partial/'virtual_reference.npz')==a['interrupted_virtual_reference_sha256']
    if (case/'planning'/f'iteration{i}'/'virtual_reference.npz').exists():
        with np.load(partial/'virtual_reference.npz') as x,np.load(case/'planning'/f'iteration{i}'/'virtual_reference.npz') as y:
            assert set(x.files)==set(y.files)
            for key in x.files:assert x[key].tobytes()==y[key].tobytes()
    for j in range(i):
        prior=old/'planning'/f'iteration{j}';assert (prior/'completed.json').exists()
        for path in prior.iterdir():
            if path.is_file():assert sha(path)==sha(case/'planning'/prior.name/path.name)
    for name,digest in m['source_sha256'].items():assert sha(case/'frozen_package/spatialpole'/name)==digest
    for name,digest in m['helper_sha256'].items():assert sha(case/'helpers'/name)==digest
    assert sha(case/'helpers/resume_spatial_native_plan.py')==a['resumer_loop_sha256']
    source=(case/'helpers/spatial_observed_synthesis.py').read_text()
    start=source.index('            for iteration in range(MAX_MODELS):',source.index('def synthesize('))
    end=source.index('        assert total() <= max_native and geometry.native_steps == 0',start)
    expected='\n'.join(v[4:] if v.startswith('    ') else v for v in source[start:end].splitlines())
    resumed=(case/'helpers/resume_spatial_native_plan.py').read_text()
    start=resumed.index('        for iteration in range(start_iteration, MAX_MODELS):')
    end=resumed.index('        assert total()<=max_native',start)
    assert '\n'.join(expected.splitlines()[1:]).strip()=='\n'.join(resumed[start:end].splitlines()[1:]).strip()
    notice=Path('D:/ChatGPT/cartpole/.workspace/consolidation-20261018/interruption-notice.json')
    assert sha(notice)==a['stop_notice_sha256']
    record=Path(read(notice)['processRecord']);assert sha(record)==a['stopped_process_record_sha256']
    assert a['stopped_worker'] in read(record) and a['stopped_worker']['ProcessId']==read(old/'worker.json')['pid']
    return dict(passed=True,manifest_sha256=sha(case/'administrative_resume_manifest.json'),
        original_case=str(old),completed_models_reused=i,completed_episodes_repeated=0,
        uncheckpointed_interrupted_native_steps_bounds=[0,a['interrupted_native_steps_upper_bound']],
        controller_numerical_loop_unchanged=True)


def completed_resume_evidence(case,summary):
    proof=validate_resume_manifest(case)
    assert summary['administrative_resumption'] and summary['administrative_resume_manifest_sha256']==proof['manifest_sha256']
    known=summary['inclusive_native_steps'];assert summary['known_completed_path_native_steps']==known
    assert summary['actual_total_native_steps_bounds']==[known,known+proof['uncheckpointed_interrupted_native_steps_bounds'][1]]
    assert not summary['native_step_total_exact']
    return proof
