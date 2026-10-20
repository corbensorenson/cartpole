"""Read-only source bundle; exact-state zero identity never edits native data."""
from pathlib import Path
import numpy as np
from eleven_synthesis_protocol import read,sha

def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}

def bundle(path):
    path=Path(path);summary=read(path/'summary.json')
    assert summary['passed'] and summary['links']==14 and (path/'completed.json').exists()
    assert sha(path/'reference.npz')==summary['reference_sha256']
    assert sha(path/'model/native_models.npz')==summary['native_model_sha256']
    v,m,c,overlay,expected=[load(path/name) for name in ('reference.npz','model/native_models.npz',
        'coordinates.npz','coherent_overlay.npz','source_policy_tape.npz')]
    assert m['predicted'].tobytes()==v['states'][1:].tobytes()
    assert m['reference_states'].tobytes()==v['states'].tobytes()
    assert m['reference_actions'].tobytes()==v['actions'].tobytes()
    assert overlay['original_d'].tobytes()==m['d'].tobytes() and np.all(overlay['d']==0.)
    assert np.all(c['feed']==0.) and np.all(c['value_residual']==0.)
    m['d']=overlay['d']
    return dict(v=v,m=m,c=c,w=overlay['w'],expected=expected)
