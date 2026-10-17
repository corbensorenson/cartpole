"""Pinned, additive fixes and provenance checks for our PufferLib 5 experiment."""
import hashlib
import subprocess
from pathlib import Path

REF = 'ea7f989371270d2948f96728bda8aec5e7634a4c'
PATCH_VERSION = 'spatial-flat-layout-v1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(checkout, *args):
    return subprocess.check_output(['git', '-C', str(checkout), *args], text=True).strip()


def checked_checkout(checkout):
    checkout = Path(checkout).resolve()
    if git(checkout, 'rev-parse', 'HEAD') != REF:
        raise ValueError('PufferLib checkout is not at the pinned commit')
    if git(checkout, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Upstream tracked sources have changed')
    return checkout


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Pinned source anchor is missing or ambiguous: {old}')
    return source.replace(old, new, 1)


def patched_sources(checkout):
    """Keep aligned slabs intact in flat copies/checkpoints and optimizer slices.

    The upstream logical total_elems omits internal 16-byte alignment gaps.
    Leave that counter alone; flattened homogeneous precision buffers must use
    their physical span instead. Mixed-type activation pools are unchanged.
    """
    checkout = checked_checkout(checkout)
    trainer = (checkout / 'src/pufferl.cu').read_text(encoding='utf-8')
    algorithm = (checkout / 'src/algo.cu').read_text(encoding='utf-8')
    for old, new in (
        ('.shape = {pol->params_alloc.total_elems}',
         '.shape = {pol->params_alloc.total_bytes / (long)sizeof(precision_t)}'),
        ('.shape = {grads->total_elems}',
         '.shape = {grads->total_bytes / (long)sizeof(precision_t)}'),
        ('.shape = {pufferl->weight_alloc.total_elems}',
         '.shape = {pufferl->weight_alloc.total_bytes / (long)sizeof(precision_t)}'),
    ):
        trainer = replace_once(trainer, old, new)
    algorithm = replace_once(algorithm,
        '.shape = {param_alloc->total_elems}',
        '.shape = {param_alloc->total_bytes / (long)sizeof(precision_t)}')
    algorithm = replace_once(algorithm,
        '        precision_t* gc_ptr = grads.data + offset;',
        '        // Match alloc_create: each tensor starts at a 16-byte boundary.\n'
        '        offset = (offset + 16 / sizeof(precision_t) - 1)\n'
        '            & ~(16 / sizeof(precision_t) - 1);\n'
        '        precision_t* gc_ptr = grads.data + offset;')
    return {'pufferl.cu': trainer, 'algo.cu': algorithm}


def verify_integration(checkout):
    import json
    checkout = checked_checkout(checkout)
    manifest = json.loads((checkout / 'resources/spatial_cartpole/integration.json').read_text())
    if manifest['puffer_ref'] != REF or manifest.get('training_started') is not False:
        raise ValueError('Invalid integration identity')
    if manifest.get('trainer_patch') != PATCH_VERSION:
        raise ValueError('Prepare a fresh checkout with the flat-layout correction')
    for name, expected in manifest['files'].items():
        path = (checkout / name).resolve()
        if not path.is_relative_to(checkout) or sha256(path) != expected:
            raise ValueError(f'Integration file changed: {name}')
    for name, content in patched_sources(checkout).items():
        if (checkout / 'spatial_patches' / name).read_text(encoding='utf-8') != content:
            raise ValueError(f'Trainer patch differs from pinned transformation: {name}')
    return manifest


def float_weight_layout(links, hidden=64, layers=2):
    """Physical span for this pinned encoder/continuous decoder/MinGRU order."""
    tensors = [('encoder', hidden*(7+7*links)), ('decoder', 3*hidden), ('logstd', 2)]
    tensors += [(f'mingru{i}', 3*hidden*hidden) for i in range(layers)]
    offset = 0; layout=[]
    for name, elements in tensors:
        offset = (offset+15)//16*16
        layout.append(dict(name=name, byte_offset=offset, elements=elements))
        offset += elements*4
    return dict(bytes=offset, float32_elements=offset//4, tensors=layout)
