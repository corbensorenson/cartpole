"""Build the native core and real PufferLib 5 adapter on Windows, without CUDA."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.request

import mujoco

PUFFER_REF = 'ea7f989371270d2948f96728bda8aec5e7634a4c'
ROOT = Path(__file__).resolve().parents[1]


def build_probe(output, links, evaluation_io=False):
    from setuptools._distutils._msvccompiler import _get_vc_env
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    native_source = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT/'native/puffer_spatial').iterdir() if p.suffix in ('.h','.cpp')}
    deps = output / 'deps'
    deps.mkdir()
    urls = {
        'pufferenv.h': f'https://raw.githubusercontent.com/PufferAI/PufferLib/{PUFFER_REF}/src/pufferenv.h',
        'ini.h': f'https://raw.githubusercontent.com/PufferAI/PufferLib/{PUFFER_REF}/src/ini.h',
        'raylib.h': 'https://raw.githubusercontent.com/raysan5/raylib/5.5/src/raylib.h',
    }
    sources = []
    for name, url in urls.items():
        data = urllib.request.urlopen(url, timeout=30).read()
        (deps / name).write_bytes(data)
        sources.append(dict(path=name, url=url, sha256=hashlib.sha256(data).hexdigest()))
    compiler_env = {**os.environ, **_get_vc_env('x64')}
    # Windows environment keys are case insensitive; avoid duplicate PATH entries.
    compiler_env = {k.lower(): v for k, v in compiler_env.items()}
    compiler = shutil.which('cl.exe', path=compiler_env['path'])
    dumpbin = shutil.which('dumpbin.exe', path=compiler_env['path'])
    librarian = shutil.which('lib.exe', path=compiler_env['path'])
    assert compiler and dumpbin and librarian
    sdk = Path(mujoco.__file__).parent
    exports = subprocess.check_output([dumpbin, '/exports', str(sdk / 'mujoco.dll')], env=compiler_env, text=True)
    names = re.findall(r'^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\w+)\s*$', exports, re.MULTILINE)
    assert 'mj_step' in names and 'mj_loadXML' in names
    definition = output / 'mujoco.def'
    definition.write_text('LIBRARY mujoco.dll\nEXPORTS\n' + '\n'.join(names) + '\n')
    lib = output / 'mujoco.lib'
    commands = [[librarian, '/nologo', '/machine:x64', f'/def:{definition}', f'/out:{lib}']]
    for n in links:
        commands.append([compiler, '/nologo', '/LD', '/O2', '/fp:strict', '/EHsc', '/MD', '/openmp', '/std:c++17',
                         f'/DSPATIAL_LINKS={n}', *(['/DSP_EVALUATION_IO'] if evaluation_io else []), f'/I{sdk / "include"}', f'/I{deps}',
                         f'/I{ROOT / "native/puffer_spatial"}',
                         str(ROOT / 'native/puffer_spatial/verify_bridge.cpp'), str(lib),
                         f'/Fo{output / ("spatial_" + str(n) + ".obj")}',
                         '/link', f'/OUT:{output / ("spatial_" + str(n) + ".dll")}'])
    with (output / 'build.log').open('w', encoding='utf-8') as log:
        for cmd in commands:
            result = subprocess.run(cmd, env=compiler_env, cwd=output, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.write(result.stdout)
            if result.returncode:
                raise RuntimeError(f'Native build failed; see {output / "build.log"}')
    report = dict(puffer_ref=PUFFER_REF, evaluation_io=evaluation_io, dependencies=sources, commands=commands,
                  native_source_sha256=native_source,
                  compiler=compiler, mujoco_version=mujoco.__version__,
                  mujoco_dll_sha256=hashlib.sha256((sdk / 'mujoco.dll').read_bytes()).hexdigest())
    (output / 'build.json').write_text(json.dumps(report, indent=2))
    assert native_source == {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (ROOT/'native/puffer_spatial').iterdir() if p.suffix in ('.h','.cpp')}
    return output


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--links', nargs='+', type=int, default=[1, 2, 8])
    p.add_argument('--evaluation-io', action='store_true')
    a = p.parse_args()
    build_probe(a.out, a.links, a.evaluation_io)
