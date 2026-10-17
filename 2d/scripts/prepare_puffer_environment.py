"""Add this environment to an isolated, pinned PufferLib 5 checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from spatialpole import Config
from spatialpole.env import model_xml

REF = 'ea7f989371270d2948f96728bda8aec5e7634a4c'
ROOT = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser()
p.add_argument('--checkout', required=True)
p.add_argument('--links', type=int, default=1)
a = p.parse_args()
checkout = Path(a.checkout).resolve()
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip() == REF
assert a.links > 0
env_dir = checkout / 'ocean/spatial_cartpole'
config = checkout / 'config/spatial_cartpole.ini'
resource = checkout / 'resources/spatial_cartpole'
driver = checkout / 'build_spatial.sh'
assert not any(x.exists() for x in (env_dir, config, resource, driver))
source_build = (checkout / 'build.sh').read_text(encoding='utf-8')
anchor = 'EXTRA_LDFLAGS=()\n'
assert source_build.count(anchor) == 1
hook = '''
if [ "$ENV" = "spatial_cartpole" ]; then
    : "${MUJOCO_DIR:?Set MUJOCO_DIR to the MuJoCo 3.3.5 Linux SDK directory}"
    : "${MUJOCO_LIBRARY:?Set MUJOCO_LIBRARY to the absolute libmujoco.so.3.3.5 path}"
    EXTRA_LDFLAGS+=("$MUJOCO_LIBRARY" "-Xlinker=-rpath" "-Xlinker=$(dirname "$MUJOCO_LIBRARY")")
fi
'''
# Include hook follows the stock EXTRA_CFLAGS initialization, not before it.
patched = source_build.replace(anchor, anchor + hook, 1)
anchor2 = 'EXTRA_CFLAGS=()\n'
assert patched.count(anchor2) == 1
patched = patched.replace(anchor2, anchor2 + '''
if [ "$ENV" = "spatial_cartpole" ]; then
    EXTRA_CFLAGS+=("-I$MUJOCO_DIR/include")
fi
''', 1)
env_dir.mkdir(parents=True)
resource.mkdir(parents=True)
shutil.copy2(ROOT / 'native/puffer_spatial/spatial_core.h', env_dir / 'spatial_core.h')
header = (ROOT / 'native/puffer_spatial/spatial_cartpole.h').read_text(encoding='utf-8')
(env_dir / 'spatial_cartpole.h').write_text(f'#define SPATIAL_LINKS {a.links}\n' + header, encoding='utf-8', newline='\n')
(resource / 'model.xml').write_text(model_xml(Config(n_links=a.links)), encoding='utf-8', newline='\n')
config.write_text('''[base]
env_name = spatial_cartpole
seed = 73

[vec]
total_agents = 64
num_buffers = 2
num_threads = 4

[env]
model_path = resources/spatial_cartpole/model.xml
angle_noise = 0.02
normalized_effort = 1

[policy]
hidden_size = 64
num_layers = 2

[train]
total_timesteps = 1000000
learning_rate = 0.001
minibatch_size = 2048
horizon = 64
gamma = 0.995
gae_lambda = 0.95
ent_coef = 0.005
''', encoding='utf-8', newline='\n')
driver.write_text(patched, encoding='utf-8', newline='\n')
driver.chmod(driver.stat().st_mode | 0o111)
files = [*env_dir.iterdir(), config, resource/'model.xml', driver]
manifest = dict(puffer_ref=REF, links=a.links, training_started=False,
                files={x.relative_to(checkout).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in files})
(resource / 'integration.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(checkout=str(checkout), links=a.links, files=len(files), training_started=False)))
