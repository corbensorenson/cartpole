from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_inventory.json').read_text())
for name,expected in manifest.items():
    p=root/name
    assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
print('All source, controller and evidence hashes verified:',len(manifest))
