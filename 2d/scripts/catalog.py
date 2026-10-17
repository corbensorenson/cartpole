"""Inventory local experiment evidence without rerunning simulations."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = []
for path in sorted((root/'output').iterdir()):
    if not path.is_dir() or not (path/'inputs.json').exists() and not (path/'summary.json').exists():
        continue
    summary_file = path/'summary.json' if (path/'summary.json').exists() else path/'interruption.json' if (path/'interruption.json').exists() else path/'failure.json' if (path/'failure.json').exists() else path/'inputs.json'
    summary = json.loads(summary_file.read_text())
    row = {'directory':str(path.relative_to(root)).replace('\\','/'),
        'status':'completed' if (path/'completed.json').exists() else 'interrupted' if (path/'interruption.json').exists() else 'failed' if (path/'failure.json').exists() else 'incomplete',
        'summary_sha256':hashlib.sha256(summary_file.read_bytes()).hexdigest(),
        'passes':summary.get('passes') if isinstance(summary,dict) else None,
        'n_links':summary.get('n_links') if isinstance(summary,dict) else None,
        'source_manifest':str((path/'inputs.json').relative_to(root)).replace('\\','/') if (path/'inputs.json').exists() else None}
    correction = path/'input_correction.json'
    if correction.exists():
        row['input_correction'] = str(correction.relative_to(root)).replace('\\','/')
        row['input_correction_sha256'] = hashlib.sha256(correction.read_bytes()).hexdigest()
    rows.append(row)
receipts = []
for path in sorted((root/'output').glob('github*receipt.json')):
    receipts.append(json.loads(path.read_text()))
(root/'research/experiment_index.json').write_text(json.dumps({'experiments':rows,'publications':receipts},indent=2),encoding='utf-8')
print(f'Indexed {len(rows)} experiments and {len(receipts)} publication receipts; inventory is not a validation certificate.')
