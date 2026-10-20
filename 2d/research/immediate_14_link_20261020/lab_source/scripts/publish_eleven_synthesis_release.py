"""Upload a fully proved 2D result as a draft, then publish after digest checks."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from eleven_synthesis_release_gate import qualified_record
from eleven_synthesis_protocol import read, sha

REPOSITORY = 'corbensorenson/cartpole'
API = 'https://api.github.com/repos/'+REPOSITORY


class Github:
    def __init__(self):
        result = subprocess.run(['git', 'credential', 'fill'],
            input='protocol=https\nhost=github.com\n\n', text=True,
            capture_output=True, check=True)
        values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        self.token = values['password']

    def request(self, url, method='GET', data=None, file=None, content_type='application/json'):
        address = urlparse(url)
        assert address.scheme == 'https' and address.hostname in ('api.github.com', 'uploads.github.com')
        headers = {'Authorization':'Bearer '+self.token, 'Accept':'application/vnd.github+json',
            'X-GitHub-Api-Version':'2022-11-28', 'User-Agent':'2d-cartpole-observed-synthesis',
            'Content-Type':content_type}
        if file is not None:
            path = Path(file); headers['Content-Length'] = str(path.stat().st_size)
            with path.open('rb') as stream:
                request = Request(url, data=stream, method=method, headers=headers)
                with urlopen(request, timeout=1200) as response:
                    return json.load(response)
        if data is not None:
            data = json.dumps(data).encode('utf-8')
        with urlopen(Request(url, data=data, method=method, headers=headers), timeout=120) as response:
            return json.load(response)

    def assets(self, url):
        result = {}; page = 1
        while True:
            rows = self.request(url+f'?per_page=100&page={page}')
            for row in rows:
                assert row['name'] not in result
                result[row['name']] = row
            if len(rows) < 100:
                return result
            page += 1


def release_body(manifest, commit):
    m = manifest
    return f'''Qualified **11-link 2D immediate noisy swing-up**: **32/32 fresh starts**
after **45/45 development cases**, with zero hanging preparation. Every episode
completes 30 seconds and holds all rods within 0.15 rad of upright for at least
{m['minimum_hold_seconds']:.2f} seconds. The complete first-reserved 50 fps video
is attached, together with source, runtime and evidence archives.

The cart is free in XY and every joint is an unrestricted passive ball joint.
All 35 physical velocities, including spin, remain in the model and feedback.
The original 80 N radial force limit, 3 m square, 50 Hz commands and 200 Hz
MuJoCo RK4 physics are unchanged.

The fixed algorithm plans a full trajectory from the observed initial state
and then applies feedback during motion. Planning is offline and may take
minutes; the physical episode is not advanced during planning. This is
full-state classical control. Hardware real-time operation and additional
mid-swing force-gust robustness are unqualified.

Every qualification case passes independent reset, model, physical-rate,
feedback, substep-boundary and cold-replay audits. The first reserved case
also repeats its full synthesis and actual tape from the extracted runtime.
The physical package matches the existing 294-test source fingerprint; the
new synthesis and archive evidence has its own independent checks.

The 13 completed pilot cases are reused in development without repeating
their physics. Qualification seeds 20263516–20263547 were reserved before
testing and are reproduction data after this qualification. Failed forecasts,
internal continuation trials and corrections in this synthesis family remain
in the evidence. The 14-link conditioned comparison baseline and existing
10-link immediate result remain preserved.

[Study and reproduction](https://github.com/{REPOSITORY}/blob/{commit}/2d/research/immediate_11_link_20261018/README.md)

Video SHA256: `{m['video']['sha256']}`
Controller identity: `{m['controller_identity']}`
'''


def find_release(github, tag):
    """GitHub's tag endpoint omits drafts; find an existing draft before POST."""
    try:
        return github.request(API+'/releases/tags/'+quote(tag, safe=''))
    except HTTPError as error:
        if error.code != 404:
            raise
    matches=[];page=1
    while True:
        rows=github.request(API+f'/releases?per_page=100&page={page}')
        matches.extend(row for row in rows if row['tag_name']==tag)
        if len(rows)<100:
            break
        page+=1
    assert len(matches)<=1, 'Multiple releases share this tag; inspect rather than creating another draft'
    return matches[0] if matches else None


def same_published_proof(key, published, current):
    """Allow relocated asset aliases only when all proof data and file identity agree."""
    if published == current:
        return True
    if key not in ('runtime_archive', 'video') or not isinstance(published, dict) or not isinstance(current, dict):
        return False
    if published.keys() != current.keys() or 'path' not in published:
        return False
    if any(published[name] != current[name] for name in published if name != 'path'):
        return False
    old, new = Path(published['path']), Path(current['path'])
    return old.is_file() and new.is_file() and old.samefile(new)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--qualification-run', required=True)
    p.add_argument('--qualification-audit', required=True); p.add_argument('--runtime-receipt', required=True)
    p.add_argument('--first-reserved-reproduction', required=True); p.add_argument('--video', required=True)
    p.add_argument('--video-qa', required=True); p.add_argument('--commit', required=True)
    p.add_argument('--publication-repo')
    p.add_argument('--dry-run', action='store_true'); a = p.parse_args()
    # Reject incomplete science before reading publication assets or credentials.
    base = qualified_record(a.qualification_run, a.qualification_audit, a.runtime_receipt,
        a.first_reserved_reproduction, a.video, a.video_qa)
    assert re.fullmatch('[0-9a-f]{40}', a.commit)
    root = Path(__file__).resolve().parents[1]; tag = base['tag']
    manifest_path = root/f'output/{tag}-manifest.json'; manifest = read(manifest_path)
    bundle = read(root/f'output/{tag}-bundle-receipt.json')
    for key, value in base.items():
        assert same_published_proof(key, manifest[key], value), f'Published manifest proof drift: {key}'
    assert bundle['qualified'] and bundle['all_zip_hashes_and_crc_verified']
    assert bundle['manifest_sha256'] == sha(manifest_path)
    descriptions = [manifest['video'], manifest['runtime_archive'], *bundle['archives']]
    descriptions.append(dict(name=manifest_path.name, path=str(manifest_path),
                             sha256=sha(manifest_path), bytes=manifest_path.stat().st_size))
    assert len({row['name'] for row in descriptions}) == len(descriptions)
    for row in descriptions:
        path = Path(row['path'])
        assert path.name == row['name'] and path.stat().st_size == row['bytes']
        assert sha(path) == row['sha256']
    body = release_body(manifest, a.commit)
    publication = Path(a.publication_repo) if a.publication_repo else root/'publication'
    study = '2d/research/immediate_11_link_20261018/'
    public_manifest = subprocess.check_output(['git', 'show', a.commit+':'+study+'manifest.json'], cwd=publication)
    assert json.loads(public_manifest) == manifest, 'Publication commit does not contain the proved manifest'
    subprocess.run(['git', 'cat-file', '-e', a.commit+':'+study+'README.md'], cwd=publication, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    expected_blob = hashlib.sha1(b'blob '+str(len(public_manifest)).encode()+b'\0'+public_manifest).hexdigest()
    if a.dry_run:
        print(json.dumps(dict(dry_run=True, tag=tag, assets=[{k:r[k] for k in ('name', 'bytes', 'sha256')}
            for r in descriptions], body=body)), flush=True)
        return
    github = Github()
    ref = github.request(API+'/git/ref/tags/'+quote(tag, safe=''))['object']
    for _ in range(4):
        if ref['type'] == 'commit':
            break
        assert ref['type'] == 'tag'
        ref = github.request(API+'/git/tags/'+ref['sha'])['object']
    assert ref['type'] == 'commit' and ref['sha'] == a.commit
    remote_manifest = github.request(API+'/contents/'+study+'manifest.json?ref='+a.commit)
    assert remote_manifest['type'] == 'file' and remote_manifest['sha'] == expected_blob
    release = find_release(github, tag)
    if release is None:
        release = github.request(API+'/releases', 'POST', dict(tag_name=tag, target_commitish=a.commit,
            name='2D Cartpole — 11 Links: Immediate Noisy Swing-Up', body=body,
            draft=True, prerelease=False, make_latest='false'))
    assert release['tag_name'] == tag and not release['prerelease']
    assets = github.assets(release['assets_url']); verified = []
    for row in descriptions:
        path = Path(row['path'])
        if row['name'] not in assets:
            content_type = 'video/mp4' if path.suffix == '.mp4' else 'application/zip' if path.suffix == '.zip' else 'application/json'
            print(json.dumps(dict(uploading=row['name'], bytes=row['bytes'])), flush=True)
            assets[row['name']] = github.request(release['upload_url'].split('{')[0]+'?name='+quote(row['name'], safe=''),
                'POST', file=path, content_type=content_type)
        asset = assets[row['name']]
        assert asset['state'] == 'uploaded' and asset['size'] == row['bytes']
        assert asset.get('digest') == 'sha256:'+row['sha256'], 'Server digest differs; never overwrite an existing asset'
        verified.append(dict(name=row['name'], bytes=row['bytes'], sha256=row['sha256'], url=asset['browser_download_url']))
    # Uploads resume by matching immutable names/digests; no asset is deleted.
    if release['draft'] or release.get('body') != body:
        release = github.request(API+'/releases/'+str(release['id']), 'PATCH',
            dict(draft=False, body=body, make_latest='false'))
    assert not release['draft'] and not release['prerelease']
    final_assets = github.assets(release['assets_url'])
    for row in verified:
        assert final_assets[row['name']].get('digest') == 'sha256:'+row['sha256']
        row['url'] = final_assets[row['name']]['browser_download_url']
    receipt = dict(tag=tag, commit=a.commit, url=release['html_url'], assets=verified,
        server_digests_verified=True, controller_identity=manifest['controller_identity'],
        qualified=True, immediate_links=11, conditioned_comparison_links=14)
    target = root/'output/github_2d_11_link_immediate_receipt.json'
    if target.exists():
        assert read(target) == receipt
    else:
        target.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
