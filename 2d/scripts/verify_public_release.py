"""Independently verify public release assets and the count-specific Git tag."""
import argparse
import json
from pathlib import Path
from urllib.request import Request,urlopen

p = argparse.ArgumentParser()
p.add_argument('--links',type=int,required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
expected = json.loads((root/f'output/github_2d_{a.links}_link_receipt.json').read_text())
tag = f'2d-cartpole-{a.links}-link-v1'
assert a.links>0 and expected['tag']==tag
base = 'https://api.github.com/repos/corbensorenson/cartpole'

def get(path):
    request = Request(base+path,headers={'User-Agent':'spatialpole-public-verification',
        'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    with urlopen(request,timeout=60) as response:
        return json.load(response)

release = get('/releases/tags/'+tag)
assert release['html_url']==expected['url'] and not release['draft'] and not release['prerelease']
assets = {x['name']:x for x in release['assets']}
assert {x['name'] for x in expected['assets']}=={
    f'2d-cartpole-{a.links}-link-swingup.mp4',f'2d-cartpole-{a.links}-link-evidence.zip'}
for item in expected['assets']:
    public = assets[item['name']]
    assert public['state']=='uploaded' and public['size']==item['size']
    assert public['digest']=='sha256:'+item['sha256']
    assert public['browser_download_url']==item['url']
obj = get('/git/ref/tags/'+tag)['object']
while obj['type']=='tag':
    obj = get('/git/tags/'+obj['sha'])['object']
assert obj['type']=='commit' and obj['sha']==expected['commit']
report = {'url':release['html_url'],'public':True,'tag_commit':obj['sha'],
          'assets':[{'name':assets[x['name']]['name'],'digest':assets[x['name']]['digest'],
                     'size':assets[x['name']]['size']} for x in expected['assets']]}
(root/f'output/github_2d_{a.links}_link_verification.json').write_text(
    json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
