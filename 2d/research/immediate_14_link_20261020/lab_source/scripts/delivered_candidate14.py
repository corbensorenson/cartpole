"""Read-only identity and input contracts for the delivered-feedback planner."""
import hashlib
import json
from pathlib import Path
from spatialpole import Config
from delivered_planner14 import PROTOCOL,MAX_MODELS,ACTOR_NATIVE,AUDIT_NATIVE
from delivered_reference_bank14 import identity
from stage_six_source_synthesis14 import helper_closure
from eleven_synthesis_protocol import read,sha


def numeric_helpers():
    return helper_closure({'delivered_planner14.py','general_delivered_mean14.py','delivered_reference_bank14.py'})


def identity_payload(source_sha,helpers,bank,runtime_sha):
    return dict(protocol=PROTOCOL,config=Config(n_links=14).__dict__,source_sha256=source_sha,
        numerical_helpers_sha256=helpers,portable_profile_identity=bank,solver_receipt_sha256=runtime_sha,
        max_models=MAX_MODELS,max_actor_native_steps=ACTOR_NATIVE,max_audit_native_steps=AUDIT_NATIVE,
        geometry_and_native_merit='original delivered80N/3m/final251tilts<.15/physicalcost<=.8baseline',
        selector='exact102F64bytecache else one nearest fullphysical source, tiebyseed',
        model_schedule='warm constrained plan/delivered mean, <=4 fresh models/fixed then native forecasts, one delivered mean per failed model')


def cid(payload):return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def candidate(folder):
    folder=Path(folder);m=read(folder/'frozen_manifest.json')
    assert (folder/'completed.json').exists() and m['protocol']==PROTOCOL and m['config']==Config(n_links=14).__dict__
    assert cid(m['identity_payload'])==m['controller_identity']
    for name,digest in m['helper_sha256'].items():assert sha(folder/'helpers'/name)==digest==sha(Path(__file__).with_name(name))
    package=Path(__file__).resolve().parents[1]/'src/spatialpole'
    for name,digest in m['source_sha256'].items():assert sha(package/name)==digest
    bank=read(folder/'profiles.json');assert sha(folder/'profiles.json')==m['profiles_sha256']
    assert identity(bank)==m['identity_payload']['portable_profile_identity']
    for meta in bank:
        for path,digest in meta['input_sha256'].items():assert sha(Path(path))==digest
    runtime=Path(m['runtime']);assert sha(runtime/'receipt.json')==m['identity_payload']['solver_receipt_sha256']
    return bank,runtime,m
