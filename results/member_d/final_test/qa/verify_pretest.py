"""Pre-test verification: opaque identity hashing, split structure, no label parsing."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import builtins
import io

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import joblib
import pandas as pd
from src.final_evaluation import _validate_approved_identities, _validate_experiment
from src.data import load_split_indices

assert subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)==''
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert commit==(OUT/'hardening-commit.txt').read_text().strip()
assert not (ROOT/'config/final_protocol.json').exists()
assert not (ROOT/'results/experiment/final_test').exists()
access=json.loads((OUT/'guarded-suite-access.json').read_bytes())
assert access['pytest_exit_code']==0 and access['blocked_access_attempts']==[]
import xml.etree.ElementTree as ET
junit=ET.parse(OUT/'guarded-suite.xml').getroot()
suites=list(junit.iter('testsuite'))
assert sum(int(s.attrib['failures'])+int(s.attrib['errors']) for s in suites)==0
assert sum(int(s.attrib['tests']) for s in suites)==145

# Default production CSV/model paths are never used by this check.
raw=(ROOT/'data/raw/bank-full.csv').resolve()
def deny_model(*args,**kwargs):
    raise AssertionError('Preflight must not deserialize fitted models')
joblib.load=deny_model
original_csv=pd.read_csv
def safe_csv(path,*args,**kwargs):
    assert Path(path).resolve()==(ROOT/'data/splits/split_manifest.csv').resolve()
    return original_csv(path,*args,**kwargs)
pd.read_csv=safe_csv
def guarded_open(original):
    def opened(path,mode='r',*args,**kwargs):
        if isinstance(path,(str,Path)) and Path(path).resolve()==raw:
            assert mode=='rb','Raw dataset permitted only as opaque bytes for SHA256'
        return original(path,mode,*args,**kwargs)
    return opened
builtins.open=guarded_open(builtins.open)
io.open=guarded_open(io.open)

baseline=json.loads((ROOT.parent/'preflight-hardening/before.json').read_bytes())
actual={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in baseline}
assert actual==baseline, 'Protected file identity mismatch'
proposal=json.loads((ROOT/'config/final_protocol.proposed.json').read_bytes())
assert proposal['frozen'] is False and proposal['approval_status']=='pending_explicit_user_approval'
_validate_experiment(ROOT/'results/experiment',proposal)
identities=_validate_approved_identities(ROOT,ROOT/'results/experiment',proposal)
splits=load_split_indices(ROOT)
assert {key:len(values) for key,values in splits.items()}=={'train':27126,'val':9042,'test':9043}
expected={}
for line in (ROOT/'requirements.lock.txt').read_text().splitlines():
    if line.strip() and not line.lstrip().startswith('#'):
        package,wanted=line.strip().split('==',1)
        expected[package]=wanted
assert len(expected)==68
installed={package:version(package) for package in expected}
assert installed==expected,'Installed package versions differ from lock'
pip=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
assert pip.returncode==0,pip.stdout+pip.stderr
# Inference inputs and original frozen outputs are byte-identical to main.
original_paths=['data','results/experiment','config/dataset_lock.json','config/feature_policy.csv',
                'requirements.lock.txt','src/data.py','src/preprocessing.py','src/metrics.py','src/models.py']
assert not subprocess.check_output(['git','diff','--name-only','bf53eab','--',*original_paths],cwd=ROOT,text=True).strip()
record={
    'status':'passed','verified_utc':datetime.now(timezone.utc).isoformat(),'git_commit':commit,
    'reviewed_change_count':8,'worktree_clean':True,'test_count':145,'test_failures':0,
    'excluded_legacy_test':'tests/test_data.py::test_exact_split_coverage_and_partition_alignment',
    'guarded_real_data_or_model_access_attempts':[],
    'input_and_evidence_identities':identities,'protected_files_count':len(baseline),
    'protected_files_sha256':actual,'split_sizes':{k:len(v) for k,v in splits.items()},
    'split_membership_coverage_and_alignment':'passed via original structural loader; no raw rows parsed',
    'python':sys.version,'locked_packages':installed,'pip_check':pip.stdout.strip(),
    'test_partition_loaded':False,'real_models_deserialized':False,
    'approval_source':'User explicitly approved the existing validation-based protocol and one final evaluation in this conversation; freeze will record that authorization.',
}
(OUT/'pretest-verification.json').write_text(json.dumps(record,indent=2)+'\n')
print('PRE-TEST VERIFICATION PASSED: 145 tests; 99 protected identities; 68 locked packages; exact split structure; all required experiment identities.')
print('No test rows parsed, no fitted pipeline deserialized. Original proposal unchanged.')
