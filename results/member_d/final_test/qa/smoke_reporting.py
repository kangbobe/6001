from pathlib import Path
import importlib.util,json,shutil,tempfile,sys
import numpy as np
import pandas as pd

base=Path(__file__).resolve().parent
repo=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path.cwd().resolve()
spec=importlib.util.spec_from_file_location('report',repo/'scripts/report_final_results.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
root=Path(tempfile.mkdtemp(prefix='ca6001-synthetic-report-'))
final=root/'results/experiment/final_test';final.mkdir(parents=True)
val=root/'results/member_d/evaluation';val.mkdir(parents=True)
y=np.array([0,0,0,0,1,0,1,0,1,0,0,1])
thresholds={m:.5 for m in r.MODELS}
pred=pd.DataFrame({'row_id':np.arange(len(y)), 'y_true':y})
metric_rows=[]
for i,m in enumerate(r.MODELS):
    p=np.full(len(y),y.mean()) if m=='dummy' else np.clip(.15+.5*y+np.random.default_rng(i).normal(0,.2,len(y)),0,1)
    pred[m+'_p_yes']=p;pred[m+'_prediction']=(p>=.5).astype(int)
    metric_rows.append({'model':m,**r.independent_metrics(y,p,.5)})
protocol={'frozen':True,'approval_status':'approved','models':r.MODELS,'thresholds':thresholds,'pipeline_sha256':{m:'a'*64 for m in r.MODELS},'cv_recommended_family':'random_forest'}
(final/'frozen_protocol.json').write_text(json.dumps(protocol))
receipt={'status':'complete','protocol_sha256':r.sha256(final/'frozen_protocol.json'),'thresholds':thresholds,'pipeline_sha256':protocol['pipeline_sha256'],'preflight':{'status':'passed','approval_status':'approved'},'n':len(y)}
(final/'evaluation_receipt.json').write_text(json.dumps(receipt))
pd.DataFrame(metric_rows).to_csv(final/'test_metrics.csv',index=False)
pred.to_csv(final/'test_predictions.csv',index=False)
v=pd.DataFrame(metric_rows);v.insert(0,'partition','val');v.insert(2,'rule','selected');v.to_csv(val/'metrics.csv',index=False)
out=root/'derived';r.generate_report(root,out)
assert len([p for p in out.rglob('*') if p.is_file()])==11
assert len(json.loads((out/'derivation_receipt.json').read_text())['checks'])==5
try:r.generate_report(root,out)
except ValueError as e:assert 'overwrite' in str(e)
else:raise AssertionError('Existing output accepted')
# Corrupt a saved decision while retaining probabilities.
bad=pred.copy();bad.loc[0,'random_forest_prediction']=1-bad.loc[0,'random_forest_prediction'];bad.to_csv(final/'test_predictions.csv',index=False)
try:r.generate_report(root,root/'refused')
except ValueError as e:assert 'decisions disagree' in str(e)
else:raise AssertionError('Corrupt saved decision accepted')
assert not (root/'refused').exists()
pred.to_csv(final/'test_predictions.csv',index=False)
# Corrupt a final metric.
badmetrics=pd.DataFrame(metric_rows);badmetrics.loc[1,'f1']+=.1;badmetrics.to_csv(final/'test_metrics.csv',index=False)
try:r.generate_report(root,root/'refused-metric')
except ValueError as e:assert 'metric mismatch' in str(e)
else:raise AssertionError('Corrupt metric accepted')
assert not (root/'refused-metric').exists()
print('PASS: synthetic successful derivation (11 outputs), existing output refusal, decision mismatch refusal, metric mismatch refusal')
print(root)
