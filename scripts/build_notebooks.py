"""Build and execute portable companion notebooks from the completed experiment."""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model_plots import generate_model_plots


def md(text):
    return nbf.v4.new_markdown_cell(inspect.cleandoc(text))


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


SETUP = '''
from pathlib import Path
import sys, json
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
from threadpoolctl import threadpool_limits
from src.data import load_partition, audit_data
from src.metrics import binary_metrics
import joblib
EXPERIMENT = ROOT / 'results/experiment'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
'''


def notebook(cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {'display_name':'Python 3','language':'python','name':'python3'}
    nb.metadata.language_info = {'name':'python','version':sys.version.split()[0]}
    return nb


def create_notebooks():
    A = notebook([
        md('''# A — Data, quality and preprocessing
        ## tl;dr
        This notebook recomputes training-only data checks and exploratory plots from the locked UCI legacy CSV.
        The training partition contains 27,126 records with 3,173 positive outcomes (11.697%).
        The research target is a retrospective simulation of pre-campaign customer screening. It is not a causal marketing-effect estimate.
        Exact `pdays` is replaced by prior-contact status; current-campaign fields are excluded. See the computed tables below for observed values.
        '''),
        md('''## Context & Methods
        Source: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank), legacy `bank-full.csv`; CC BY 4.0.
        Frozen train / validation / test sizes: 27,126 / 9,042 / 9,043. All target relationships below use train only.
        ### Key Assumptions
        Records have no customer identifier or verified decision-time snapshots. Random splits assess historical record-level generalization.
        `unknown` is a category, negative balances are valid observations, and matching profiles are not proof of duplicate customers.
        '''), code(SETUP),
        md('## Data\n### 1. Validate the source and fixed partitions'),
        code('''audit = audit_data()
print('Source shape:', audit['raw_shape'])
print('Partition sizes:', audit['partition_sizes'])
display(pd.Series(audit['train'], name='Training audit').to_frame())
X, y = load_partition('train')
assert X.index.equals(y.index)
from src.preprocessing import HistoryFeatures
duplicate_counts=pd.DataFrame({
    'profile':['16 original inputs','17 original columns including target',
               '11 source fields with exact pdays','11 fields after pdays becomes contact flag'],
    'excess_rows':[audit['train']['exact_16_input_duplicates'],audit['train']['exact_17_column_duplicates'],
                   int(X.duplicated().sum()),int(HistoryFeatures().fit_transform(X).duplicated().sum())]})
display(duplicate_counts)
print('Excess rows = rows minus unique profiles. Matching profiles do not identify customers; no rows removed.')
'''),
        md('## Results\n### 2. Inspect class balance'),
        code('''counts = y.value_counts().reindex([0,1])
fig, ax = plt.subplots(figsize=(6,3.5), layout='constrained')
bars=ax.bar(['No','Yes'], counts, color=['#657786','#2166a0'])
ax.bar_label(bars, labels=[f'{v:,} ({v/len(y):.1%})' for v in counts])
ax.set(title=f'Train subscription outcome (n={len(y):,})',ylabel='Records',ylim=(0,counts.max()*1.2))
plt.show()
print(f'Training positive rate = {y.mean():.4%}. Accuracy must be read alongside minority-class metrics.')
'''),
        md('### 3. Examine distributions without clipping observations'),
        code('''fig, axes = plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
axes[0].hist(X.age,bins=30,color='#2166a0'); axes[0].set(title='Training age',xlabel='Age (years)',ylabel='Records')
axes[1].ecdf(X.balance,color='#9b6a26')
axes[1].set_xscale('symlog',linthresh=100)
axes[1].set(title='Training annual balance distribution',xlabel='Balance (EUR; symmetric-log axis)',ylabel='Cumulative share of records')
plt.show()
display(X[['age','balance','previous']].describe().round(2))
'''),
        md('### 4. Prior outcomes and observed subscription rates'),
        code('''rates = X.assign(y=y).groupby('poutcome',observed=True).agg(n=('y','size'),rate=('y','mean')).sort_values('rate')
fig, ax=plt.subplots(figsize=(8,3.8),layout='constrained')
bars=ax.barh(rates.index,rates.rate,color='#2166a0')
ax.bar_label(bars,labels=[f'{v.rate:.1%}; n={int(v.n):,}' for v in rates.itertuples()],padding=5)
ax.set(xlim=(0,1),xlabel='Observed subscription rate',title='Previous-campaign outcome | train only')
plt.show()
display(rates)
'''),
        md('''### 5. Fit a preprocessing example on train only
        Literal `unknown` is retained as an ordinary category. Only categories absent from the current fitting
        subset become an all-zero one-hot block when encountered later; the record is retained.
        '''),
        code('''from src.preprocessing import build_preprocessor
preprocessor=build_preprocessor(scale_numeric=True)
encoded=preprocessor.fit_transform(X)
names=preprocessor.get_feature_names_out()
assert encoded.shape[0]==len(y) and np.isfinite(encoded).all()
assert not any('pdays' in n or 'duration' in n for n in names)
print('Raw source fields:', X.shape[1], '; transformed inputs:', encoded.shape[1])
display(pd.DataFrame({'encoded_feature':names}))
'''),
        md('''## Takeaways
        The exact training counts and category summaries above are descriptive associations, not effects of marketing or customer identity evidence.
        Use the shared preprocessing constructor inside each CV fit; the illustrative fitted transformer here is never fed into model tuning.
        No test prediction or test target-association analysis runs in this notebook. Full CSV tables and report-ready figures are under `results/eda`.
        中文交接：字段可得性、数据来源、保留异常值的原因，以及划分限制应保留在报告中。
        ''')
    ])
    generated = [('01_A_data', A)]
    for prefix,title,names,description in [
        ('02_B_baselines','B — Logistic regression and decision tree',['logistic_regression','decision_tree'],
         'Logistic regression models linear log-odds with L2 regularization: a useful interpretable baseline, but limited for nonlinear interactions. A decision tree recursively partitions features by impurity reduction: it captures interactions but can overfit; depth and minimum leaf size control complexity.'),
        ('03_C_advanced','C — Random forest and MLP',['random_forest','mlp'],
         'Random forest averages randomized bootstrap trees to reduce variance, at the cost of a larger and less transparent model. The MLP learns nonlinear representations with ReLU hidden layers and a sigmoid output, using weighted cross-entropy, L2 regularization and Adam; scaling, initialization and convergence matter.')]:
        cells=[md(f'# {title}\n## tl;dr\nThis executed companion reads the completed training experiment and independently reproduces validation predictions from saved pipelines. It does not silently rerun the hyperparameter search or evaluate test.'),
               md(f'## Context & Methods\n{description}\n\n### Key Assumptions\nSame fixed train folds, AP selection, and fold-local balanced weights for all families. Validation metrics use p≥0.5 and are development diagnostics. Three-fold SD is not a confidence interval. D chooses thresholds and conducts the final test. Total search: 36 configurations across four families, 108 base CV fits.'),
               code(SETUP),
               md('''## Data
               The original 11 fields enter an importable pipeline; exact `pdays` becomes prior-contact status. Scaling is used for LR/MLP.
               No learned transformation is fitted outside each CV training subset.
               '''),
               code("X_val,y_val=load_partition('val')\nprint('Validation records:',len(y_val),'(not final test)')"),
               md('### Inspect the actual model constructor and search space'),
               code(f"from src.models import search_configs,build_model\nfrom src.training import fit_candidate\nimport inspect\nnames={names!r}\nfor name in names:\n    print(name)\n    display(pd.DataFrame([{{k: ('None' if v is None else v) for k,v in cfg.items()}} for cfg in search_configs(name)]))\nprint(inspect.getsource(build_model))"),
               md('''### Reproduce training explicitly
               From the project root with the supplied environment:
               ```powershell
               .venv\\Scripts\\python.exe scripts/train.py --output results/rerun --jobs 2
               ```
               The output directory must be new. This command reruns all fixed configurations and records every fold, warning and refit. It produces no test predictions or performance metrics; the source CSV is read for structural checks.
               Below, the completed `results/experiment` is inspected to keep notebook execution fast and deterministic.
               '''),
               md('## Results\n### Selected parameters and training-fold scores'),
               code('''metadata={}
for name in names:
    metadata[name]=json.loads((EXPERIMENT/name/'model_metadata.json').read_text(encoding='utf-8'))
    print(name,metadata[name]['selected_params'])
    table=pd.read_csv(EXPERIMENT/name/'cv_candidates.csv')
    display(table[['config_id','mean_average_precision','std_average_precision','mean_train_ap','unconverged_folds']].rename(columns={
        'mean_average_precision':'CV AP mean','std_average_precision':'CV AP SD',
        'mean_train_ap':'Train AP','unconverged_folds':'Unconverged folds'}).round(5))
'''),
               md('### Reload and verify all validation predictions'),
               code('''metric_rows=[]
for name in names:
    model=joblib.load(EXPERIMENT/name/'pipeline.joblib')
    with threadpool_limits(limits=1):
        p=model.predict_proba(X_val)[:,1]
    saved=pd.read_csv(EXPERIMENT/name/'validation_predictions.csv')
    np.testing.assert_array_equal(saved.row_id.to_numpy(),X_val.index.to_numpy())
    np.testing.assert_array_equal(saved.y_true.to_numpy(),y_val.to_numpy())
    np.testing.assert_allclose(saved.p_yes,p,rtol=1e-12,atol=1e-12)
    metric_rows.append({'model':name,**binary_metrics(y_val,p)})
metrics=pd.DataFrame(metric_rows).set_index('model')
display(metrics[['average_precision','roc_auc','precision','recall','f1','accuracy']].rename(columns={
    'average_precision':'AP','roc_auc':'ROC-AUC','precision':'Precision','recall':'Recall','f1':'F1','accuracy':'Accuracy'}).round(5))
display(metrics[['tn','fp','fn','tp','brier_score','log_loss']].rename(columns={
    'brier_score':'Brier score','log_loss':'Log loss'}).round(5))
print('All saved validation rows, labels and probabilities reproduced.')
'''),
               md('### Training versus CV tuning diagnostics'),
               code(f"from src.model_plots import generate_model_plots\ngenerate_model_plots(EXPERIMENT)\ndisplay(Image(filename=str(EXPERIMENT/'figures/{'B_baselines' if prefix.startswith('02') else 'C_advanced'}_tuning.png')))"),
        ]
        if prefix.startswith('03'):
            cells.extend([md('### MLP convergence\nInternal accuracy-based early stopping is disabled. Every capped fit with ConvergenceWarning restarts once at 600 epochs, with the same seed. The trace is a training objective, not held-out AP.'),
                code("""display(Image(filename=str(EXPERIMENT/'figures/MLP_training_loss.png')))
folds=pd.read_csv(EXPERIMENT/'mlp/cv_folds.csv')
print(f'MLP CV folds: {len(folds)}; additional retry fits: {int((folds.attempt_count-1).sum())}; final unconverged folds: {int(folds.final_convergence_warning.sum())}')
selected=folds[folds.config_id==metadata['mlp']['config_id']]
display(selected[['config_id','fold_id','attempt_count','n_iter','final_convergence_warning']])
for attempt in metadata['mlp']['refit_attempts']:
    print({k:v for k,v in attempt.items() if k!='loss_curve'})
""")])
        cells.append(md('''## Takeaways
        The tables show the results actually produced by the frozen experiment. A larger training-to-CV AP gap is evidence of overfitting under this protocol, not proof of future failure.
        Weighted model outputs are not assumed to be calibrated probabilities; 0.5 is a comparison convention rather than an optimized business threshold.
        The complete model and validation CSV are ready for D. D retains threshold selection, test evaluation, error analysis, feature importance and final cross-model conclusions.
        中文交接：这里的成绩属于开发阶段；不要改名为测试集成绩，也不要据此补调参数。
        '''))
        generated.append((prefix,notebook(cells)))
    return generated


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args=parser.parse_args()
    if args.execute:
        generate_model_plots()
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    for name, nb in create_notebooks():
        target=ROOT/'notebooks'/f'{name}.ipynb'
        nbf.validate(nb)
        if args.execute:
            NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
        nbf.write(nb,target)
        if args.execute:
            html,_=HTMLExporter().from_notebook_node(nb)
            target.with_suffix('.html').write_text(html,encoding='utf-8')
        print(f'{name}: saved; executed={args.execute}',flush=True)
