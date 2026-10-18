"""Plots of development tuning and convergence, not final test comparisons."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from src.data import ROOT


def generate_model_plots(experiment=ROOT / 'results/experiment'):
    experiment = Path(experiment)
    destination = experiment / 'figures'
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white'})
    for group, names in [('B_baselines',['logistic_regression','decision_tree']),
                         ('C_advanced',['random_forest','mlp'])]:
        fig, axes = plt.subplots(1,2,figsize=(12,4.5), layout='constrained')
        for ax, name in zip(axes,names):
            table = pd.read_csv(experiment/name/'cv_candidates.csv')
            ax.plot(table.config_id, table.mean_train_ap, 's--', color='#9b6a26', label='Training AP (mean)')
            ax.errorbar(table.config_id, table.mean_average_precision, yerr=table.std_average_precision,
                        fmt='o-', capsize=3,color='#2166a0',label='CV AP (mean ± fold SD)')
            warned = table.loc[table.unconverged_folds > 0]
            if not warned.empty:
                ax.scatter(warned.config_id, warned.mean_average_precision, marker='x', s=70,
                           color='#b2182b', zorder=4, label='Final warning in at least one fold')
            ax.set(title='MLP' if name == 'mlp' else name.replace('_',' ').title(), xlabel='Configuration ID (see CSV for parameters)', ylabel='Average precision', ylim=(0,1))
            ax.set_xticks(table.config_id)
            ax.grid(axis='y',alpha=.2)
            ax.legend(fontsize=9,loc='upper left')
        fig.suptitle('Training-only tuning | 3 fixed stratified folds | fold SD is not a confidence interval',fontsize=11)
        fig.savefig(destination/f'{group}_tuning.png',dpi=170)
        plt.close(fig)
    metadata = json.loads((experiment/'mlp/model_metadata.json').read_text(encoding='utf-8'))
    fig, ax = plt.subplots(figsize=(8,4.5), layout='constrained')
    for i, attempt in enumerate(metadata['refit_attempts']):
        loss = attempt['loss_curve']
        ax.plot(range(1,len(loss)+1),loss,label=f'Attempt {i+1}: cap={attempt["max_iter"]}',
                color=['#2166a0','#9b6a26'][i],linestyle=['-','--'][i])
    last = metadata['refit_attempts'][-1]
    ax.set(title=f'Selected MLP: training loss on 27,126 records\niterations={last["n_iter"]}; convergence warning={last["convergence_warning"]}',
           xlabel='Training epoch',ylabel='Training objective (cross-entropy + regularization)')
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(destination/'MLP_training_loss.png',dpi=170)
    plt.close(fig)
    return destination
