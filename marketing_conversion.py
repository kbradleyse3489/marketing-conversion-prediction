"""Reproducible continuation of Keisha Walker's uploaded Steps 1–6.
Run: python marketing_conversion.py. No deployment or customer decisions.
"""
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import hashlib
import json
import os
import platform
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'conversion-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
FEATURES = ['job', 'education', 'default', 'housing', 'loan', 'previous', 'poutcome']
CATEGORICAL = [c for c in FEATURES if c != 'previous']
URL = 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'
SEED = 42
EXPECTED_DATA_SHA256 = '74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8'

def load_data():
    path = ROOT / 'data/bank-additional-full.csv'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with urlopen(URL, timeout=60) as response:
            archive = ZipFile(BytesIO(response.read()))
        inner = ZipFile(BytesIO(archive.read('bank-additional.zip')))
        path.write_bytes(inner.read('bank-additional/bank-additional-full.csv'))
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != EXPECTED_DATA_SHA256:
        raise ValueError('Dataset checksum changed; review source before running evaluation.')
    return pd.read_csv(path, sep=';'), checksum

def split_data(df):
    """Ordered 60/20/20 split, matching original script; no shuffle."""
    n = len(df)
    return {'train': df.iloc[:int(n*.6)].copy(),
            'validation': df.iloc[int(n*.6):int(n*.8)].copy(),
            'test': df.iloc[int(n*.8):].copy()}

def build_models():
    def pipeline(estimator):
        preprocessing = ColumnTransformer([
            ('categories', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
            ('numbers', StandardScaler(), ['previous'])])
        return Pipeline([('preprocessing', preprocessing), ('model', estimator)])
    return {
        'Logistic regression': pipeline(LogisticRegression(max_iter=2000, random_state=SEED)),
        'Random forest': pipeline(RandomForestClassifier(n_estimators=300, max_depth=8,
             min_samples_leaf=30, random_state=SEED, n_jobs=1))}

def evaluate(y, scores):
    return {'average_precision': float(average_precision_score(y, scores)),
            'roc_auc': float(roc_auc_score(y, scores)),
            'brier_score': float(brier_score_loss(y, scores))}

def top_fraction(y, scores, fraction=.2):
    """Deterministic row-order tie break. Descriptive ranking, not causal uplift."""
    y = np.asarray(y)
    k = max(1, int(np.ceil(len(y)*fraction)))
    selected = np.argsort(-np.asarray(scores), kind='stable')[:k]
    precision = float(y[selected].mean())
    return {'fraction': fraction, 'selected_records': k, 'subscribers_selected': int(y[selected].sum()),
            'precision': precision, 'recall': float(y[selected].sum()/y.sum()),
            'lift_vs_random': float(precision/y.mean()),
            'boundary_score': float(np.asarray(scores)[selected[-1]])}

def main():
    reports = ROOT / 'reports'
    figures = reports / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    df, checksum = load_data()
    assert df.shape == (41188, 21), 'Unexpected dataset; check source before proceeding.'
    assert set(df.y.unique()) == {'yes', 'no'}
    assert df[FEATURES].isna().sum().sum() == 0
    assert (df.previous >= 0).all()
    assert 'duration' not in FEATURES and 'y' not in FEATURES
    split = split_data(df)
    train, val = split['train'], split['validation']
    xt, xv = train[FEATURES], val[FEATURES]
    yt, yv = train.y.eq('yes').astype(int), val.y.eq('yes').astype(int)
    models = build_models()
    scores = {'Training-rate baseline': np.full(len(val), yt.mean())}
    for name, model in models.items():
        model.fit(xt, yt)
        scores[name] = model.predict_proba(xv)[:, 1]
    validation = pd.DataFrame({name: evaluate(yv, p) for name,p in scores.items()}).T
    validation.to_csv(reports / 'validation_metrics.csv', index_label='model')
    pd.DataFrame({'row_index': val.index, 'actual_subscription': yv.to_numpy(), **scores}).to_csv(reports/'validation_predictions.csv', index=False)
    # Freeze selection BEFORE using the holdout labels or scores. AP is primary.
    # Fixed model definitions, no hyperparameter search, logistic wins exact ties.
    selected_name = max(models, key=lambda name: validation.loc[name, 'average_precision'])
    decision = {'selected_model': selected_name, 'selection_metric': 'validation average precision',
                'candidate_order_tiebreak': list(models), 'refit_on_validation': False,
                'test_used_for_selection': False, 'contact_fraction': .2,
                'contact_fraction_rationale': 'Illustrative fixed capacity, not optimized for profit.',
                'features': FEATURES, 'random_seed': SEED,
                'random_forest_parameters': {'n_estimators':300, 'max_depth':8, 'min_samples_leaf':30}}
    (reports/'frozen_model_decision.json').write_text(json.dumps(decision, indent=2)+'\n')
    # Final evaluation: selected training-fitted model only, plus fixed baseline.
    test = split['test']
    ytest = test.y.eq('yes').astype(int)
    test_scores = models[selected_name].predict_proba(test[FEATURES])[:,1]
    test_metrics = {selected_name: evaluate(ytest,test_scores),
                    'Training-rate baseline': evaluate(ytest,np.full(len(test),yt.mean()))}
    pd.DataFrame(test_metrics).T.to_csv(reports/'test_metrics.csv', index_label='model')
    ranking = {'validation': top_fraction(yv,scores[selected_name]), 'test': top_fraction(ytest,test_scores)}
    (reports/'ranking_metrics.json').write_text(json.dumps(ranking,indent=2)+'\n')
    summary = pd.DataFrame([{'split':k,'records':len(v),'subscribers':int(v.y.eq('yes').sum()),
              'subscription_rate':float(v.y.eq('yes').mean()),'first_row_zero_based':int(v.index.min()),
              'last_row_zero_based':int(v.index.max())} for k,v in split.items()])
    summary.to_csv(reports/'split_summary.csv',index=False)
    # Exact duplicate observations kept: no customer ID to resolve actual repeat contacts.
    hashes = {k:set(pd.util.hash_pandas_object(v,index=False)) for k,v in split.items()}
    quality = {'shape':list(df.shape),'null_counts':df.isna().sum().to_dict(),
       'unknown_counts':df.select_dtypes(include=['object','string']).eq('unknown').sum().to_dict(),
       'exact_duplicate_rows':int(df.duplicated().sum()),
       'cross_split_exact_row_hash_overlap': {a+'__'+b:len(hashes[a]&hashes[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]},
       'duplicate_policy':'Retained to preserve original ordered split; customer identities unavailable.',
       'dataset_sha256': checksum}
    (reports/'data_quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    pd.DataFrame({'row_index':test.index,'actual_subscription':ytest.to_numpy(),'score':test_scores}).to_csv(reports/'holdout_predictions.csv',index=False)
    # Importance is validation-only, after selection; never a causal explanation.
    importance = permutation_importance(models[selected_name],xv,yv,scoring='average_precision',n_repeats=5,random_state=SEED,n_jobs=1)
    imp = pd.DataFrame({'feature':FEATURES,'mean_ap_decrease':importance.importances_mean,
                         'std_ap_decrease':importance.importances_std}).sort_values('mean_ap_decrease',ascending=False)
    imp.to_csv(reports/'validation_permutation_importance.csv',index=False)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'font.size':10})
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar(summary.split, summary.subscription_rate*100,color=['#36567A','#408C93','#D8A24B'])
    ax.set(ylabel='Subscribed (%)',title='Outcome rate changes across chronological splits')
    for i,r in summary.iterrows(): ax.text(i,r.subscription_rate*100+.4,f'{r.subscription_rate:.1%}',ha='center')
    fig.tight_layout();fig.savefig(figures/'01_outcome_drift.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for name,p in scores.items():
        if name == 'Training-rate baseline':
            ax.axhline(yv.mean(), ls='--', color='gray', label=f'Random-ranking prevalence: {yv.mean():.3f}')
        else:
            precision,recall,_=precision_recall_curve(yv,p);ax.plot(recall,precision,label=f'{name}: AP {validation.loc[name,"average_precision"]:.3f}')
    ax.set(xlabel='Recall',ylabel='Precision',title='Validation comparison (selection uses AP)',ylim=(0,1));ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figures/'02_validation_precision_recall.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.barh(imp.feature[::-1],imp.mean_ap_decrease[::-1],xerr=imp.std_ap_decrease[::-1],color='#408C93')
    ax.axvline(0,color='gray',lw=.8);ax.set(xlabel='Decrease in validation average precision',title='Validation permutation importance (not causal)')
    fig.tight_layout();fig.savefig(figures/'03_validation_importance.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));p,r,_=precision_recall_curve(ytest,test_scores)
    ax.plot(r,p,label=selected_name);ax.axhline(ytest.mean(),ls='--',color='gray',label='Random-ranking prevalence')
    ax.set(xlabel='Recall',ylabel='Precision',title='Frozen model: final holdout ranking',ylim=(0,1));ax.legend()
    fig.tight_layout();fig.savefig(figures/'04_holdout_precision_recall.png');plt.close(fig)
    versions = {'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,
                'scikit-learn':sklearn.__version__,'matplotlib':matplotlib.__version__}
    (reports/'run_environment.json').write_text(json.dumps(versions,indent=2)+'\n')
    print('VALIDATION\n',validation.round(4));print('\nSELECTED:',selected_name)
    print('\nFINAL HOLDOUT\n',pd.DataFrame(test_metrics).T.round(4));print('\nSPLIT RATES\n',summary)
    print('\nRANKING\n',json.dumps(ranking,indent=2));print('\nQUALITY\n',json.dumps(quality,indent=2))
    return {'validation': validation,'test':pd.DataFrame(test_metrics).T,'selected_model':selected_name,'ranking':ranking,'splits':summary}

if __name__ == '__main__':
    main()
