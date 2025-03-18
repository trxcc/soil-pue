import os 
import wandb
import numpy as np 
import pandas as pd 
import argparse 
import datetime
import shap 
import torch 
import matplotlib.pyplot as plt 
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from solver import get_model

def set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determinstic = True

all_columns = None 

def preprocess_data(data: pd.DataFrame,
                    one_hot_indices: List[str] = [],
                    remove_indices: List[str] = []) -> pd.DataFrame:
    one_hot_encoded_data = None
    for one_hot_index in one_hot_indices:
        data[one_hot_index] = data[one_hot_index].str.lower()
        now_encoded_data = pd.get_dummies(data[one_hot_index], prefix=one_hot_index).astype(float)
        if one_hot_encoded_data is None:
            one_hot_encoded_data = now_encoded_data
        else:
            one_hot_encoded_data = pd.concat([one_hot_encoded_data, now_encoded_data], axis=1)
    data.drop(one_hot_indices + remove_indices, axis=1, inplace=True)
    if one_hot_encoded_data is not None:
        data = pd.concat([data, one_hot_encoded_data], axis=1)
    global all_columns
    if all_columns is None:
        all_columns = data.columns
    else:
        for column in all_columns:
            if column not in data.columns:
                data[column] = 0
        data = data[all_columns]
    return data

parser = argparse.ArgumentParser()
parser.add_argument('--which-obj', type=str, choices=['MurA', 'GlcN', 'F_GN', 'A_GN', 'k', 'PUE'], default='MurA')
parser.add_argument('--model-name', type=str, choices=['RandomForest', 'AutoSklearn', 'XGBoost',
                                                       'LightGBM', 'CNN', 'CatBoost', 'DeepForest',
                                                       'FTTransformer', 'MLP', 'ResNet'], default='RandomForest')
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--optimize-hyperparams', action='store_true')
parser.add_argument('--optimize-method', type=str, choices=['BayesOpt', 'GridSearch'], default='BayesOpt')
args = parser.parse_args()

set_seed(args.seed)

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_path, 'results', args.which_obj + "-all_data", f'{args.seed}-' + args.model_name \
                           + ("GridSearch" if args.optimize_method == 'GridSearch' else ""))
model_dir = os.path.join(base_path, 'models', args.which_obj  + "all_data-only_train" + f'{args.seed}' + ("GridSearch" if args.optimize_method == 'GridSearch' else ""))
os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
args.__dict__.update({"results_dir": results_dir, "model_dir": model_dir})

data_all = pd.read_excel(os.path.join(base_path, 'train1.xlsx')).dropna().copy() 
data_all = preprocess_data(data_all, remove_indices=['ID'], one_hot_indices=['Climate_Zone', 'Crop_type', 'fertilizer_type', 'fertilizer_placement', 'crop_residue', 'tillage', 'Cropping_Systems'])

X = data_all.drop(args.which_obj.upper(), axis=1).to_numpy()
y = data_all[args.which_obj.upper()].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
name = f'{args.model_name}-{args.seed}'
ts_name = f'-{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}'
wandb.init(project="soil-pue",
           group=f'{args.which_obj}-test', 
            name=f'{args.seed}-' + name + ts_name + ("GridSearch" if args.optimize_method == 'GridSearch' else ""), 
            config=args.__dict__, 
            job_type='train'
            )
model = get_model(args.model_name, task_name=args.which_obj , seed=args.seed, model_dir=model_dir, results_dir=results_dir)
model.fit(X_train=X_train, y_train=y_train, optimize_hyperparams=args.optimize_hyperparams,
          optimize_method=args.optimize_method)

R2score = model.evaluate(X_test, y_test)
y_test_pred = model.predict(X_test)
R2score_all = model.evaluate(X, y)
y_pred = model.predict(X)

MSE_score_test = mean_squared_error(y_test, y_test_pred)
MSE_score_all = mean_squared_error(y, y_pred)

results_df = pd.DataFrame()
results_df[f'{args.which_obj}_true'] = pd.Series(y.ravel())
results_df[f'{args.which_obj}_pred'] = pd.Series(y_pred.ravel())
results_df.to_excel(os.path.join(results_dir, f'{args.which_obj}-{args.model_name}-{args.seed}.xlsx'))

test_results_df = pd.DataFrame()
test_results_df[f'{args.which_obj}_true'] = pd.Series(y_test.ravel())
test_results_df[f'{args.which_obj}_pred'] = pd.Series(y_test_pred.ravel())
test_results_df.to_excel(os.path.join(results_dir, f'{args.which_obj}-{args.model_name}-test-{args.seed}.xlsx'))
print(R2score)
print(R2score_all)

csv_path = os.path.join(results_dir, '..', ("GridSearch" if args.optimize_method == 'GridSearch' else "") + 'final-results.csv')
result = {
    "model_name": args.model_name,
    f"{args.seed}": R2score
}

if not os.path.exists(csv_path):
    new_df = pd.DataFrame([result])
    new_df.to_csv(csv_path, index=False)
else:
    existing_df = pd.read_csv(csv_path, header=0, index_col=0)
    updated_df = existing_df.copy()
    updated_df.loc[args.model_name, f"{args.seed}"] = R2score
    updated_df.columns = updated_df.columns.astype(int)
    updated_df = updated_df.sort_index(axis=1)
    updated_df.to_csv(csv_path, index=True, mode='w')

csv_path = os.path.join(results_dir, '..', ("GridSearch" if args.optimize_method == 'GridSearch' else "") + 'all-final-results.csv')
result = {
    "model_name": args.model_name,
    f"{args.seed}": R2score_all
}

if not os.path.exists(csv_path):
    new_df = pd.DataFrame([result])
    new_df.to_csv(csv_path, index=False)
else:
    existing_df = pd.read_csv(csv_path, header=0, index_col=0)
    updated_df = existing_df.copy()
    updated_df.loc[args.model_name, f"{args.seed}"] = R2score_all
    updated_df.columns = updated_df.columns.astype(int)
    updated_df = updated_df.sort_index(axis=1)
    updated_df.to_csv(csv_path, index=True, mode='w')

csv_path = os.path.join(results_dir, '..', ("GridSearch" if args.optimize_method == 'GridSearch' else "") + 'final-mse.csv')
result = {
    "model_name": args.model_name,
    f"{args.seed}": MSE_score_test
}

if not os.path.exists(csv_path):
    new_df = pd.DataFrame([result])
    new_df.to_csv(csv_path, index=False)
else:
    existing_df = pd.read_csv(csv_path, header=0, index_col=0)
    updated_df = existing_df.copy()
    updated_df.loc[args.model_name, f"{args.seed}"] = MSE_score_test
    updated_df.columns = updated_df.columns.astype(int)
    updated_df = updated_df.sort_index(axis=1)
    updated_df.to_csv(csv_path, index=True, mode='w')
    
csv_path = os.path.join(results_dir, '..', ("GridSearch" if args.optimize_method == 'GridSearch' else "") + 'all-final-mse.csv')
result = {
    "model_name": args.model_name,
    f"{args.seed}": MSE_score_all
}

if not os.path.exists(csv_path):
    new_df = pd.DataFrame([result])
    new_df.to_csv(csv_path, index=False)
else:
    existing_df = pd.read_csv(csv_path, header=0, index_col=0)
    updated_df = existing_df.copy()
    updated_df.loc[args.model_name, f"{args.seed}"] = MSE_score_all
    updated_df.columns = updated_df.columns.astype(int)
    updated_df = updated_df.sort_index(axis=1)
    updated_df.to_csv(csv_path, index=True, mode='w')


model.save(model_dir)


def explain_model(regression_model):

    def f(X):
        nonlocal regression_model
        return regression_model.predict(X).flatten()
    
    global results_dir, X

    explainer = shap.KernelExplainer(f, X[:50, :])
    shap_values = explainer.shap_values(X[50, :], nsamples=500)
    np.save(arr=shap_values, file=os.path.join(results_dir, 'shap_values.npy'))
    shap.initjs()
    shap_plot = shap.force_plot(explainer.expected_value, shap_values, X[50, :])
    shap.save_html(os.path.join(results_dir, 'shap_single_predictions.html'), shap_plot)

    try:
        shap_values50 = explainer.shap_values(X[:, :], nsamples=max(500, X.shape[0]))
        np.save(arr=shap_values50, file=os.path.join(results_dir, 'shap_values50.npy'))
        shap_plot_all = shap.force_plot(explainer.expected_value, shap_values50, X[:, :])
        shap.save_html(os.path.join(results_dir, 'shap_many_predictions.html'), shap_plot_all)
    except:
        try:
            shap_values50 = explainer.shap_values(X[:, :], nsamples=max(500, X.shape[0]))
            np.save(arr=shap_values50, file=os.path.join(results_dir, 'shap_values50.npy'))
            shap_plot_all = shap.force_plot(explainer.expected_value, shap_values50, X[:, :])
            shap.save_html(os.path.join(results_dir, 'shap_many_predictions.html'), shap_plot_all)
        except:
            pass 

explain_model(model)


model_dir = os.path.join(base_path, 'models', args.which_obj  + "-all_data" + f'{args.seed}' + ("GridSearch" if args.optimize_method == 'GridSearch' else ""))
os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
args.__dict__.update({"results_dir": results_dir, "model_dir": model_dir})

model = get_model(args.model_name, task_name=args.which_obj , seed=args.seed, model_dir=model_dir, results_dir=results_dir)
model.fit(X_train=X, y_train=y, optimize_hyperparams=False)
model.save(model_dir)

def predict_test(data_name):

    predict_file = os.path.join(base_path, f'{data_name}.xlsx')
    pred_data_TOP = pd.read_excel(predict_file).dropna().copy()
    # pred_data_SUB = pd.read_excel(predict_file, sheet_name='SUB').dropna().copy()

    TOP_data_copy = pred_data_TOP.copy()
    # SUB_data_copy = pred_data_SUB.copy()

    # TOP_data_copy.drop(['ID'], axis=1, inplace=True)
    # SUB_data_copy.drop(['ID'], axis=1, inplace=True)

    pred_data_TOP = preprocess_data(pred_data_TOP, remove_indices=['ID'], one_hot_indices=['Climate_Zone', 'Crop_type', 'fertilizer_type', 'fertilizer_placement', 'crop_residue', 'tillage', 'Cropping_Systems']).drop(args.which_obj.upper(), axis=1).to_numpy()
    # pred_data_SUB = preprocess_data(pred_data_SUB).drop(args.which_obj, axis=1).to_numpy()

    # print(pred_data_TOP.shape, len(all_columns), all_columns)

    pred_TOP: np.ndarray = model.predict(pred_data_TOP)
    # pred_TOP = 1 - pred_TOP
    # pred_SUB: np.ndarray = model.predict(pred_data_SUB)

    TOP_data_copy[args.which_obj] = pd.Series(pred_TOP.ravel(), index=TOP_data_copy.index)
    # SUB_data_copy[args.which_obj] = pd.Series(pred_SUB.ravel(), index=SUB_data_copy.index)

    TOP_data_copy.dropna(axis=1, inplace=True)
    # SUB_data_copy.dropna(axis=1, inplace=True)

    TOP_data_copy.to_excel(os.path.join(results_dir, f'{args.which_obj}-all_data-{data_name}-{args.model_name}.xlsx'))
    # SUB_data_copy.to_excel(os.path.join(results_dir, f'SUB-{args.which_obj}-{args.model_name}.xlsx'))


predict_test('maize-total')
predict_test('rice-total')
predict_test('wheat-total')
