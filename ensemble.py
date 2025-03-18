import torch 
import pickle 
import os 
import shap 
import numpy as np 
import pandas as pd 
from typing import List
from solver.catboost import CatBoostModel
from solver.xgboost import XGBoostModel
from solver.random_forest import RandomForestModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import argparse 

parser = argparse.ArgumentParser() 
parser.add_argument('--seed', type=int, default=2024)
args = parser.parse_args() 

seed = args.seed

def set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determinstic = True
    
set_seed(seed)

model1 = CatBoostModel(
    task_name="PUE",
    seed=seed,
)

with open(os.path.join(
    "models", f"PUEall_data-only_train{seed}", "CatBoost.pkl"
), "rb+") as f:
    model1.model = pickle.load(f)

print(model1.model)

model2 = XGBoostModel(
    task_name="PUE",
    seed=seed,
)

with open(os.path.join(
    "models", f"PUEall_data-only_train{seed}", "XGBoost.pkl"
), "rb+") as f:
    model2.model = pickle.load(f)

print(model2.model)

model4 = RandomForestModel(
    task_name="PUE",
    seed=seed,
)

with open(os.path.join(
    "models", f"PUEall_data-only_train{seed}", "RandomForestModel.pkl"
), "rb+") as f:
    model4.model = pickle.load(f)

print(model4.model)

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

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_path, 'results', "PUE-ensemble-all", f'{seed}-' + "ensemble-CB+XG+RF")
os.makedirs(results_dir, exist_ok=True)

data_all = pd.read_excel(os.path.join(base_path, 'train1.xlsx')).dropna().copy() 
data_all = preprocess_data(data_all, remove_indices=['ID', 'PUE'], one_hot_indices=['Climate_Zone', 'Crop_type', 'fertilizer_type', 'fertilizer_placement', 'crop_residue', 'tillage', 'Cropping_Systems'])

class WrapedModel:
    def __init__(self, model_list):
        self.model_list = model_list
    
    def predict(self, X):
        res = np.zeros((X.shape[0], 1))
        for model in self.model_list:
            res += ((1 / len(self.model_list)) * model.predict(X).reshape(-1, 1))
        return res 

    
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
        from solver.auto_sklearn import AutoSklearnModel
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

set_seed(seed)

wraped_model = WrapedModel([model1, model2, model4])

X = data_all.drop(args.which_obj.upper(), axis=1).to_numpy()
y = data_all[args.which_obj.upper()].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

y_test_pred = wraped_model.predict(X_test)
R2score = r2_score(y_test, y_test_pred)

csv_path = os.path.join(base_path, 'ensemble-r2.csv')
result = {
    "model_name": "Ensemble (CB + XG + RF)",
    f"{args.seed}": R2score
}

if not os.path.exists(csv_path):
    new_df = pd.DataFrame([result])
    new_df.to_csv(csv_path, index=False)
else:
    existing_df = pd.read_csv(csv_path, header=0, index_col=0)
    updated_df = existing_df.copy()
    updated_df.loc["Ensemble (CB + XG + RF)", f"{args.seed}"] = R2score
    updated_df.columns = updated_df.columns.astype(int)
    updated_df = updated_df.sort_index(axis=1)
    updated_df.to_csv(csv_path, index=True, mode='w')
    
y_pred = wraped_model.predict(X)
R2score = r2_score(y, y_pred)

csv_path = os.path.join(base_path, 'all-ensemble-new-r2.csv')
result = {
    "model_name": "Ensemble (CB + XG + RF)",
    f"{args.seed}": R2score
}

if not os.path.exists(csv_path):
    new_df = pd.DataFrame([result])
    new_df.to_csv(csv_path, index=False)
else:
    existing_df = pd.read_csv(csv_path, header=0, index_col=0)
    updated_df = existing_df.copy()
    updated_df.loc["Ensemble (CB + XG + RF)", f"{args.seed}"] = R2score
    updated_df.columns = updated_df.columns.astype(int)
    updated_df = updated_df.sort_index(axis=1)
    updated_df.to_csv(csv_path, index=True, mode='w')

y_test_pred = wraped_model.predict(X_test)
# R2score_all = model.evaluate(X, y)
y_pred = wraped_model.predict(X)

MSE_score_test = mean_squared_error(y_test, y_test_pred)
MSE_score_all = mean_squared_error(y, y_pred)

results_df = pd.DataFrame()
results_df[f'k_true'] = pd.Series(y.ravel())
results_df[f'k_pred'] = pd.Series(y_pred.ravel())
results_df.to_excel(os.path.join(results_dir, f'k-Ensemble (CB + XG + RF)-{args.seed}.xlsx'))

test_results_df = pd.DataFrame()
test_results_df[f'k_true'] = pd.Series(y_test.ravel())
test_results_df[f'k_pred'] = pd.Series(y_test_pred.ravel())
test_results_df.to_excel(os.path.join(results_dir, f'k-Ensemble (CB + XG + RF)-test-{args.seed}.xlsx'))


wraped_model = WrapedModel([model1, model2, model4])

explain_model(wraped_model)

with open(os.path.join(
    "models", f"PUE-all_data{seed}", "CatBoost.pkl"
), "rb+") as f:
    model1.model = pickle.load(f)

with open(os.path.join(
    "models", f"PUE-all_data{seed}", "XGBoost.pkl"
), "rb+") as f:
    model2.model = pickle.load(f)

with open(os.path.join(
    "models", f"PUE-all_data{seed}", "RandomForestModel.pkl"
), "rb+") as f:
    model4.model = pickle.load(f)

wraped_model = WrapedModel([model1, model2, model4])

fertilizer_type = ['mineral fertilizer', 'organic fertilizer', 'microbial fertilizer', 'enhanced efficiency fertilisers']
fertilizer_placement = ['SBC', 'DPM', 'Band']
crop_residue = ['yes', 'no']
tillage = ['traditional', 'reduced'] 
Cropping_Systems = ['monocropping', 'intercropping', 'crop ratation']

combinations_gen = [(t, f, r, c, p) for t in fertilizer_type for f in fertilizer_placement for r in crop_residue for c in tillage for p in Cropping_Systems]

predict_file = os.path.join(base_path, 'rice-total.xlsx')
pred_data = pd.read_excel(predict_file).dropna().copy()
all_predictions = pd.DataFrame()

for i, combination in enumerate(combinations_gen):
    new_columns = {
        'fertilizer_type' : [combination[0]] * len(pred_data),
        'fertilizer_placement' : [combination[1]] * len(pred_data),
        'crop_residue' : [combination[2]] * len(pred_data),
        'tillage' : [combination[3]] * len(pred_data),
        'Cropping_Systems' : [combination[4]] * len(pred_data)
    }
    pred_data_new = pred_data.assign(**new_columns)
    pred_data_processed = preprocess_data(pred_data_new, remove_indices=['ID'], one_hot_indices=['Climate_Zone', 'Crop_type', 'fertilizer_type', 'fertilizer_placement', 'crop_residue', 'tillage', 'Cropping_Systems'])
    pred_data_processed = pred_data_processed.to_numpy()
    
    pred_TOP = wraped_model.predict(pred_data_processed)

    all_predictions[f'{i}'] = pred_TOP.flatten()
    
    
combinations_list = list(combinations_gen)
max_indices = all_predictions.idxmax(axis=1)
min_indices = all_predictions.idxmin(axis=1)
max_indices = max_indices.astype(int) 
min_indices = min_indices.astype(int) 

max_combinations = [combinations_list[idx] for idx in max_indices]
min_combinations = [combinations_list[idx] for idx in min_indices]
print(f"Length of combinations_list: {len(combinations_list)}")
print(f"Max index in max_indices: {max(max_indices)}")
pred_data['Max_Prediction_Combination'] = max_combinations
pred_data['Max_Value'] = all_predictions.max(axis=1)
pred_data['Min_Prediction_Combination'] = min_combinations
pred_data['Min_Value'] = all_predictions.min(axis=1)

pred_data.to_excel(os.path.join(results_dir, f'MAX-Predict-rice-PUE-Ensemble (CB + XG + RF).xlsx'))

def predict_file(data_name):
    predict_file = os.path.join(base_path, f'{data_name}.xlsx')
    pred_data_TOP = pd.read_excel(predict_file).dropna().copy()

    TOP_data_copy = pred_data_TOP.copy()

    pred_data_TOP = preprocess_data(pred_data_TOP, remove_indices=['ID'], one_hot_indices=['Climate_Zone', 'Crop_type', 'fertilizer_type', 'fertilizer_placement', 'crop_residue', 'tillage', 'Cropping_Systems']).drop('PUE', axis=1).to_numpy()
    pred_TOP = wraped_model.predict(pred_data_TOP)

    TOP_data_copy['PUE'] = pd.Series(pred_TOP.ravel(), index=TOP_data_copy.index)

    TOP_data_copy.dropna(axis=1, inplace=True)

    TOP_data_copy.to_excel(os.path.join(results_dir, f'PUE-all_data-{data_name}-Ensemble (CB + XG + RF).xlsx'))

predict_file('wheat-total')
predict_file('maize-total')
predict_file('rice-total')