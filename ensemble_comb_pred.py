import torch 
import pickle 
import os 
import shap 
import numpy as np 
import pandas as pd 
from typing import List, Dict
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
                    remove_indices: List[str] = [],
                    all_possible_values: Dict[str, List[str]] = {}) -> pd.DataFrame:
    one_hot_encoded_data = None
    for one_hot_index in one_hot_indices:
        data[one_hot_index] = data[one_hot_index].str.lower()
        
        if one_hot_index in all_possible_values:
            all_values = [v.lower() for v in all_possible_values[one_hot_index]]
            now_encoded_data = pd.get_dummies(data[one_hot_index], prefix=one_hot_index, columns=all_values)
            
            for value in all_values:
                col_name = f"{one_hot_index}_{value}"
                if col_name not in now_encoded_data.columns:
                    now_encoded_data[col_name] = 0
        else:
            now_encoded_data = pd.get_dummies(data[one_hot_index], prefix=one_hot_index)
        
        now_encoded_data = now_encoded_data.astype(float)
        
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
list0 = data_all.columns.to_list()


class WrapedModel:
    def __init__(self, model_list):
        self.model_list = model_list
    
    def predict(self, X):
        res = np.zeros((X.shape[0], 1))
        for model in self.model_list:
            res += ((1 / len(self.model_list)) * model.predict(X).reshape(-1, 1))
        return res 


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



def combine_pred(pred_file):

    fertilizer_type = ['mineral fertilizer', 'organic fertilizer', 'microbial fertilizer', 'enhanced efficiency fertilisers']
    fertilizer_placement = ['SBC', 'DPM', 'Band']
    crop_residue = ['yes', 'no']
    tillage = ['traditional', 'reduced'] 
    Cropping_Systems = ['monocropping', 'intercropping', 'crop ratation']

    all_possible_values = {
        'fertilizer_type': fertilizer_type,
        'fertilizer_placement': fertilizer_placement,
        'crop_residue': crop_residue,
        'tillage': tillage,
        'Cropping_Systems': Cropping_Systems
    }

    combinations_gen = [(t, f, r, c, p) for t in fertilizer_type for f in fertilizer_placement for r in crop_residue for c in tillage for p in Cropping_Systems]

    predict_file = os.path.join(base_path, f'{pred_file}-total.xlsx')
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
        pred_data_processed = preprocess_data(pred_data_new, remove_indices=['ID'], one_hot_indices=['Climate_Zone', 'Crop_type', 'fertilizer_type', 'fertilizer_placement', 'crop_residue', 'tillage', 'Cropping_Systems'], all_possible_values=all_possible_values)
        
        pred_TOP = wraped_model.predict(pred_data_processed)


        
        all_predictions[f'{i}'] = pred_TOP.flatten()
        
    combinations_list = list(combinations_gen)
    def find_prediction(row):
        combination = (row['fertilizer_type'], row['fertilizer_placement'], row['crop_residue'], row['tillage'], row['Cropping_Systems'])
        if combination in combinations_list:
            idx = combinations_list.index(combination)
            try:
                return all_predictions[f'{idx}'].iloc[row.name]
            except:
                return None
        return None

    pred_data['PUE'] = pred_data.apply(find_prediction, axis=1)

    max_indices = all_predictions.idxmax(axis=1)
    min_indices = all_predictions.idxmin(axis=1)
    max_indices = max_indices.astype(int) 
    min_indices = min_indices.astype(int) 

    max_combinations = [combinations_list[idx] for idx in max_indices]
    min_combinations = [combinations_list[idx] for idx in min_indices]

    pred_data['Max_Prediction_Combination'] = max_combinations
    pred_data['Max_Value'] = all_predictions.max(axis=1)
    pred_data['Min_Prediction_Combination'] = min_combinations
    pred_data['Min_Value'] = all_predictions.min(axis=1)

    pred_data.to_excel(os.path.join(results_dir, f'MAX-Predict-{pred_file}-PUE-Ensemble (CB + XG + RF).xlsx'))

combine_pred('rice')
combine_pred('maize')
combine_pred('wheat')