import shap 
import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from typing import List
from solver import get_model
import matplotlib

params = {
    'lines.linewidth': 1.5,
    'legend.fontsize': 17,
    'axes.labelsize': 17,
    'axes.titlesize': 17,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
}
matplotlib.rcParams.update(params)

plt.rc('font',family='Times New Roman')

env2methodseed = {
    'PUE': [('Ensemble (CB + XG + RF)', i) for i in range(1000, 9001, 1000)]
}

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
shap_path = os.path.join(base_path, 'shap_fig')
data_path = os.path.join(base_path, 'shap_data')
os.makedirs(shap_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)

shap.initjs()

occur_dict = {}

one_hot_ind = ['Climate_Zone', 'Crop_type', 'fertilizer_type', 'fertilizer_placement', 'crop_residue', 'tillage', 'Cropping_Systems']

for env, method_seed_pairs in env2methodseed.items():
    
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

    data_all = pd.read_excel(os.path.join(base_path, 'train1.xlsx')).dropna().copy() 
    data_all = preprocess_data(data_all, remove_indices=['ID'], one_hot_indices=one_hot_ind)

    X = data_all.drop('PUE', axis=1).to_numpy()
    y = data_all['PUE'].to_numpy()
        
    for method, seed in method_seed_pairs:
        print(method, seed)
        path = os.path.join(results_dir, env + '-ensemble-all', f'{seed}-ensemble-CB+XG+RF')
        if not os.path.exists(os.path.join(path, 'shap_values50.npy')):
            continue
        shap_values = np.load(os.path.join(path, 'shap_values50.npy'))
        print(shap_values.shape)
        shap_mean_values = np.mean(shap_values, axis=0)
        print(shap_mean_values.shape)
        
        print(shap_values)
        print(shap_mean_values)
        
        new_shap_values = {}
        one_hot_shap_values = {}
        
        features = data_all.drop(['PUE'], axis=1).columns.to_list()  
        print(features)
        print(one_hot_ind)
        
        # 遍历特征和对应的 SHAP 值
        for i, feature in enumerate(features):
            flag = False
            for tmp_str in one_hot_ind:
                # print(tmp_str)
                # if feature == "Climate_Zone_a":
                #     print(tmp_str, feature.lower().startswith(tmp_str))
                if feature.startswith(tmp_str):
                    print(feature, tmp_str)
                    if tmp_str not in one_hot_shap_values.keys():
                        one_hot_shap_values[tmp_str] = [shap_mean_values[i]]
                    else:
                        one_hot_shap_values[tmp_str].append(shap_mean_values[i])
                    flag = True
            if not flag:
                new_shap_values[feature] = shap_mean_values[i]
        # 计算 'Ecosystem' 特征的平均 SHAP 值并添加到字典
        # if ecosystem_shap_values:
        #     new_shap_values['Ecosystem'] = np.mean(ecosystem_shap_values)
       
        # assert 0, one_hot_shap_values
       
        for k, l in one_hot_shap_values.items():
            new_shap_values[k] = np.mean(l)
       
        # 获取排序后的特征和 SHAP 值
        sorted_features = sorted(new_shap_values, key=lambda x: abs(new_shap_values[x]), reverse=True)
        sorted_shap_values = np.array([new_shap_values[feature] for feature in sorted_features])
        
        df = pd.DataFrame(list(sorted_shap_values), index=sorted_features)
        df.to_excel(os.path.join(data_path, f'{method}-{seed}-{env}-ShapValues.xlsx'))
        
        print(sorted_features)
        
        # occur_dict[f"{method}-{seed}-{env}"] = sorted_features.index('WTD') + sorted_features.index('C-N')
        
        # assert 0, sorted_features.index('WTD')
        
        print(sorted_shap_values)
        
        # 创建一个图和轴
        fig, ax = plt.subplots(figsize=(20, 12))

        y_pos = np.arange(len(sorted_features))
        ax.barh(y_pos, sorted_shap_values, color=np.where(sorted_shap_values >= 0, 'blue', 'red'))

        # 添加特征名作为 y 轴标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)

        # 添加 x 轴和标题
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature importances of {method} on {env}')
        
        ax.invert_yaxis()

        # 显示图形
        plt.savefig(os.path.join(shap_path, f'{method}-{seed}-{env}.png'))
        plt.savefig(os.path.join(shap_path, f'{method}-{seed}-{env}.pdf'))
        plt.close()

sorted_items = sorted(occur_dict.items(), key=lambda item: item[1])

# 输出排序后的键值对
print("Sorted key-value pairs by ascending values:")
for key, value in sorted_items:
    print(f"{key}: {value + 2}")
        
        