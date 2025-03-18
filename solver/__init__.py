try:
    from .random_forest import RandomForestModel
    from .auto_sklearn import AutoSklearnModel
    from .ft_transformer import FTTransformerModel
    from .mlp import MLPModel
    from .resnet import ResNetModel
    from .xgboost import XGBoostModel
    from .catboost import CatBoostModel
    from .lightgbm import LightGBMModel
    from .cnn import CNNModel
except:
    from .deep_forest import DeepForestModel

# from .random_forest import RandomForestModel
# from .auto_sklearn import AutoSklearnModel
# from .ft_transformer import FTTransformerModel
# from .mlp import MLPModel
# from .resnet import ResNetModel
# from .xgboost import XGBoostModel
# from .catboost import CatBoostModel
# from .lightgbm import LightGBMModel
# from .cnn import CNNModel

def get_model(model_name, *args, **kwargs):
    model_name = model_name.lower()

    try:
        MODELS = {
            'randomforest': RandomForestModel,
            'autosklearn': AutoSklearnModel,
            'fttransformer': FTTransformerModel,
            'mlp': MLPModel,
            'resnet': ResNetModel,
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'cnn': CNNModel,
            'catboost': CatBoostModel,
        }
    except:
        MODELS = {
            'deepforest': DeepForestModel,
        }

    if model_name not in MODELS.keys():
        raise Exception(f"Model {model_name} not found.")
    
    return MODELS[model_name](*args, **kwargs)
