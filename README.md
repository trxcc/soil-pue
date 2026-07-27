# Soil-PUE

Official implementation of the 2026 *Nature Food* paper **“Global patterns and feasible improvement potential of phosphorus use efficiency in cereal croplands.”**

This repository contains the core machine-learning code used to model phosphorus use efficiency (PUE), compare tabular regression methods, construct the ensemble model, and evaluate cropland-management scenarios for maize, rice, and wheat.

## Repository structure

- `main.py`: trains and evaluates an individual model.
- `ensemble.py`: evaluates the CatBoost, XGBoost, and Random Forest ensemble and generates predictions.
- `ensemble_comb_pred.py`: evaluates management-practice combinations with the ensemble model.
- `read_shap.py`: processes SHAP outputs for model interpretation.
- `solver/`: implementations of the supported regression models.
- `run.sh`, `run_ensemble.sh`, and `run_comb.sh`: batch scripts used in the original computing environment.

The model pool includes Random Forest, XGBoost, LightGBM, CatBoost, Deep Forest, Auto-sklearn, MLP, CNN, ResNet, and FT-Transformer.

## Environment

The experiments were developed with Python 3.9. We recommend creating an isolated Conda environment:

```bash
conda create -n soil-pue python=3.9
conda activate soil-pue
```

### Dependencies

The code depends on common scientific Python and machine-learning packages, including PyTorch, pandas, NumPy, scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, Optuna, Weights & Biases, and **EconML 0.16.0**, together with the model-specific packages imported in `solver/`.

Install the required EconML version with:

```bash
pip install econml==0.16.0
```

Weights & Biases is used for experiment tracking. Configure your account by following the [Weights & Biases quickstart](https://docs.wandb.ai/quickstart), or set `WANDB_MODE=offline` when running without online logging.

## Running the code

To train and evaluate one model, run:

```bash
python main.py \
  --which-obj PUE \
  --model-name RandomForest \
  --seed 42
```

Available model names are:

```text
RandomForest, AutoSklearn, XGBoost, LightGBM, CNN,
CatBoost, DeepForest, FTTransformer, MLP, ResNet
```

To enable hyperparameter optimization, add `--optimize-hyperparams`. The optimization method can be selected with `--optimize-method BayesOpt` or `--optimize-method GridSearch`.

After training Random Forest, CatBoost, and XGBoost for the same seed, run the ensemble workflow with:

```bash
python ensemble.py --seed 42
```

The experiments use 13 consecutive random seeds, from `42` through `54` (inclusive). The batch scripts run this complete seed set. Review their GPU and concurrency settings before running them on a different system.

## Data and model availability

The training data, global prediction inputs, and trained model checkpoints referenced by the scripts are not included in this public repository. For detailed source code, preprocessing and modeling information, trained models, data access, or other research materials, please contact the corresponding author.

## Citation

If you use this repository in your research, please cite the paper:

```bibtex
@article{sun2026global,
  title   = {Global patterns and feasible improvement potential of phosphorus use efficiency in cereal croplands},
  author  = {Sun, Yishen and Hu, Han and Tan, Rong-Xi and Helfenstein, Julian and McDowell, Richard W. and Gu, Baojing and Ni, Haowei and Huang, Weigen and Ding, Jixian and Xue, Ke and Qian, Chao and Crowther, Thomas W. and Zhou, Jizhong and Zhou, Zhi-Hua and Zhang, Jiabao and Liang, Yuting},
  journal = {Nature Food},
  year    = {2026}
}
```

The volume, issue, page range, and DOI should be added when the final bibliographic record becomes available.

The same entry is available in [`CITATION.bib`](CITATION.bib).

## Contact

- For questions about using the public code, please open a GitHub issue.
- For detailed code, trained models, data, or scientific inquiries, please contact the corresponding author: **Yuting Liang** ([ytliang@issas.ac.cn](mailto:ytliang@issas.ac.cn)).
