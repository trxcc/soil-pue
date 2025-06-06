# Optimizing global cropland management to address phosphorus sustainability

Official implementation of paper: *Optimizing global cropland management to address phosphorus sustainability*.

## Installation

For reproducing our results, you have to successfully install ``conda`` first according to [official documentation of Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).


### Main Algorithms

We implement nine algorithms in this code repository. In this subsection, we will show how to install environment for eight major methods, except for ``Deep-Forest``, the installation of which will be demonstrated in the next subsection.

First, create a ``conda`` environment via 
```shell
conda create -n soil python=3.9
conda activate soil
```

Then, install required packages that we have listed in ``soil_requirements.txt`` via 
```shell
pip install -r soil_requirements.txt
```

### Wandb (Weight & Bias)

[Wandb](https://wandb.ai/) is a convenient package to track and visualize all the pieces of machine learning pipeline, and we use it to track our results.

Please refer to the [documentation of wandb](https://docs.wandb.ai/quickstart) to sign up for a free account and paste your API key to the command line accordingly.

## Reproducing Results
To reproduce the performance of baseline algorithms reported in our work, you may then run ``run.sh``, or run the following series of commands in a bash terminal. 

> Please ensure that the conda environment ``soil`` is activated in the bash session when you are running our algorithms.
<!-- ```shell
for seed in 2024; do
    for which_obj in "PUE"; do
        for model_name in "CatBost"; do
            CUDA_VISIBLE_DEVICES=0 python main.py \
            --model ${model_name} \
            --seed ${seed} 
        done
    done
done
``` -->

To utilize ensemble models, you may first train the model under ``RandomForest``, ``CatBoost``, and ``XGBoost``, then ensemble them using ``python ensemble.py --seed ${SEED}``.

### Notes for algorithms

- For ``seed``, you can set it to any positive integers, since it is task-agnostic and model-agnostic.
- For ``target``, there is one available option that represent different targets in our paper: ``PUE``.
- For ``model``, there are few available options that represent different methods: ``RandomForest``, ``CatBoost``, ``LightGBM``, ``XGBoost``, ``MLP``, ``CNN``, ``ResNet``, ``FTTransformer``, ``DeepForest``. 

## Contact

If you have any questions, feel free to contact us or raise an issue.
