# sde-finance

Spring 2025 10-716 course project: Stochastic Dynamical Models for Financial Market
Prediction and Analysis


## Getting started

The project is organized as follows:
- `finsde` contains the core code defining all the models and training utilities used for the project
- `notebooks` contains notebooks used for exploratory analysis of trained models, particularly `volatility.ipynb`
- `figures` contains the resulting figures from those exploratory analyses that made it into the paper
- `scripts` contains scripts for processing the data and training models

The data used for model training can be downloaded from [here](https://github.com/Zdong104/FNSPID_Financial_News_Dataset/tree/main/dataset_test/CNN-for-Time-Series-Prediction/data). After downloading, all files need to be renamed to be in all caps to work with existing config files.

You can find the project report at `final_report.pdf`.

## Running experiments

`finsde` relies on `hydra` for configuration, `Lightning` for training management, and WandB for logging. To train models, you simply need to run `scripts/train.py`, or if submitting jobs on SLURM, `scripts/train.sh`. Training options can be chosen using `hydra` CLI overrides. For example, to train an SRDM (also called discrete SDE in this codebase) on 3-step forecasting and 50 stocks, you can simply run

```
python train.py 'model=discrete_sde_posterior_full'
```