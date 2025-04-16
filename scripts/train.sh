#!/bin/bash
#SBATCH --job-name=sde
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:A6000:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=fcpei@andrew.cmu.edu
#SBATCH --exclude=babel-3-21

echo "$@";
source /data/user_data/fcpei/miniconda3/etc/profile.d/conda.sh;
conda activate finsde;
cd $userdata/10716-temp/sde-finance/scripts;
HYDRA_FULL_ERROR=1 WANDB__SERVICE_WAIT=300 python train.py "$@"
