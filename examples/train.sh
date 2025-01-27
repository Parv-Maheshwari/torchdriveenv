#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --array=0-4
#SBATCH --output="slurm_output/slurm-%A_%a.out"

ln -sf $SBATCH_OUTPUT latest
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python
which python
which pip

pip install torchdriveenv[baselines]

export IAI_API_KEY=oOIwcHIF0v4g4t7awNvEk6QObz7eflCq9MCCQNLQ
export WANDB_API_KEY=8160f073728588c7ce58759b1b24ce284b05705c

python rl_training.py
