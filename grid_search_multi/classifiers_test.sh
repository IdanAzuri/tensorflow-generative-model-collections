#!/bin/bash
#SBATCH --mem=4g
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=1-20
# SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --array=0-8%2
SEEDS=(88 125 12 7 49 21 23 45 11)
SEED=${SEEDS[((SLURM_ARRAY_TASK_ID ))]}
echo $SEED
echo
SEED=${SEEDS[SLURM_ARRAY_TASK_ID]}
echo $SEED
module load tensorflow/1.5.0

dir=/cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections

cd $dir
source /cs/labs/daphna/idan.azuri/venv_64/bin/activate


