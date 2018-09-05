#!/bin/bash
#SBATCH --mem=8g
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=0-20
# SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMITc.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --array=0-8%8
SEEDS=(88 125 12 7 49 21 23 45 11)
SEED=${SEEDS[SLURM_ARRAY_TASK_ID]}
module load tensorflow/1.5.0

dir=/cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections

cd $dir
source /cs/labs/daphna/idan.azuri/venv_64/bin/activate

python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-uniform --batch_size 64 --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultiModalUniformSample_mu_0_sigma_0.15_ndist_10 --original fashion-mnist --seed $SEED

python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler uniform --batch_size 64 --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_UniformSample_mu_0.0_sigma_0.15_ndist_10 --original fashion-mnist --seed $SEED
