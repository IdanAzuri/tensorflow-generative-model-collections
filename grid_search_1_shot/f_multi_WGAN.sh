#!/bin/bash
#SBATCH --mem=20g
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --time=0-20
# SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --array=0-9%1
SEEDS=(88 125 12 7 49 21 23 45 11)
SEED=${SEEDS[((SLURM_ARRAY_TASK_ID ))]}
dir=/cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections

cd $dir
source /cs/labs/daphna/idan.azuri/venv_64/bin/activate





python3 main.py --gan_type MultiModalInfoGAN --epoch 6000 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.3  --ndist 5 --seed 12 --pref one-shot_wgan --wgan True
python3 main.py --gan_type MultiModalInfoGAN_phase2 --epoch 6000 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.3  --ndist 5 --seed 12 --pref one-shot_wgan --wgan True
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_ndist_5_WGAN --original mnist --seed 12

python3 main.py --gan_type MultiModalInfoGAN --epoch 6000 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.3  --ndist 3 --seed 12 --pref one-shot_wgan --wgan True
python3 main.py --gan_type MultiModalInfoGAN_phase2 --epoch 6000 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.3  --ndist 3 --seed 12 --pref one-shot_wgan --wgan True
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_ndist_3_WGAN --original mnist --seed 12

python3 main.py --gan_type MultiModalInfoGAN --epoch 6000 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.2 --sigma 0.2  --ndist 5 --seed 12 --pref one-shot_wgan --wgan True
python3 main.py --gan_type MultiModalInfoGAN_phase2 --epoch 6000 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.2 --sigma 0.2  --ndist 5 --seed 12 --pref one-shot_wgan --wgan True
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultivariateGaussianSampler_mu_0.2_sigma_0.2_ndist_5_WGAN --original mnist --seed 12

python3 main.py --gan_type MultiModalInfoGAN --epoch 6000 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.2 --sigma 0.2  --ndist 3 --seed 12 --pref one-shot_wgan --wgan True
python3 main.py --gan_type MultiModalInfoGAN_phase2 --epoch 6000 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.2 --sigma 0.2  --ndist 3 --seed 12 --pref one-shot_wgan --wgan True
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultivariateGaussianSampler_mu_0.2_sigma_0.2_ndist_3_WGAN --original mnist --seed 12