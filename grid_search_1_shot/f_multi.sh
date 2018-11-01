#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=0-10
#SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

module load tensorflow/1.5.0

dir=/cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections

cd $dir
source /cs/labs/daphna/idan.azuri/venv_64/bin/activate





python3 main.py --gan_type MultiModalInfoGAN --epoch 6000 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.3  --ndist 5 --seed 12 --pref one-shot_gram 
python3 main.py --gan_type MultiModalInfoGAN_phase2 --epoch 6000 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.3  --ndist 5 --seed 12 --pref one-shot_gram 
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_ndist_5 --original mnist --seed 12

python3 main.py --gan_type MultiModalInfoGAN --epoch 6000 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.3  --ndist 3 --seed 12 --pref one-shot_gram 
python3 main.py --gan_type MultiModalInfoGAN_phase2 --epoch 6000 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.3  --ndist 3 --seed 12 --pref one-shot_gram 
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_ndist_3 --original mnist --seed 12

python3 main.py --gan_type MultiModalInfoGAN --epoch 6000 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.2 --sigma 0.2  --ndist 5 --seed 12 --pref one-shot_gram 
python3 main.py --gan_type MultiModalInfoGAN_phase2 --epoch 6000 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.2 --sigma 0.2  --ndist 5 --seed 12 --pref one-shot_gram 
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultivariateGaussianSampler_mu_0.2_sigma_0.2_ndist_5 --original mnist --seed 12

python3 main.py --gan_type MultiModalInfoGAN --epoch 6000 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.2 --sigma 0.2  --ndist 3 --seed 12 --pref one-shot_gram 
python3 main.py --gan_type MultiModalInfoGAN_phase2 --epoch 6000 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.2 --sigma 0.2  --ndist 3 --seed 12 --pref one-shot_gram 
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultivariateGaussianSampler_mu_0.2_sigma_0.2_ndist_3 --original mnist --seed 12