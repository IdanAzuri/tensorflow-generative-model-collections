#!/bin/bash
#SBATCH --mem=60g
#SBATCH -c 60
#SBATCH --gres=gpu:1
#SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

module load tensorflow/1.5.0

dir=/cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections

cd $dir
source /cs/labs/daphna/idan.azuri/venv_64/bin/activate


#TODO truncated normal!

python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler truncated --batch_size 64 --mu 0.0 --sigma 2.0
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler truncated --batch_size 64 --mu 0.0 --sigma 3.0
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler truncated --batch_size 64 --mu 0.0 --sigma 5.0
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler truncated --batch_size 64 --mu 0.0 --sigma 10.0

#python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler truncated --batch_size 64 --mu 0.1 --sigma 0.15
#python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler truncated --batch_size 64 --mu 0.1 --sigma 0.2
#python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler truncated --batch_size 64 --mu 0.1 --sigma 0.3
#python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler truncated --batch_size 64 --mu 0.1 --sigma 0.4
#python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler truncated --batch_size 64 --mu 0.1 --sigma 0.5

python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_2.0 --preprocess True
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_3.0 --preprocess True
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_5.0 --preprocess True
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_10.0 --preprocess True

python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_2.0
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_3.0
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_5.0
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_10.0