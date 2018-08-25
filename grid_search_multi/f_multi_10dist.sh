#!/bin/bash
#SBATCH --mem=70g
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --time=1-20
# SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

module load tensorflow/1.5.0

dir=/cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections

cd $dir
source /cs/labs/daphna/idan.azuri/venv_64/bin/activate





python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.1 --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.17 --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.2 --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.25 --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.2 --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.3 --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.4 --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.8 --sigma 0.3 --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.8 --sigma 0.4 --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.15 --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.2 --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.4 --ndist 10

python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.2_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.4_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.4_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.15_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_ndist_10 --original fashion-mnist
