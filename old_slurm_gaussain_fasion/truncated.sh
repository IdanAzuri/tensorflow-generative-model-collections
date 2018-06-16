#!/bin/bash
#SBATCH --mem=60g
#SBATCH -c 30
#SBATCH --time=1-12
#SBATCH --gres=gpu:1
#SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

module load tensorflow/1.5.0

dir=/cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections

cd $dir
source /cs/labs/daphna/idan.azuri/venv_64/bin/activate


#TODO truncated normal!

python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler truncated --batch_size 64 --mu 0.0 --sigma 0.2
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler truncated --batch_size 64 --mu 0.0 --sigma 0.5
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler truncated --batch_size 64 --mu 0.0 --sigma 1.0


python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_0.2 --preprocess True
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_0.5 --preprocess True
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_1.0 --preprocess True

python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_0.2
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_0.5
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_1.0