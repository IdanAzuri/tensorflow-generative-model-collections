#!/bin/bash
#SBATCH --mem=60g
#SBATCH -c 30
#SBATCH --gres=gpu:1
#SBATCH --time=0-20
#SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

module load tensorflow/1.5.0

dir=/cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections

cd $dir
source /cs/labs/daphna/idan.azuri/venv_64/bin/activate






python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler gaussian --batch_size 64 --mu 0.0 --sigma 0.4 --dataset_order "czcc czrc rzcc rzrc"
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler gaussian --batch_size 64 --mu 0.0 --sigma 0.5 --dataset_order "czcc czrc rzcc rzrc"
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler gaussian --batch_size 64 --mu 0.0 --sigma 1.0 --dataset_order "czcc czrc rzcc rzrc"
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler gaussian --batch_size 64 --mu 0.1 --sigma 0.15 --dataset_order "czcc czrc rzcc rzrc"

python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_GaussianSampler_mu_0.0_sigma_0.4_czcc_czrc_rzcc_rzrc --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_GaussianSampler_mu_0.0_sigma_0.5_czcc_czrc_rzcc_rzrc --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_GaussianSampler_mu_0.0_sigma_1.0_czcc_czrc_rzcc_rzrc --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_GaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc --original fashion-mnist
