#!/bin/bash
#SBATCH --mem=60g
#SBATCH -c 30
#SBATCH --gres=gpu:1
#SBATCH --time=1-00
#SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

module load tensorflow/1.5.0

dir=/cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections

cd $dir
source /cs/labs/daphna/idan.azuri/venv_64/bin/activate

python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset mnist --sampler uniform --batch_size 64 --dataset_order "czrc rzcc"
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_UniformSample_mu_0_sigma_0.15_czrc_rzcc --original mnist

python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset mnist --sampler uniform --batch_size 64 --dataset_order "rzcc czrc"
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_UniformSample_mu_0_sigma_0.15_rzcc_czrc --original mnist
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_UniformSample_mu_0_sigma_0.15_czcc_czrc_rzcc_rzrc --preprocess True --original mnist
