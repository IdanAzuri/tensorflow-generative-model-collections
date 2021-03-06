#!/bin/bash
#SBATCH --mem=30g
#SBATCH -c 20
#SBATCH --gres=gpu:1
#SBATCH --time=1-00
#SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

module load tensorflow/1.5.0

dir=/cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections

cd $dir
source /cs/labs/daphna/idan.azuri/venv_64/bin/activate


python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset mnist --sampler uniform --batch_size 64 --dataset_order "czcc rzrc"
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_UniformSample_mu_0_sigma_0.15_czcc_rzrc --original mnist

python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset mnist --sampler uniform --batch_size 64 --dataset_order "rzrc czcc"
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_UniformSample_mu_0_sigma_0.15_rzrc_czcc --original mnist


