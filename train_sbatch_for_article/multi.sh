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



python3 main.py --gan_type MultiModalInfoGAN --epoch 60 --dataset mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.3 --dataset_order "czcc czrc rzcc rzrc"
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultivariateGaussianSampler --original mnist



