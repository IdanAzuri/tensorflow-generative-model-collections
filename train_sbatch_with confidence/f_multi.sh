#!/bin/bash
#SBATCH --mem=30g
#SBATCH -c 20
#SBATCH --gres=gpu:1
#SBATCH --time=0-20
#SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

module load tensorflow/1.5.0

dir=/cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections

cd $dir
source /cs/labs/daphna/idan.azuri/venv_64/bin/activate



python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.3 --dataset_order "czcc czrc rzcc rzrc" --result_dir results_with_confidence --original fashion-mnist




python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.3_czcc_czrc_rzcc_rzrc --original fashion-mnist --use_confidence True
