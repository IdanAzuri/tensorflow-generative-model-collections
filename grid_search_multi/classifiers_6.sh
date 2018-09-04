#!/bin/bash
#SBATCH --mem=4g
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=0-20
# SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --array=0-9%3
SEEDS=(125 12)

SEED=${SEEDS[((SLURM_ARRAY_TASK_ID ))]}
module load tensorflow/1.5.0

dir=/cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections

cd $dir
source /cs/labs/daphna/idan.azuri/venv/bin/activate


python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_ndist_10 --original fashion-mnist --seed $SEED
