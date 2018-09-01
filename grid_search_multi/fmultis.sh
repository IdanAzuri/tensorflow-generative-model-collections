#!/bin/bash
#SBATCH --mem=16g
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --time=1-20
# SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --array=0-89%9


module load tensorflow/1.5.0

dir=/cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections

cd $dir
source /cs/labs/daphna/idan.azuri/venv_64/bin/activate

SEEDS=(88 125 12 7 49 21 23 45 11)

SEED=${SEEDS[((SLURM_ARRAY_TASK_ID/10))]}

if [ $(( $SLURM_ARRAY_TASK_ID % 9))  -eq 1 ]
then
echo $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.1 --ndist 3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.17 --ndist 3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.2 --ndist 3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.25 --ndist 3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.2 --ndist 3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.3 --ndist 3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.4 --ndist 3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.8 --sigma 0.3 --ndist 3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.8 --sigma 0.4 --ndist 3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.15 --ndist 3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.4 --ndist 3 --seed $SEED

python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.2_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.4_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.4_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.15_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_ndist_3 --original fashion-mnist --seed $SEED

elif [ $(( $SLURM_ARRAY_TASK_ID % 9)) -eq 2 ]
then
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.25 --ndist 3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.8 --sigma 0.2 --ndist 3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.2 --ndist 3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.22 --ndist 3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.25 --ndist 3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.3 --ndist 3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.5 --ndist 3 --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_ndist_3 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_ndist_3 --original fashion-mnist --seed $SEED

elif [ $(( $SLURM_ARRAY_TASK_ID % 9)) -eq 3 ]
then

python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.1 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.17 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.2 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.25 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.2 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.3 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.4 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.8 --sigma 0.3 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.8 --sigma 0.4 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.15 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.2 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.4 --ndist 5 --seed $SEED

python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.2_ndist_5 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_ndist_5 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.4_ndist_5 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_ndist_5 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.4_ndist_5 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.15_ndist_5 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_ndist_5 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_ndist_5 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_ndist_5 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_ndist_5 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_ndist_5 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_ndist_5 --original fashion-mnist

elif [ $(( $SLURM_ARRAY_TASK_ID % 9)) -eq 4 ]
then
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.25 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.8 --sigma 0.2 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.2 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.22 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.25 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.3 --ndist 5 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.5 --ndist 5 --seed $SEED

python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_ndist_5 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_ndist_5 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_ndist_5 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_ndist_5 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_ndist_5 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_ndist_5 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_ndist_5 --original fashion-mnist --seed $SEED

elif [ $(( $SLURM_ARRAY_TASK_ID % 9)) -eq 5 ]
then
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.1 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.17 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.2 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.25 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.2 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.3 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.4 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.8 --sigma 0.3 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.8 --sigma 0.4 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.15 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.2 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.4 --ndist 10 --seed $SEED

python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.2_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.4_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.4_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.15_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_ndist_10 --original fashion-mnist --seed $SEED

elif [ $(( $SLURM_ARRAY_TASK_ID % 9)) -eq 6 ]
then

python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.25 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.8 --sigma 0.2 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.8 --sigma 0.25 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.2 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.22 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.25 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.3 --ndist 10 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.5 --ndist 10 --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.25_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_ndist_10 --original fashion-mnist --seed $SEED

elif [ $(( $SLURM_ARRAY_TASK_ID % 9)) -eq 7 ]
then
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_GaussianSample_mu_0.0_sigma_0.1_ndist_10 --original fashion-mnist --seed $SEED


elif [ $(( $SLURM_ARRAY_TASK_ID % 9)) -eq 8 ]
then
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler gaussian --batch_size 64 --mu 0.0 --sigma 0.1 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler gaussian --batch_size 64 --mu 0.0 --sigma 0.17 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler gaussian --batch_size 64 --mu 0.0 --sigma 0.2 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler gaussian --batch_size 64 --mu 0.0 --sigma 0.25 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler gaussian --batch_size 64 --mu 0.0 --sigma 0.3 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler gaussian --batch_size 64 --mu 0.0 --sigma 0.35 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler gaussian --batch_size 64 --mu 0.0 --sigma 0.4 --seed $SEED
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler gaussian --batch_size 64 --mu 0.0 --sigma 0.5 --seed $SEED

python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_GaussianSample_mu_0.0_sigma_0.1_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_GaussianSample_mu_0.0_sigma_0.17_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_GaussianSample_mu_0.0_sigma_0.2_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_GaussianSample_mu_0.0_sigma_0.25_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_GaussianSample_mu_0.0_sigma_0.3_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_GaussianSample_mu_0.0_sigma_0.35_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_GaussianSample_mu_0.0_sigma_0.4_ndist_10 --original fashion-mnist --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_GaussianSample_mu_0.0_sigma_0.5_ndist_10 --original fashion-mnist --seed $SEED

elif [ $(( $SLURM_ARRAY_TASK_ID % 9)) -eq 9 ]
then

python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset mnist --sampler multi-uniform --batch_size 64 --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultiModalUniformSample_mu_0_sigma_0.15_ndist_10 --original mnist --seed $SEED

python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler uniform --batch_size 64 --seed $SEED
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_UniformSample_mu_0.0_sigma_0.15_ndist_10 --original fashion-mnist --seed $SEED

fi


