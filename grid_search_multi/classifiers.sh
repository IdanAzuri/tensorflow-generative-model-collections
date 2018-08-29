#!/bin/bash
#SBATCH --mem=20g
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --time=1-20
# SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --array=0-7%7


module load tensorflow/1.5.0

dir=/cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections

cd $dir
source /cs/labs/daphna/idan.azuri/venv_64/bin/activate




if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]
then
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_ndist_3 --original fashion-mnist --seed 125
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_ndist_3 --original fashion-mnist --seed 125
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_ndist_3 --original fashion-mnist --seed 125
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_ndist_3 --original fashion-mnist --seed 125
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_ndist_3 --original fashion-mnist --seed 125
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_ndist_3 --original fashion-mnist --seed 125
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_ndist_3 --original fashion-mnist --seed 125
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_ndist_3 --original fashion-mnist --seed 125

elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]
then
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_ndist_3 --original fashion-mnist --seed 125
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_ndist_3 --original fashion-mnist --seed 125
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_ndist_3 --original fashion-mnist --seed 125
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_ndist_3 --original fashion-mnist --seed 125
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_ndist_3 --original fashion-mnist --seed 125
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_ndist_3 --original fashion-mnist --seed 125
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_ndist_3 --original fashion-mnist --seed 125

elif [ ${SLURM_ARRAY_TASK_ID} -eq 3 ]
then

python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_ndist_5 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_ndist_5 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_ndist_5 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_ndist_5 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_ndist_5 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_ndist_5 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_ndist_5 --original fashion-mnist --seed 88

elif [ ${SLURM_ARRAY_TASK_ID} -eq 4 ]
then
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_ndist_5 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_ndist_5 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_ndist_5 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_ndist_5 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_ndist_5 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_ndist_5 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_ndist_5 --original fashion-mnist --seed 88

elif [ ${SLURM_ARRAY_TASK_ID} -eq 5 ]
then
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_ndist_10 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_ndist_10 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_ndist_10 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_ndist_10 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_ndist_10 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_ndist_10 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_ndist_10 --original fashion-mnist --seed 88

elif [ ${SLURM_ARRAY_TASK_ID} -eq 6 ]
then

python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_ndist_10 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_ndist_10 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_ndist_10 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_ndist_10 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_ndist_10 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_ndist_10 --original fashion-mnist --seed 88
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_ndist_10 --original fashion-mnist --seed 88
elif [ ${SLURM_ARRAY_TASK_ID} -eq 0 ]
then
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_GaussianSample_mu_0.0_sigma_0.1_ndist_10 --original fashion-mnist --seed 125

fi


