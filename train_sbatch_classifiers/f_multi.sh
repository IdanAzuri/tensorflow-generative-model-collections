#!/bin/bash
#SBATCH --mem=60g
#SBATCH -c 20
#SBATCH --gres=gpu:1
#SBATCH --time=0-20
#SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

module load tensorflow/1.5.0

dir=/cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections

cd $dir
source /cs/labs/daphna/idan.azuri/venv_64/bin/activate










#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.3_czcc_czrc_rzcc_rzrc --preprocess True --original fashion-mnist
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.3_czcc_rzcc_czrc_rzrc --preprocess True --original fashion-mnist
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.3_rzcc_rzrc_czcc_czrc --preprocess True --original fashion-mnist
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.3_czrc_czcc_rzcc_rzrc --preprocess True --original fashion-mnist

python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.3_czcc_czrc_rzcc_rzrc --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.3_czcc_rzcc_czrc_rzrc --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.3_rzcc_rzrc_czcc_czrc --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.3_czrc_czcc_rzcc_rzrc --original fashion-mnist










#
#
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15 --preprocess True
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1 --preprocess True
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2 --preprocess True
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.3 --preprocess True
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.4 --preprocess True
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.5 --preprocess True
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_1.0 --preprocess True
#
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.3
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.4
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.5
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_1.0
