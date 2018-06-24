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


#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_UniformSample_mu_0_sigma_0.15_czcc_czrc_rzcc_rzrc --preprocess True --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_UniformSample_mu_0_sigma_0.15_czcc_czrc_rzcc_rzrc --original fashion-mnist


#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_UniformSample_mu_0_sigma_0.15_czcc_rzcc_czrc_rzrc --preprocess True --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_UniformSample_mu_0_sigma_0.15_czcc_rzcc_czrc_rzrc --original fashion-mnist


#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_UniformSample_mu_0_sigma_0.15_rzcc_rzrc_czcc_czrc --preprocess True --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_UniformSample_mu_0_sigma_0.15_rzcc_rzrc_czcc_czrc --original fashion-mnist



#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_UniformSample_mu_0_sigma_0.15_czrc_czcc_rzcc_rzrc --preprocess True --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_UniformSample_mu_0_sigma_0.15_czrc_czcc_rzcc_rzrc --original fashion-mnist

