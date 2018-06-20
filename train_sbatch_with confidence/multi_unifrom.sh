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

python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset mnist --sampler multi-uniform --batch_size 64 --dataset_order "czcc czrc rzcc rzrc" --result_dir results_with_confidence
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultiModalUniformSample_mu_0_sigma_0.15_czcc_czrc_rzcc_rzrc --preprocess True --original mnist  --use_confidence True
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultiModalUniformSample_mu_0_sigma_0.15_czcc_czrc_rzcc_rzrc --original mnist

python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset mnist --sampler multi-uniform --batch_size 64 --dataset_order "czcc rzcc czrc rzrc" --result_dir results_with_confidence
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultiModalUniformSample_mu_0_sigma_0.15_czcc_rzcc_czrc_rzrc --preprocess True --original mnist  --use_confidence True
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultiModalUniformSample_mu_0_sigma_0.15_czcc_rzcc_czrc_rzrc --original mnist

python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset mnist --sampler multi-uniform --batch_size 64 --dataset_order "rzcc rzrc czcc czrc" --result_dir results_with_confidence
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultiModalUniformSample_mu_0_sigma_0.15_rzcc_rzrc_czcc_czrc --preprocess True --original mnist  --use_confidence True
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultiModalUniformSample_mu_0_sigma_0.15_rzcc_rzrc_czcc_czrc --original mnist


python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset mnist --sampler multi-uniform --batch_size 64 --dataset_order "czrc czcc rzcc rzrc" --result_dir results_with_confidence
#python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultiModalUniformSample_mu_0_sigma_0.15_czrc_czcc_rzcc_rzrc --preprocess True --original mnist  --use_confidence True
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname mnist_MultiModalUniformSample_mu_0_sigma_0.15_czrc_czcc_rzcc_rzrc --original mnist

