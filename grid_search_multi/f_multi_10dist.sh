python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.1 --dataset_order "czcc czrc rzcc rzrc" --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.17 --dataset_order "czcc czrc rzcc rzrc" --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.2 --dataset_order "czcc czrc rzcc rzrc" --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.1 --sigma 0.25 --dataset_order "czcc czrc rzcc rzrc" --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.2 --dataset_order "czcc czrc rzcc rzrc" --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.3 --dataset_order "czcc czrc rzcc rzrc" --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.7 --sigma 0.4 --dataset_order "czcc czrc rzcc rzrc" --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.8 --sigma 0.3 --dataset_order "czcc czrc rzcc rzrc" --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 0.8 --sigma 0.4 --dataset_order "czcc czrc rzcc rzrc" --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.15 --dataset_order "czcc czrc rzcc rzrc" --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.2 --dataset_order "czcc czrc rzcc rzrc" --ndist 10
python3 main.py --gan_type MultiModalInfoGAN --epoch 40 --dataset fashion-mnist --sampler multi-gaussian --batch_size 64 --mu 1.0 --sigma 0.4 --dataset_order "czcc czrc rzcc rzrc" --ndist 10

python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10 --original fashion-mnist
python3 classifier.py --dir_name /cs/labs/daphna/idan.azuri/tensorflow-generative-model-collections/ --fname fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10 --original fashion-mnist
