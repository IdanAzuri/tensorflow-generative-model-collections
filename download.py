########################################################################
#
# Functions for downloading and extracting data-files from the internet.
#
# Implemented in Python 3.5
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################
from __future__ import print_function
import tarfile
import urllib.request


########################################################################


def _print_download_progress(count, block_size, total_size):
	"""
	Function used for printing the download progress.
	Used as a call-back function in maybe_download_and_extract().
	"""

	# Percentage completion.
	pct_complete = float(count * block_size) / total_size

	# Status-message. Note the \r which means the line should overwrite itself.
	msg = "\r- Download progress: {0:.1%}".format(pct_complete)

	# Print it.
	sys.stdout.write(msg)
	sys.stdout.flush()


########################################################################


def maybe_download_and_extract(url, download_dir):
	"""
	Download and extract the data if it doesn't already exist.
	Assumes the url is a tar-ball file.
	:param url:
		Internet URL for the tar-file to download.
		Example: "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
	:param download_dir:
		Directory where the downloaded file is saved.
		Example: "data/CIFAR-10/"
	:return:
		Nothing.
	"""

	# Filename for saving the file downloaded from the internet.
	# Use the filename from the URL and add it to the download_dir.
	filename = url.split('/')[-1]
	file_path = os.path.join(download_dir, filename)

	# Check if the file already exists.
	# If it exists then we assume it has also been extracted,
	# otherwise we need to download and extract it now.
	if not os.path.exists(file_path):
		# Check if the download directory exists, otherwise create it.
		if not os.path.exists(download_dir):
			os.makedirs(download_dir)

		# Download the file from the internet.
		file_path, _ = urllib.request.urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

		print()
		print("Download finished. Extracting files.")

		if file_path.endswith(".zip"):
			# Unpack the zip-file.
			zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
		elif file_path.endswith((".tar.gz", ".tgz")):
			# Unpack the tar-ball.
			tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

		print("Done.")
	else:
		print("Data has apparently already been downloaded and unpacked.")


########################################################################

"""
Modification of https://github.com/stanfordnlp/treelstm/blob/master/scripts/download.py
Downloads the following:
- Celeb-A dataset
- LSUN dataset
- MNIST dataset
"""

<<<<<<< HEAD
=======

>>>>>>> d05f275b65195dd6a63406c3a0f34c7fa3b2e406
import os
import sys
import json
import zipfile
import argparse
import requests
import subprocess
from tqdm import tqdm
from six.moves import urllib

parser = argparse.ArgumentParser(description='Download dataset for DCGAN.')
parser.add_argument('datasets', metavar='N', type=str, nargs='+', choices=['celebA', 'lsun', 'mnist'],
                    help='name of dataset to download [celebA, lsun, mnist]')


def download(url, dirpath):
	filename = url.split('/')[-1]
	filepath = os.path.join(dirpath, filename)
	u = urllib.request.urlopen(url)
	f = open(filepath, 'wb')
	filesize = int(u.headers["Content-Length"])
	print("Downloading: %s Bytes: %s" % (filename, filesize))

	downloaded = 0
	block_sz = 8192
	status_width = 70
	while True:
		buf = u.read(block_sz)
		if not buf:
			print('')
			break
		else:
			print('', end='\r')
		downloaded += len(buf)
		f.write(buf)
		status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") % (
		'=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
		print(status, end='')
		sys.stdout.flush()
	f.close()
	return filepath


def download_file_from_google_drive(id, destination):
	URL = "https://docs.google.com/uc?export=download"
	session = requests.Session()

	response = session.get(URL, params={'id': id}, stream=True)
	token = get_confirm_token(response)

	if token:
		params = {'id': id, 'confirm': token}
		response = session.get(URL, params=params, stream=True)

	save_response_content(response, destination)


def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value
	return None


def save_response_content(response, destination, chunk_size=32 * 1024):
	total_size = int(response.headers.get('content-length', 0))
	with open(destination, "wb") as f:
		for chunk in tqdm(response.iter_content(chunk_size), total=total_size, unit='B', unit_scale=True, desc=destination):
			if chunk:  # filter out keep-alive new chunks
				f.write(chunk)


def unzip(filepath):
	print("Extracting: " + filepath)
	dirpath = os.path.dirname(filepath)
	with zipfile.ZipFile(filepath) as zf:
		zf.extractall(dirpath)
	os.remove(filepath)


def download_celeb_a(dirpath):
	data_dir = 'celebA'
	if os.path.exists(os.path.join(dirpath, data_dir)):
		print('Found Celeb-A - skip')
		return

	filename, drive_id = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
	save_path = os.path.join(dirpath, filename)

	if os.path.exists(save_path):
		print('[*] {} already exists'.format(save_path))
	else:
		download_file_from_google_drive(drive_id, save_path)

	zip_dir = ''
	with zipfile.ZipFile(save_path) as zf:
		zip_dir = zf.namelist()[0]
		zf.extractall(dirpath)
	os.remove(save_path)
	os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))


def _list_categories(tag):
	url = 'http://lsun.cs.princeton.edu/htbin/list.cgi?tag=' + tag
	f = urllib.request.urlopen(url)
	return json.loads(f.read())


def _download_lsun(out_dir, category, set_name, tag):
	url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
	      '&category={category}&set={set_name}'.format(**locals())
	print(url)
	if set_name == 'test':
		out_name = 'test_lmdb.zip'
	else:
		out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
	out_path = os.path.join(out_dir, out_name)
	cmd = ['curl', url, '-o', out_path]
	print('Downloading', category, set_name, 'set')
	subprocess.call(cmd)


def download_lsun(dirpath):
	data_dir = os.path.join(dirpath, 'lsun')
	if os.path.exists(data_dir):
		print('Found LSUN - skip')
		return
	else:
		os.mkdir(data_dir)

	tag = 'latest'
	# categories = _list_categories(tag)
	categories = ['bedroom']

	for category in categories:
		_download_lsun(data_dir, category, 'train', tag)
		_download_lsun(data_dir, category, 'val', tag)
	_download_lsun(data_dir, '', 'test', tag)


def download_mnist(dirpath):
	data_dir = os.path.join(dirpath, 'mnist')
	if os.path.exists(data_dir):
		print('Found MNIST - skip')
		return
	else:
		os.mkdir(data_dir)
	url_base = 'http://yann.lecun.com/exdb/mnist/'
	file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
	for file_name in file_names:
		url = (url_base + file_name).format(**locals())
		print(url)
		out_path = os.path.join(data_dir, file_name)
		cmd = ['curl', url, '-o', out_path]
		print('Downloading ', file_name)
		subprocess.call(cmd)
		cmd = ['gzip', '-d', out_path]
		print('Decompressing ', file_name)
		subprocess.call(cmd)


def prepare_data_dir(path='./data'):
	if not os.path.exists(path):
		os.mkdir(path)


if __name__ == '__main__':
	args = parser.parse_args()
	prepare_data_dir()

	if any(name in args.datasets for name in ['CelebA', 'celebA', 'celebA']):
		download_celeb_a('./data')
	if 'lsun' in args.datasets:
		download_lsun('./data')
	if 'mnist' in args.datasets:
		download_mnist('./data')
