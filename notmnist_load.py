# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Config the matplotlib backend as plotting inline in IPython
#matplotlib inline

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir','D:/python/learntensor/',
						   """Path to the notmnist data""")



num_classes = 8
np.random.seed(133)

def maybe_extract(filename,force=False):
	root = os.path.join(FLAGS.data_dir,filename)
	data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root))
					if os.path.isdir(os.path.join(root, d))]
	if len(data_folders) != num_classes:
		raise Exception(
			'Expected %d folders, one per class. Found %d instead.' % (num_classes, len(data_folders)))
	return data_folders

train_filename = 'notMNIST_large/'
test_filename = 'notMNIST_small/notMNIST_small/'
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)



image_size = 28
pixel_depth = 255.0

def load_letter(folder,min_num_images):

	image_files = os.listdir(folder)
	dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
						 dtype=np.float32)
	print(folder)
	num_images = 0
	for image in image_files:
		image_file = os.path.join(folder, image)
		try:
			image_data = (imageio.imread(image_file).astype(float) - 
						 pixel_depth / 2) / pixel_depth
			if image_data.shape != (image_size, image_size):
				raise Exception('Unexpected image shape: %s' % str(image_data.shape))
			dataset[num_images, :, :] = image_data
			num_images = num_images + 1
		except (IOError, ValueError) as e:
			print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
	
	dataset = dataset[0:num_images, :, :]
	if num_images < min_num_images:
		raise Exception('Many fewer images than expected: %d < %d' %
						(num_images, min_num_images))
	
	print('Full dataset tensor:', dataset.shape)
	print('Mean:', np.mean(dataset))
	print('Standard deviation:', np.std(dataset))
	return dataset
		
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
	dataset_names = []
	for folder in data_folders:
		set_filename = folder + '.pickle'
		dataset_names.append(set_filename)
		if os.path.exists(set_filename) and not force:
		# You may override by setting force=True.
			print('%s already present - Skipping pickling.' % set_filename)
		else:
			print('Pickling %s.' % set_filename)
			dataset = load_letter(folder, min_num_images_per_class)
			try:
				with open(set_filename, 'wb') as f:
					pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
			except Exception as e:
				print('Unable to save data to', set_filename, ':', e)
  
	return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)




def make_arrays(nb_rows, img_size):
	if nb_rows:
		dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
		labels = np.ndarray(nb_rows, dtype=np.int32)
	else:
		dataset, labels = None, None
	return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
	num_classes = len(pickle_files)
	valid_dataset, valid_labels = make_arrays(valid_size, image_size)
	train_dataset, train_labels = make_arrays(train_size, image_size)
	vsize_per_class = valid_size // num_classes
	tsize_per_class = train_size // num_classes
	
	start_v, start_t = 0, 0
	end_v, end_t = vsize_per_class, tsize_per_class
	end_l = vsize_per_class+tsize_per_class
	for label, pickle_file in enumerate(pickle_files):       
		try:
			with open(pickle_file, 'rb') as f:
				letter_set = pickle.load(f)
		# let's shuffle the letters to have random validation and training set
				np.random.shuffle(letter_set)
				if valid_dataset is not None:
					valid_letter = letter_set[:vsize_per_class, :, :]
					valid_dataset[start_v:end_v, :, :] = valid_letter
					valid_labels[start_v:end_v] = label
					start_v += vsize_per_class
					end_v += vsize_per_class
					
				train_letter = letter_set[vsize_per_class:end_l, :, :]
				train_dataset[start_t:end_t, :, :] = train_letter
				train_labels[start_t:end_t] = label
				start_t += tsize_per_class
				end_t += tsize_per_class
		except Exception as e:
			print('Unable to process data from', pickle_file, ':', e)
			raise
	
	return valid_dataset, valid_labels, train_dataset, train_labels
			
			
train_size = 20000
valid_size = 1000
test_size = 1000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)







#from http://www.hankcs.com/ml/notmnist.html
import hashlib
 
 
def extract_overlap_hash_where(dataset_1, dataset_2):
	dataset_hash_1 = np.array([hashlib.sha256(img).hexdigest() for img in dataset_1])
	dataset_hash_2 = np.array([hashlib.sha256(img).hexdigest() for img in dataset_2])
	overlap = {}
	for i, hash1 in enumerate(dataset_hash_1):
		duplicates = np.where(dataset_hash_2 == hash1)
		if len(duplicates[0]):
			overlap[i] = duplicates[0]
	return overlap
 
 

 
 
def sanitize(dataset_1, dataset_2, labels_1):
	dataset_hash_1 = np.array([hashlib.sha256(img).hexdigest() for img in dataset_1])
	dataset_hash_2 = np.array([hashlib.sha256(img).hexdigest() for img in dataset_2])
	overlap = []  # list of indexes
	for i, hash1 in enumerate(dataset_hash_1):
		duplicates = np.where(dataset_hash_2 == hash1)
		if len(duplicates[0]):
			overlap.append(i)
	return np.delete(dataset_1, overlap, 0), np.delete(labels_1, overlap, None)
 
 
overlap_test_train = extract_overlap_hash_where(test_dataset, train_dataset)
print('Number of overlaps:', len(overlap_test_train.keys()))

 
test_dataset_sanit, test_labels_sanit = sanitize(test_dataset, train_dataset, test_labels)
print('Overlapping images removed from test_dataset: ', len(test_dataset) - len(test_dataset_sanit))
valid_dataset_sanit, valid_labels_sanit = sanitize(valid_dataset, train_dataset, valid_labels)
print('Overlapping images removed from valid_dataset: ', len(valid_dataset) - len(valid_dataset_sanit))
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_labels_sanit.shape, valid_labels_sanit.shape)
print('Testing:', test_dataset_sanit.shape, test_labels_sanit.shape)
pickle_file = os.path.join(FLAGS.data_dir,'notMNIST_data/notMNIST.pickle')
 
try:
	f = open(pickle_file, 'wb')
	save = {
		'train_dataset': train_dataset,
		'train_labels': train_labels,
		'valid_dataset': valid_dataset_sanit,
		'valid_labels': valid_labels_sanit,
		'test_dataset': test_dataset_sanit,
		'test_labels': test_labels_sanit,
	}
	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	f.close()
except Exception as e:
	print('Unable to save data to', pickle_file, ':', e)
	raise
