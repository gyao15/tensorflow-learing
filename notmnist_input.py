from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

def read_data(batch_size,shuffle,iseval):
	pickle_file = 'D:/python/learntensor/notMNIST_data/notMNIST.pickle'
	with open(pickle_file,'rb') as f:
		save = pickle.load(f)
		if iseval:
			dataset = save['test_dataset']
			labels = save['test_labels']
		else:
			dataset = save['train_dataset']
			labels = save['train_labels']
		del save
	image = dataset.reshape(-1,28*28)
	labels = (np.arange(8)==labels[:,None]).astype(np.int32)
	_,labels = np.nonzero(labels)
	images = tf.train.input_producer(image,shuffle=None)
	label = tf.train.input_producer(labels,shuffle=None)
	num_examples = np.shape(image)[0]
	image = images.dequeue()
	labels = label.dequeue()
	

	min_queue_examples = int(0.2*num_examples)

	num_preprocess_threads = 16
	if shuffle:
		images,label_batch = tf.train.shuffle_batch(
			[image,labels],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples+3*batch_size,
			min_after_dequeue=min_queue_examples)
	else:
		images,label_batch = tf.train.batch(
			[image,labels],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples+3*batch_size)
	images = tf.reshape(images,[-1,28,28,1])

	tf.summary.image('images',images)
	
	print(num_examples)

	return num_examples,images,label_batch
