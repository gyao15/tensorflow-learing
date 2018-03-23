from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
import notmnist

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir','D:/python/learntensor/notmnist_eval',
	"""Directory where to write event logs""")
tf.app.flags.DEFINE_string('checkpoint_dir','D:/python/learntensor/notmnist_train',
	"""Directory where to read model parameters""")

def evaluate():
	with tf.Graph().as_default() as g:
		num_examples,images,labels = notmnist.inputs(True)

		logits = notmnist.inference(images)
		topk_op = tf.nn.in_top_k(logits,labels,1)

		variable_averages = tf.train.ExponentialMovingAverage(
			notmnist.MOVING_AVERAGE_DECAY)
		variable_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variable_to_restore)

		summary_op = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,g)

		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess,ckpt.model_checkpoint_path)
				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
			else:
				print('No checkpoint file found')
				return

			coord = tf.train.Coordinator()
			try:
				threads = []
				for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
					threads.extend(qr.create_threads(sess,coord=coord,daemon=True,start=True))

				num_iter = int(math.ceil(num_examples/FLAGS.batch_size))
				step = 0
				true_count = 0
				total_samples = num_iter * FLAGS.batch_size
				while step < num_iter and not coord.should_stop():
					predictions = sess.run([topk_op])
					true_count += np.sum(predictions)
					step += 1

				precision = true_count/total_samples
				print('%s: precision @ 1 = %.3f' % (datetime.now(),precision))

				summary = tf.Summary()
				summary.ParseFromString(sess.run(summary_op))
				summary.value.add(tag='precision@1',simple_value=precision)
				summary_writer.add_summary(summary,global_step)
			except Exception as e:
				coord.request_stop(e)

			coord.request_stop()
			coord.join(threads, stop_grace_period_secs=10)

def main(argv=None):
	if tf.gfile.Exists(FLAGS.eval_dir):
		tf.gfile.DeleteRecursively(FLAGS.eval_dir)
	tf.gfile.MakeDirs(FLAGS.eval_dir)
	evaluate()

if __name__ == '__main__':
	tf.app.run()