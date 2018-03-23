from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import tensorflow as tf
import notmnist

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir','D:/python/learntensor/notmnist_train',
	"""Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps',50000,
	"""Number of batches to run""")
tf.app.flags.DEFINE_integer('log_frequency', 50,
                            """How often to log results to the console.""")


def train():
	with tf.Graph().as_default():
		global_step = tf.train.get_or_create_global_step()

		with tf.device('/cpu:0'):
			images,labels = notmnist.inputs()

		logits = notmnist.inference(images)
		print(logits.get_shape())

		loss = notmnist.loss(logits,labels)
		accur = notmnist.accuracy(logits,labels)

		train_op = notmnist.train(loss,global_step)
		
		n = False

		class _LoggerHook(tf.train.SessionRunHook):
			def begin(self):
				self._step = -1
				self._start_time = time.time()

			def before_run(self,run_context):
				self._step += 1
				return tf.train.SessionRunArgs(loss)

			def after_run(self,run_context,run_values):
				if self._step%FLAGS.log_frequency == 0:
					current_time = time.time()
					duration = current_time-self._start_time
					self._start_time = current_time

					loss_value = run_values.results
					examples_per_sec = FLAGS.log_frequency*FLAGS.batch_size/duration
					sec_per_batch = float(duration/FLAGS.log_frequency)
					format_str = ('%s: step %d, loss = %.2f  (%.1f examples/sec; %.3f '
								'sec/batch)')
					print(format_str % (datetime.now(),self._step,loss_value,
									examples_per_sec,sec_per_batch))

		with tf.train.MonitoredTrainingSession(
			checkpoint_dir=FLAGS.train_dir,
			hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
			tf.train.NanTensorHook(loss),
			_LoggerHook()],
			config=tf.ConfigProto(log_device_placement=False)) as mon_sess:
			while not mon_sess.should_stop():
				mon_sess.run(train_op)
				#if n == Flase:
					#merged = tf.summary.merge_all()
					#sw = tf.summary.FileWriter(FLAGS.train_dir,mon_sess.graph)
					#n = True
				#summary = mon_sess.run(merged)
				#sw.add_summary(summary)

def main(argv=None):
	if tf.gfile.Exists(FLAGS.train_dir):
		tf.gfile.DeleteRecursively(FLAGS.train_dir)
	tf.gfile.MakeDirs(FLAGS.train_dir)
	train()

if __name__ == '__main__':
	tf.app.run()