import numpy as np
import tensorflow as tf


labels = tf.constant([0, 0, 0, 0, 1], dtype=tf.int32)
predictions = tf.constant([0, 0, 0, 0, 0], dtype=tf.int32)
specificity = 0

metrics = tf.metrics.sensitivity_at_specificity(labels, predictions, specificity)

with tf.Session() as sess:
    Sens, Update_op = sess.run(metrics)
