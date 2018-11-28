"""
Logistic Regression Object Patch 1.0

Patch notes:  -

Date of last edit: November-27-2018
Rui Nian

Current issues:  Output is always one, train test split not completed within class
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

import gc
import argparse

import os

import warnings
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


class LogisticRegression:

    """
    Class for Logistic Regression

    Attributes
        ------
                    sess: Tensorflow session

                features: Features of the data set, X matrix.  Shape of [Nx, m]
                  labels: Labels corresponding to the features, y matrix. Shape of [1, m]
                     nx: Number of features
                     ny: Number of outputs
                      m: Number of training examples
             train_size: % of data used for training

                      lr: Learning rate of logistic regression.  Default value = 0.003
          minibatch_size: Size of mini-batch for mini-batch gradient descent

                       X: Placeholder for feeding minibatches of features
                       y: Placeholder for feeding minibatches of labels

                       W: Weights of the logistic regression
                       b: Biases of the logistic regression





    """

    def __init__(self, sess, data, lr=0.003, minibatch_size=64, train_size=0.9):

        """
        Input data must be of shape: [Nx, m] and the labels must be in the first column
        """
        # Tensorflow session
        self.sess = sess

        # Data
        self.features = data[1:, :]
        self.labels = data[0, :]
        self.nx = self.features.shape[0]
        self.ny = 1
        self.m = self.features.shape[1]
        self.train_size = train_size

        # Machine learning parameters
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.total_batch_number = int((self.m / self.minibatch_size) * self.train_size)

        # Tensorflow variables
        self.X = tf.placeholder(dtype=tf.float32, shape=[self.nx, None])
        self.y = tf.placeholder(dtype=tf.float32, shape=[1, None])

        self.W = tf.get_variable('weights', shape=[self.ny, self.nx],
                                 initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable('biases', shape=[self.ny, 1])

        self.z = tf.matmul(self.W, self.X) + self.b

        # Loss function
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.z, labels=self.y))

        # Adam optimization
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        # Prediction
        self.pred = tf.sigmoid(self.z)

        # Accuracies
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.y), tf.float32))
        self.precision, self.prec_ops = tf.metrics.precision(labels=self.y, predictions=self.pred)
        self.recall, self.recall_ops = tf.metrics.recall(labels=self.y, predictions=self.pred)

        self.loss_history = []

        self.init = tf.global_variables_initializer()
        self.init_l = tf.local_variables_initializer()

    def train(self, features, labels):
        self.sess.run(self.optimizer, feed_dict={self.X: features, self.y: labels})
        curr_loss = self.sess.run(self.loss, feed_dict={self.X: features, self.y: labels})
        self.loss_history.append(curr_loss)
        return curr_loss

    def test(self, features, labels):
        return self.sess.run(self.pred, feed_dict={self.X: features, self.y: labels})

    def test_percent(self, features, labels):
        pred = tf.sigmoid(self.z)
        return self.sess.run(pred, feed_dict={self.X: features, self.y: labels})

    def model_evaluation(self, features, labels):
        return self.sess.run([self.accuracy, self.prec_ops, self.recall_ops], feed_dict={self.X: features, self.y: labels})

    def weights_and_biases(self):
        return sess.run([self.W, self.b])

    @staticmethod
    def random_seed(seed=8):
        np.random.seed(seed)
        tf.set_random_seed(seed)

    # Min max normalization
    @staticmethod
    def min_max_normalization(data):

        col_max = np.max(data, axis=1).reshape(data.shape[0], 1)
        col_min = np.min(data, axis=1).reshape(data.shape[0], 1)

        denominator = abs(col_max - col_min)

        # Fix divide by zero, replace value with 1 because these usually happen for boolean columns
        for index, value in enumerate(denominator):
            if value[0] == 0:
                denominator[index] = 1

        return np.divide((data - col_max), denominator)

    # Define Normalization
    @staticmethod
    def normalization(data):
        col_mean = np.mean(data, axis=1).reshape(data.shape[0], 1)
        col_std = np.std(data, axis=1).reshape(data.shape[0], 1)

        # Fix divide by 0 since sometimes, standard deviation can be zero
        for index, value in enumerate(col_std):
            if value[0] == 0:
                col_std[index] = 1

        return np.divide((data - col_mean), col_std)

    # Modified Normalization robust to outliers
    @staticmethod
    def mod_normalization(data):
        col_median = np.median(data, axis=1).reshape(data.shape[0], 1)
        mad = np.median(abs(data - col_median), axis=1).reshape(data.shape[0], 1)

        # Fix divide by 0 for when MAD = 0
        for index, value in enumerate(mad):
            if value[0] == 0:
                mad[index] = 1

        return np.divide(0.6745 * (data - col_median), mad)

    def __str__(self):
        return "Logistic Regression using {} features.".format(self.nx)

    def __repr__(self):
        return "LogisticRegression()"


if __name__ == "__main__":

    # Loading data
    raw_data = pd.read_csv('data/64_data.csv')

    # Get feature headers
    feature_names = list(raw_data)

    # Delete Unnamed: 0 and label column
    del feature_names[0]
    del feature_names[0]

    # Turn Pandas dataframe into NumPy Array
    raw_data = raw_data.values
    print("Raw data has {} features with {} examples.".format(raw_data.shape[1], raw_data.shape[0]))

    # Delete the index column given by Pandas
    raw_data = np.delete(raw_data, [0], axis=1)
    np.random.shuffle(raw_data)
    raw_data = raw_data.T

    # Data partitation into features and labels
    features = raw_data[1:, :]
    labels = raw_data[0, :]

    # Train / test split
    train_size = 0.9
    train_values = int(features.shape[1] * train_size)

    train_X = features[:, 0:train_values]
    train_y = labels[0:train_values].reshape(1, train_X.shape[1])

    test_X = features[:, train_values:]
    test_y = labels[train_values:].reshape(1, test_X.shape[1])

    epochs = 2

    with tf.Session() as sess:

        log_reg = LogisticRegression(sess, raw_data, train_size=0.9)
        log_reg.random_seed()

        sess.run(log_reg.init)
        sess.run(log_reg.init_l)

        for epoch in range(epochs):

            for i in range(log_reg.total_batch_number):
                batch_index = i * log_reg.minibatch_size
                batch_X = train_X[:, batch_index:(batch_index + log_reg.minibatch_size)]
                batch_y = train_y[:, batch_index:(batch_index + log_reg.minibatch_size)]

                current_loss = log_reg.train(batch_X, batch_y)
                train_accuracy, _, _ = log_reg.model_evaluation(train_X, train_y)
                test_accuracy, _, _ = log_reg.model_evaluation(test_X, test_y)

                if i % 10 == 0:
                    print("Epoch: {} | loss: {:5f} | train acc: {:5f} | test acc: {:5f}".format(epoch + 1,
                                                                                                current_loss,
                                                                                                train_accuracy,
                                                                                                test_accuracy))

        Pred = log_reg.test(test_X, test_y)
        Acc, Precision, Recall = log_reg.model_evaluation(test_X, test_y)
