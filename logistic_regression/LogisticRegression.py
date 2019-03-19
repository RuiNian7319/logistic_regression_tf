"""
Logistic Regression Object Patch 1.2

Patch notes:  Updates to locations

Date of last edit: Jan-16-2018
Rui Nian

Current issues:  Output shape is always one
                 Maybe retranspose the data
                 Add threshold to rounding?
                 Add more user inputs

   Patch notes:  Added plotting feature
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import pickle

import gc
import argparse

from Accuracy_Plots import plots

from copy import deepcopy

import os

import warnings
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


class LogisticRegression:

    """
    Class for Logistic Regression

    Attributes
        ------
                    sess:  Tensorflow session

                 train_X:  Training features, shape: [Nx, m_train]
                 train_y:  Training labels, shape: [1, m_train]
                  test_X:  Testing features, shape: [Nx, m_test]
                  test_y:  Testing labels, shape: [1, m_test]

                      nx:  Number of features
                      ny:  Number of outputs
                       m:  Number of training examples
              train_size:  % of data used for training

                      lr:  Learning rate of logistic regression.  Default value = 0.003
          minibatch_size:  Size of mini-batch for mini-batch gradient descent
      total_batch_number:  Total possible amount of batches for the current data set
                  epochs:  Number of times to iterate through the data
                   lambd:  Regularization parameter

                       X: Placeholder for feeding minibatches of features
                       y: Placeholder for feeding minibatches of labels

                       W: Weights of the logistic regression
                       b: Biases of the logistic regression

    Methods
        -------
                   train:  Train the model using the current structure of the computational graph
                    test:  Output is the predicted outputs, rounded up at a threshold of 50%
            test_percent:  Output the %, rather than a 0 or 1
        model_evaluation:  Outputs the accuracy, precision, and recall
      weights_and_biases:  Outputs the weights and biases of the current model
    """

    def __init__(self, sess, train_X, train_y, test_X, test_y, lr=0.003, minibatch_size=64, train_size=0.9, epochs=5,
                 lambd=0.001):

        """
        Input data must be of shape: [Nx, m] and the labels must be in the first column
        """
        # Tensorflow session
        self.sess = sess

        # Data
        self.train_X = train_X
        self.train_y = train_y

        self.test_X = test_X
        self.test_y = test_y

        self.nx = train_X.shape[0]
        self.ny = 1
        self.m = train_X.shape[1]
        self.train_size = train_size

        # Machine learning parameters
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.total_batch_number = int((self.m / self.minibatch_size) * self.train_size)
        self.epochs = epochs
        self.lambd = lambd

        # Tensorflow variables
        self.X = tf.placeholder(dtype=tf.float32, shape=[self.nx, None])
        self.y = tf.placeholder(dtype=tf.float32, shape=[self.ny, None])

        self.W = tf.get_variable('weights', shape=[self.ny, self.nx],
                                 initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable('biases', shape=[self.ny, 1])

        self.z = tf.matmul(self.W, self.X) + self.b

        # Loss function
        self.regularizer = tf.nn.l2_loss(self.W)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.z, labels=self.y)
                                   + self.lambd * self.regularizer)

        # Adam optimization
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        # Prediction
        self.pred = tf.round(tf.sigmoid(self.z))

        # Accuracies
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.y), tf.float32))
        self.precision, self.prec_ops = tf.metrics.precision(labels=self.y, predictions=self.pred)
        self.recall, self.recall_ops = tf.metrics.recall(labels=self.y, predictions=self.pred)

        self.loss_history = []

        self.init = tf.global_variables_initializer()
        self.init_l = tf.local_variables_initializer()

        self.saver = tf.train.Saver()

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
        return self.sess.run([self.accuracy, self.prec_ops, self.recall_ops], feed_dict={self.X: features,
                                                                                         self.y: labels})

    def weights_and_biases(self):
        return self.sess.run([self.W, self.b])

    def __str__(self):
        return "Logistic Regression using {} features.".format(self.nx)

    def __repr__(self):
        return "LogisticRegression()"


# Min max normalization
class MinMaxNormalization:

    def __init__(self, data):
        self.col_min = np.min(data, axis=1).reshape(data.shape[0], 1)
        self.col_max = np.max(data, axis=1).reshape(data.shape[0], 1)
        self.denominator = abs(self.col_max - self.col_min)

        # Fix divide by zero, replace value with 1 because these usually happen for boolean columns
        for index, value in enumerate(self.denominator):
            if value[0] == 0:
                self.denominator[index] = 1

    def __call__(self, data):
        return np.divide((data - self.col_min), self.denominator)


class DeviationVariables:

    """
    data: Comes in with shape [Nx, m]
    """

    def __init__(self, data):
        self.median = np.median(data, axis=1)
        self.mad = []

        # Populate the MAD values
        for i in range(data.shape[0]):
            med_abs_dev = np.median([np.abs(y - self.median[i]) for y in data[i, :]])
            self.mad.append(med_abs_dev)

        # Make sure the MAD values are not 0
        for i, value in enumerate(self.mad):
            if self.mad[i] == 0:
                self.mad[i] = 0.1

    def __call__(self, data, features, threshold):
        # Input data should be [Nx, 1]
        data = abs(data - self.median)

        abnormal_features = []
        for index, value in enumerate(data):
            if abs(value) > self.mad[index] * threshold:
                abnormal_features.append(index)

        features = np.array(features)

        return features[abnormal_features]

    def plot(self, data, index, threshold):
        # Input data should be [Nx, m]
        plt.plot(data[index, :])
        plt.axhline(self.median[index], color='green')
        plt.axhline(self.median[index] - self.mad[index] * threshold, color='red')
        plt.axhline(self.median[index] + self.mad[index] * threshold, color='red')

        plt.show()


def save(item, path):
    pickle_out = open(path, 'wb')
    pickle.dump(item, pickle_out)
    pickle_out.close()


def load(path):
    pickle_in = open(path, 'rb')
    item = pickle.load(pickle_in)
    return item


def random_seed(seed=8):
    np.random.seed(seed)
    tf.set_random_seed(seed)


def important_features(weights, feature_list, threshold):
    """
    Returns all the important features above the threshold

    important_features(weights, feature_names, 2)
    """

    index = np.linspace(0, weights.shape[0] - 1, weights.shape[0])
    index = [int(i) for (i, j) in zip(index, weights) if abs(j) > threshold]
    feature_list = [feature_list[i] for i in index]
    weights_list = [weights[i][0] for i in index]

    return pd.DataFrame([feature_list, weights_list], index=["Features", "Weights"]).T


def simulation(data_path, model_path, norm_path, label_name, train_size, testing):

    # Loading data
    raw_data = pd.read_csv(data_path)

    cols = [col for col in raw_data.columns if (col[-6:] == "_label" and col != label_name)]
    raw_data.drop(columns=cols, axis=1, inplace=True)

    # Get feature headers
    feature_names = list(raw_data)

    # Delete Unnamed: 0 and label column
    del feature_names[0]

    # Turn Pandas dataframe into NumPy Array
    raw_data = raw_data.values
    print("Raw data has {} features with {} examples.".format(raw_data.shape[1], raw_data.shape[0]))

    # Shuffle and transpose the data
    if testing:
        pass
    else:
        np.random.shuffle(raw_data)
    raw_data = raw_data.T

    # Data partition into features and labels
    labels = raw_data[0, :]
    features = raw_data[1:, :]

    # Train / test split
    train_index = int(features.shape[1] * train_size)

    train_X = features[:, 0:train_index].reshape(features.shape[0], train_index)
    train_y = labels[0:train_index].reshape(1, train_index)

    assert(train_y.shape[1] == train_X.shape[1])

    test_X = features[:, train_index:]
    test_y = labels[train_index:].reshape(1, features.shape[1] - train_index)

    assert(test_y.shape[1] == test_X.shape[1])

    # Restore the parameters from normalizer
    if testing:
        min_max_normalization = load(norm_path)

    else:
        min_max_normalization = MinMaxNormalization(train_X)

    train_X = min_max_normalization(train_X)
    test_X = min_max_normalization(test_X)

    assert(not np.isnan(train_X).any())
    assert(not np.isnan(test_X).any())

    with tf.Session() as sess:

        # Initialize logistic regression object
        log_reg = LogisticRegression(sess, train_X, train_y, test_X, test_y, minibatch_size=16, epochs=25)

        # If testing the model, restore the tensorflow graph
        if testing:
            log_reg.saver.restore(sess, model_path)
            sess.run(log_reg.init_l)

            test_accuracy, test_precision, test_recall = log_reg.model_evaluation(log_reg.test_X, log_reg.test_y)
            print("Test acc: {:5f} | Test precision: {:5f} | Test recall: {:5f}".format(test_accuracy,
                                                                                        test_precision,
                                                                                        test_recall))

        else:
            sess.run(log_reg.init)
            sess.run(log_reg.init_l)

            for epoch in range(log_reg.epochs):

                for i in range(log_reg.total_batch_number):
                    batch_index = i * log_reg.minibatch_size
                    batch_X = log_reg.train_X[:, batch_index:(batch_index + log_reg.minibatch_size)]
                    batch_y = log_reg.train_y[:, batch_index:(batch_index + log_reg.minibatch_size)]

                    current_loss = log_reg.train(batch_X, batch_y)

                    if i % int(log_reg.total_batch_number / 3) == 0:
                        train_accuracy, _, _ = log_reg.model_evaluation(log_reg.train_X, log_reg.train_y)
                        test_accuracy, _, _ = log_reg.model_evaluation(log_reg.test_X, log_reg.test_y)
                        print("Epoch: {} | loss: {:5f} | train acc: {:5f} | test acc: {:5f}".format(epoch + 1,
                                                                                                    current_loss,
                                                                                                    train_accuracy,
                                                                                                    test_accuracy))

            train_accuracy, _, _ = log_reg.model_evaluation(log_reg.train_X, log_reg.train_y)
            test_accuracy, _, _ = log_reg.model_evaluation(log_reg.test_X, log_reg.test_y)

            print("Final results: train acc: {:5f} | test acc: {:5f}".format(train_accuracy, test_accuracy))

        pred_test = log_reg.test(log_reg.test_X, log_reg.test_y)
        pred_train = log_reg.test(log_reg.train_X, log_reg.train_y)
        weights, biases = log_reg.weights_and_biases()

        if testing:
            pass
        else:
            save(min_max_normalization, norm_path)
            log_reg.saver.save(sess, model_path)

        return pred_train, train_y, pred_test, test_y, weights, biases


if __name__ == "__main__":

    random_seed(8)

    # Paths for MacOS Mojave
    # path = '/Users/ruinian/Documents/Logistic_Reg_TF/data/64_data_sampled.csv'    # Location of data
    # model_path = '/Users/ruinian/Documents/Logistic_Reg_TF/checkpoints/test.ckpt'
    # norm_path = '/Users/ruinian/Documents/Logistic_Reg_TF/pickles/norm.pickle'

    # Paths for Ubuntu 18.04
    path = '/home/rui/Documents/Willowglen/data/test_data1.csv'    # Location of data
    model_path = '/home/rui/Documents/Willowglen/logistic_regression_tf/checkpoints/test.ckpt'
    norm_path = '/home/rui/Documents/Willowglen/logistic_regression_tf/pickles/norm.pickle'

    # Beware, for statistical event detection, it includes _630, but for limit, it may not
    label_name = '175642862_630_label'
    testing = False    # Are you training or testing

    if testing:
        train_size = 0
    else:
        train_size = 0.9  # Train / test split size

    Pred_train, Train_y, Pred_test, Test_y, Weights, Biases = simulation(path, model_path, norm_path, label_name,
                                                                         train_size, testing)

    # Plots train data from 0 to 400
    plots(Pred_test, Test_y, 0, 200)
