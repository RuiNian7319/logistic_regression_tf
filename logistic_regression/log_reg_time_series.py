"""
Logistic Regression Patch 1.3

Patch notes:  Added tensorboard, saver
              Different normalization methods
              Currently, min-max is the best
              Added Precision and Recall metrics
              Fixed MinMaxNormalization, now test data is normalized with respect to train
              Added DeviationVariable class to identify which variables are outside their expected ranges
              Added prec and recall during training

Date of last edit: Dec-27th-2018
Rui Nian

Current issues: -
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

import gc
import argparse

import os
import pickle

import warnings

from suncor_ts_tester import suncor_early_pred

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# Ubuntu 18.04
# path = '/home/rui/Documents/logistic_regression_tf/'

# MacOS
path = '/Users/ruinian/Documents/Logistic_Reg_TF/'

"""
Parsing section, to define parameters to be ran in the code
"""

# Initiate the parser
parser = argparse.ArgumentParser(description="Inputs to the logistic regression")

# Arguments
parser.add_argument("--data", help="Data to be loaded into the model", default=path + 'data/labeled_data.csv')
parser.add_argument("--normalization", help="folder with normalization info", default=path + 'pickles/norm.pickle')
parser.add_argument("--train_size", help="% of whole data set used for training", default=0.9999)
parser.add_argument('--lr', help="learning rate for the logistic regression", default=0.003)
parser.add_argument('--lambd', help="regularization term", default=0.0005)
parser.add_argument("--minibatch_size", help="mini batch size for mini batch gradient descent", default=512)
parser.add_argument("--epochs", help="Number of times data should be recycled through", default=50)
parser.add_argument("--threshold", help="Threshold for positive classification, norm=0.5", default=0.5)
parser.add_argument("--tensorboard_path", help="Location of saved tensorboards", default=path + "./tensorboard")
parser.add_argument("--model_path", help="Location of saved tensorflow models", default=path + 'checkpoints/10time.ckpt')
parser.add_argument("--save_graph", help="Save the current tensorflow computational graph", default=False)

# Test Model
parser.add_argument("--restore_graph", help="Reload model parameters from saved location", default=True)
parser.add_argument("--test", help="put as true if you want to test the current model", default=True)

# Makes a dictionary of parsed args
Args = vars(parser.parse_args())

"""
Logistic Regression
"""

# Seed for reproducability
seed = 18
np.random.seed(seed)
tf.set_random_seed(seed)


# Min max normalization
class MinMaxNormalization:

    """
    data: Comes in with shape [Nx, m]
    """

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


def save_info(obj, path):
    # save_info(min_max_normalization, path + 'pickles/norm.pickle')
    pickle_out = open(path, "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()


# Define Normalization
def normalization(data):
    col_mean = np.mean(data, axis=1).reshape(data.shape[0], 1)
    col_std = np.std(data, axis=1).reshape(data.shape[0], 1)

    # Fix divide by 0 since sometimes, standard deviation can be zero
    for index, value in enumerate(col_std):
        if value[0] == 0:
            col_std[index] = 1

    return np.divide((data - col_mean), col_std)


# Modified Normalization robust to outliers
def mod_normalization(data):
    col_median = np.median(data, axis=1).reshape(data.shape[0], 1)
    mad = np.median(abs(data - col_median), axis=1).reshape(data.shape[0], 1)

    # Fix divide by 0 for when MAD = 0
    for index, value in enumerate(mad):
        if value[0] == 0:
            mad[index] = 1

    return np.divide(0.6745 * (data - col_median), mad)


# Loading data
raw_data = pd.read_csv(Args['data'])

# Get feature headers
feature_names = list(raw_data)

# Delete label column
del feature_names[0]

# Turn Pandas dataframe into NumPy Array
raw_data = raw_data.values
print("Raw data has {} features with {} examples.".format(raw_data.shape[1], raw_data.shape[0]))

# Delete the index column given by Pandas
# np.random.shuffle(raw_data)
raw_data = raw_data.T

# Data partitation into features and labels
features = raw_data[1:, :]
labels = raw_data[0, :]

# Train / test split
train_size = Args['train_size']
train_values = int(features.shape[1] * train_size)

train_X = features[:, 0:train_values]
train_y = labels[0:train_values].reshape(1, train_X.shape[1])

test_X = features[:, train_values:]
test_y = labels[train_values:].reshape(1, test_X.shape[1])

"""
For feeding t and t - 1
"""

train_X = np.concatenate([train_X[:, 0:-1], train_X[:, 1:]], axis=0)
train_y = train_y[:, :-1]

test_X = np.concatenate([test_X[:, 0:-1], test_X[:, 1:]], axis=0)
test_y = test_y[:, :-1]

# Neural network parameters
input_size = train_X.shape[0]
output_size = 1
learning_rate = Args['lr']
mini_batch_size = Args['minibatch_size']
total_batch_number = int(train_X.shape[1] / mini_batch_size)
epochs = Args['epochs']

# min_max_normalization = MinMaxNormalization(train_X)
pickle_in = open(Args['normalization'], 'rb')
min_max_normalization = pickle.load(pickle_in)
train_X = min_max_normalization(train_X)
test_X = min_max_normalization(test_X)

# Test cases
assert(not np.isnan(train_X).any())
assert(not np.isnan(test_X).any())

# deviation_vars = DeviationVariables(train_X)

# Model placeholders
with tf.name_scope("Inputs"):
    x = tf.placeholder(dtype=tf.float32, shape=[input_size, None])
    y = tf.placeholder(dtype=tf.float32, shape=[output_size, None])

# Model variables
with tf.name_scope("Model"):
    with tf.variable_scope("Weights"):
        W = tf.get_variable('Weights', shape=[1, input_size], initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope("Biases"):
        b = tf.get_variable('Biases', shape=[output_size, 1], initializer=tf.contrib.layers.xavier_initializer())

    tf.summary.histogram("Weights", W)
    tf.summary.histogram("Bias", b)

# Model
z = tf.matmul(W, x) + b

# Cross entropy with logits, assumes inputs are logits before cross entropy
regularizer = tf.nn.l2_loss(W)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y) + Args['lambd'] * regularizer)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Prediction
pred = tf.round(tf.sigmoid(z) - Args['threshold'] + 0.5)

# Evaluate accuracy.  Has to be cast to tf.float32 so they are numbers rather than True/False.
# Then reduce mean to divide correct by m examples
correct = tf.cast(tf.equal(pred, y), dtype=tf.float32)
accuracy = tf.reduce_mean(correct)
precision, prec_op = tf.metrics.precision(labels=y, predictions=pred)
recall, recall_op = tf.metrics.recall(labels=y, predictions=pred)

loss_history = []

# Initialize all variables
init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

# Tensorflow graph saver
saver = tf.train.Saver()

if Args['test']:
    with tf.Session() as sess:

        saver.restore(sess, Args["model_path"])
        sess.run(init_l)

        train_accuracy = sess.run(accuracy, feed_dict={x: train_X, y: train_y})
        test_accuracy = sess.run(accuracy, feed_dict={x: test_X, y: test_y})

        pred = tf.sigmoid(z)

        Predictions = sess.run(pred, feed_dict={x: test_X, y: test_y})
        Predictions_train = sess.run(pred, feed_dict={x: train_X, y: train_y})
        print("Training data set: {:5f} | Test data set: {:5f}".format(train_accuracy, test_accuracy))

        Precision, Recall = sess.run([prec_op, recall_op], feed_dict={x: train_X, y: train_y})
        print("The precision is: {:5f} | The recall is: {:5f}".format(Precision, Recall))

        # Output weights and biases
        weights = sess.run(W).reshape(train_X.shape[0], 1)
        biases = sess.run(b)

else:
    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter(Args['tensorboard_path'], graph=sess.graph)
        merge = tf.summary.merge_all()

        if Args['restore_graph']:
            # Restore tensorflow graph
            saver.restore(sess, Args['model_path'])
        else:
            sess.run(init)
            sess.run(init_l)

        for epoch in range(epochs):

            for i in range(total_batch_number):

                # Generate mini_batch indexing
                batch_index = i * mini_batch_size
                minibatch_train_X = train_X[:, batch_index:(batch_index + mini_batch_size)]
                minibatch_train_y = train_y[:, batch_index:(batch_index + mini_batch_size)]

                _, summary = sess.run([optimizer, merge], feed_dict={x: minibatch_train_X, y: minibatch_train_y})
                current_loss = sess.run(loss, feed_dict={x: minibatch_train_X, y: minibatch_train_y})

                train_accuracy, train_predictions = sess.run([accuracy, pred], feed_dict={x: train_X, y: train_y})
                test_accuracy, test_predictions = sess.run([accuracy, pred], feed_dict={x: test_X, y: test_y})

                train_prec, train_recall = sess.run([prec_op, recall_op], feed_dict={x: train_X,
                                                                                     y: train_y})
                loss_history.append(current_loss)

                if i % 10 == 0:

                    # Add to summary writer
                    summary_writer.add_summary(summary, i)

                    print("Epoch: {} | loss: {:5f} | train acc: {:5f} | test acc: {:5f} | prec: {:5f} | recall: {:5f}".
                          format(epoch + 1, current_loss, train_accuracy, test_accuracy, train_prec, train_recall))

        if Args['save_graph']:
            save_path = saver.save(sess, Args["model_path"])
            print("Model was saved in {}".format(save_path))

        # Output weights and biases
        weights = sess.run(W).reshape(train_X.shape[0], 1)
        biases = sess.run(b)


def plots(prediction, real_value, start, end):

    """
    Plots the real plant trajectory vs the predicted

    Inputs
         -----
         perdiction:  Prediction from machine learning model
         real_value:  Real value from the data set
              start:  Start index of the plot
                end:  End index of the plot
    """
    plt.subplot(2, 1, 1)
    plt.xlabel("Time")
    plt.ylabel("Percent below Threshold, %")
    plt.step(np.linspace(0, end - start, end - start), prediction.reshape(prediction.shape[1], 1)[start:end] * 100)

    plt.axhline(y=Args['threshold'] * 100, c='r', linestyle='--')

    plt.subplot(2, 1, 2)
    plt.xlabel("Time")
    plt.ylabel("Plant Data")
    plt.step(np.linspace(0, end - start, end - start), real_value.reshape(real_value.shape[1], 1)[start:end], c='r')

    plt.show()


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


def dataset_creator(data, columns, path):
    """
    data:  Original data, Pandas data frame structure
    columns:  Columns to be kept within the original data

    Assumes raw data's 1st column is 'Unnamed: 0', 2nd column is the label column, and there is only one label column
    """
    cols = [list(data)[1]] + columns
    data = data[cols]

    # Make sure the dimensions are correct after new dataset is created
    assert(len(columns) + 1 == data.shape[1])

    data.to_csv(path, index=False)
    return data


def k_folds(data, fold_number, train_size=0.8):

    """
    Inputs
               data:  Data with shape: [m, Nx]
        fold_number:  The current k-fold you are on.
         train_size:  Size of training data.  Test size is 1 - train_size
    Returns
        train_X, train_y, test_X, test_y
    """

    examples = data.shape[0]

    train_shape = int(train_size * examples)
    test_shape = int(examples - train_shape)

    index_end = test_shape * fold_number
    index_start = index_end - test_shape

    mask = np.linspace(index_start, index_end, index_end - index_start + 1)
    train_data = np.delete(data, mask, axis=0)
    test_data = data[index_start:index_end, :]

    train_X = train_data[:, 1:]
    train_y = train_data[:, 0]

    test_X = test_data[:, 1:]
    test_y = test_data[:, 0]

    return train_X, test_X, train_y, test_y
