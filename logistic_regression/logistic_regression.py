"""
Logistic Regression Patch 1.3

Patch notes:  Added tensorboard, saver
              Different normalization methods
              Currently, min-max is the best
              Added Precision and Recall metrics

Date of last edit: Dec-7th-2018
Rui Nian

Current issues: Output size is hard coded
                Cannot run code purely to test the accuracy of algorithm
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

"""
Parsing section, to define parameters to be ran in the code
"""

# Initiate the parser
parser = argparse.ArgumentParser(description="Inputs to the logistic regression")

# Arguments
parser.add_argument("--data", help="Data to be loaded into the model", default='data/labeled_data.csv')
parser.add_argument("--train_size", help="% of whole data set used for training", default=0.999)
parser.add_argument('--lr', help="learning rate for the logistic regression", default=0.003)
parser.add_argument("--minibatch_size", help="mini batch size for mini batch gradient descent", default=91)
parser.add_argument("--epochs", help="Number of times data should be recycled through", default=5000)
parser.add_argument("--tensorboard_path", help="Location of saved tensorboard information", default="./tensorboard")
parser.add_argument("--model_path", help="Location of saved tensorflow graph", default='checkpoints/time1.ckpt')
parser.add_argument("--save_graph", help="Save the current tensorflow computational graph", default=False)
parser.add_argument("--restore_graph", help="Reload model parameters from saved location", default=True)

# Test Model
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
def min_max_normalization(data):

    col_max = np.max(data, axis=1).reshape(data.shape[0], 1)
    col_min = np.min(data, axis=1).reshape(data.shape[0], 1)

    denominator = abs(col_max - col_min)

    # Fix divide by zero, replace value with 1 because these usually happen for boolean columns
    for index, value in enumerate(denominator):
        if value[0] == 0:
            denominator[index] = 1

    return np.divide((data - col_min), denominator)


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

# Delete Unnamed: 0 and label column
# del feature_names[0]
del feature_names[0]

# Turn Pandas dataframe into NumPy Array
raw_data = raw_data.values
print("Raw data has {} features with {} examples.".format(raw_data.shape[1], raw_data.shape[0]))

# Delete the index column given by Pandas
# raw_data = np.delete(raw_data, [0], axis=1)
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

# Neural network parameters
input_size = train_X.shape[0]
output_size = 1
learning_rate = Args['lr']
mini_batch_size = Args['minibatch_size']
total_batch_number = int(train_X.shape[1] / mini_batch_size)
epochs = Args['epochs']

train_X = min_max_normalization(train_X)
test_X = min_max_normalization(test_X)

# Test cases
assert(np.isnan(train_X).any() == False)
assert(np.isnan(test_X).any() == False)

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
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Prediction
pred = tf.round(tf.sigmoid(z))

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

        Precision, Recall = sess.run([prec_op, recall_op], feed_dict={x: test_X, y: test_y})
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
                loss_history.append(current_loss)

                if i % 10 == 0:

                    # Add to summary writer
                    summary_writer.add_summary(summary, i)

                    print("Epoch: {} | loss: {:5f} | train acc: {:5f} | test acc: {:5f}".format(epoch + 1,
                                                                                                current_loss,
                                                                                                train_accuracy,
                                                                                                test_accuracy))

        if Args['save_graph']:
            save_path = saver.save(sess, Args["model_path"])
            print("Model was saved in {}".format(save_path))

        # Output weights and biases
        weights = sess.run(W).reshape(train_X.shape[0], 1)
        biases = sess.run(b)


def plots(percent, real_value, start, end):

    """
    Plots the real vs percentage
    """
    plt.subplot(2, 1, 1)
    plt.xlabel("Time")
    plt.ylabel("Percent below Threshold, %")
    plt.step(np.linspace(0, end - start, end - start), percent.reshape(percent.shape[1], 1)[start:end] * 100)

    plt.axhline(y=70, c='r', linestyle='--')

    plt.subplot(2, 1, 2)
    plt.xlabel("Time")
    plt.ylabel("Plant Data")
    plt.step(np.linspace(0, end - start, end - start), real_value.reshape(real_value.shape[1], 1)[start:end], c='r')

    plt.show()


def important_features(weights, feature_list, threshold):
    """
    Returns all the important features above the threshold
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


def suncor_early_pred(predictions, labels, early_window, num_of_events, threshold=0):

    """
            predictions:  Predictions made by logistic regression
            actual_data:  Actual Suncor pipeline data with labels on first column
           early_window:  How big the window is for early detection
          num_of_events:  Total events in the data set
              threshold:  Threshold for rounding a number up
    """
    # Convert to boolean
    predictions = np.round(predictions + 0.5 - threshold)

    early_detected = 0
    detected = 0

    for i, event in enumerate(labels):
        # If an event occurs
        if event == 1 and labels[i - 1] != 1:
            # If predictions detected least 1 time step in advance, the event is considered as "early detected"
            if 1 in predictions[i - early_window:i]:
                early_detected += 1
            # If predictions detected the event up to event, the event is considered as "detected"
            if 1 in predictions[i - early_window:i + 1]:
                detected += 1

    recall_overall = detected / num_of_events
    recall_early = early_detected / num_of_events

    error = 0

    for i, event in enumerate(predictions):
        # If the prediction is positive
        if event == 1 and 1 not in predictions[i - 20:i]:
            # If there is an actual event, nothing happens
            if 1 in labels[i:i + early_window]:
                pass
            # If there is no event, add to error
            else:
                error += 1

    precision = detected / (error + detected)

    return recall_early, recall_overall, precision
