import numpy as np
import tensorflow as tf
import pandas as pd


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

    data.to_csv(path)
    return data


data = pd.read_csv('data/hossein_labeled_data.csv')
