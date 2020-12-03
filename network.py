# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 01:18:04 2020
Main script for program.
FILL IN LATER

@author: mlazo
"""

# Library imports
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import matplotlib.pyplot as plt

# Local imports


# Network parameters
# generation
train_set   = 100000    # size of synthetic training set
val_set     = 512       # size of synthetic validation set
window      = 128       # size of data window, number of points in one data set; 128 is default
# training
BATCH       = 256       # size of training batches
BUFFER      = 10000     # size of shuffling buffer

EPOCHS      = 100
EVAL_INT    = 200       # keras evaluation interval

num         = 5000      # number of predictions to make

def normalize(a):
    '''
    Normalize an array of values to [0, 1].
    Used to prepare data for use with Tensorflow.

    Parameters
    ----------
    a : array-like
        Array of values to be normalized.

    Returns
    -------
    norm : double array
        Normalized array.
    a : double array
        Input array, unnormalized.

    '''
    ra = np.nanmax(a) - np.nanmin(a)
    norm = (a[:] - np.nanmin(a)) / ra
    
    return norm

