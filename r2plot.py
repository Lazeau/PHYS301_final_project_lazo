# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 08:25:48 2020

@author: mlazo
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def RSquared(y, y_p):
    '''
    Calculated R-Squared value of predictions
    Parameters
    ----------
    y : tf eager tensor
        expected actual value.
    y_p : tf eager tensor
        predicted value.
    Returns
    -------
    RSquared value.
    '''
    total_error = tf.reduce_sum(tf.square(y, tf.reduce_mean(y_p)))
    unexplained_error = tf.reduce_sum(tf.square(y - y_p))
    R_squared = 1. - unexplained_error / total_error
    
    return R_squared.numpy()

def plotResult(fig, ax, y, y_p, label, title, fontsize = 14, res = 25):
    xedges = np.linspace(0, 1, res) # histogam edges. since outputs are normalized
    yedges = np.linspace(0, 1, res)
    H, xedges, yedges = np.histogram2d(y, y_p, bins = [xedges, yedges])
    H = H.T  # Let each row list bins with common y range.
    ax.imshow(H, interpolation='nearest', origin='low',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
    # Calculating R-squared
    R_squared = RSquared(y, y_p)
    R2_string = '$R^2 = $'+ str(R_squared)[:5]
    ax.set_ylabel('Predicted ' + label, fontsize = fontsize)
    ax.set_xlabel('Expected ' + label, fontsize = fontsize)
    ax.text(0.05, 0.9, R2_string, color='w', fontsize = fontsize)
    ax.set_title(title, fontsize = fontsize + 2)
