# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:29:26 2020
Function defeinitions for theoretical model of Langmuir probe data.

@author: mlazo
"""

import numpy as np
import matplotlib.pyplot as plt
from constants import *

def Ii(v, m, b):
    return m*v + b

def Ie(v, Vf, Ies, Te):
    vv = v - Vf
    return Ies * np.exp((vv/Te))
