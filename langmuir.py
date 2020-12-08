# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:29:26 2020
Function defeinitions for theoretical model of Langmuir probe data.

@author: mlazo
"""

import numpy as np
import matplotlib.pyplot as plt
from constants import *

def Jsat(n, Te):
    return Q * n * np.sqrt(Te * Q / m_i)

def A_sheath(t, m, b):
    return m*t + b

def Ii(tt, n, Te, m, b):
    
    J_s = Jsat(n, Te)
    A_s = A_sheath(tt, m, b)
    return J_s * A_s

def Je(tt, vv, Te):    
    cur = (0.5*np.sqrt(2*m_i/np.pi*M_E)*np.exp(vv/Te))
    # cur = (0.5*np.exp(vv/Te))
    return cur
    # return np.piecewise(tt, [tt<=V_ind, tt>V_ind],
    #                     [0,
    #                      lambda vv: (0.5*np.sqrt(2*m_i/np.pi*M_E)*np.exp(vv/Te))
    #                      ])

# vv = np.linspace(-5, 10, 300)
# A = 7.7515E-3
# n = 3.01E+11
# T = 0.2191
# Vf = 0.1
# m = 8.898E-8

# A_s = A_sheath(vv, m, Vf)
# #print(A_s)

# Isat = Jsat(n,T) * A
# Ie = Je(vv, n, T, Vf, A, m) * A
# Ii = Ji(vv, n, T, Vf, m) * A_s
# print(Ii)
# curr = Ie - Ii

# plt.plot(vv, curr, label='total')
# plt.plot(vv, Ie, label='electron')
# plt.plot(vv, Ii, label='ion')
# plt.xlabel('Bias (V)')
# plt.ylabel('Current (A)')
# plt.legend()
