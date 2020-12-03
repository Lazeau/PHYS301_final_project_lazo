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

def Ji(vv, n, Te, Vf, m):
    Js = Jsat(n, Te)
    return Js
    # return np.piecewise(vv, [vv<=Vf, vv>Vf],
    #                     [Js,
    #                       lambda vv: Js - np.exp(-0.5)*(A_sheath(vv,m,Vf)/A_P)
    #                       ])
    # return Js - np.exp(-0.5)*(A_sheath(vv,m,Vf)/A_P)

def A_sheath(vv, m, Vf):
    return m * vv # + 0.1

def Je(vv, n, Te, Vf, Ap, m):
    c = Q * n * np.sqrt(8 * Te * Q / (np.pi * M_E))
    #c = Q * n * np.sqrt(Te*Q/m_i)
    print('constant =', c)
    As = A_sheath(vv, m, Vf)
    
    cur = c * (0.5*np.exp(vv/Te) - As/Ap*np.exp(-0.5))
    #cur = c * (0.5*np.sqrt(2*m_i/np.pi*M_E)*np.exp(vv/Te))
    return np.piecewise(cur, [cur<=CUTOFF, cur>CUTOFF],
                        [lambda cur: cur,
                         CUTOFF
                         ])

vv = np.linspace(-5, 10, 300)
A = 7.7515E-3
n = 3.01E+11
T = 0.2191
Vf = 0.1
m = 8.898E-8

A_s = A_sheath(vv, m, Vf)
#print(A_s)

Isat = Jsat(n,T) * A
Ie = Je(vv, n, T, Vf, A, m) * A
Ii = Ji(vv, n, T, Vf, m) * A_s
print(Ii)
curr = Ie - Ii

plt.plot(vv, curr, label='total')
plt.plot(vv, Ie, label='electron')
plt.plot(vv, Ii, label='ion')
plt.xlabel('Bias (V)')
plt.ylabel('Current (A)')
plt.legend()
