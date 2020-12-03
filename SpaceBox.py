# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:16:14 2020
Generate synthetic Langmuir probe traces.

@author: mlazo
"""

import numpy as np
import langmuir as lang
from constants import *

class SpaceBox():
    def __init__(self, T_min, T_max, Is_min, Is_max, mu, A_p, window, period, traces):
        '''
        Class constructor.

        Parameters
        ----------
        T_min : double, eV
            Lower bound for temperature.
        T_max : double, eV
            Upper bound for temperature.
        Is_min : double, A
            Lower bound for ion saturation current.
        Is_max : double, A
            Upper bound for ion saturation current.
        mu : int
            Plasma ion mass number.
        A_p : double, m
            Collection area of Langmuir probe.
        window : int
            Number of values within one trace.
        period : double, s
            Time for one period of voltage sweep.
        traces : int
            Number of traces to produce.
        
        Returns
        -------
        None.

        '''
        self.T_min  = T_min
        self.T_max  = T_max
        self.Is_min = Is_min
        self.Is_max = Is_max
        self.m_i    = mu * M_P # Mass of plasma ions based on atomic number, in kg
        self.A_p    = A_p
        self.window = window
        self.period = period
        self.traces = traces
        return
    
    def Te_Is_picker(self, T_min, T_max, Is_min, Is_max):
        '''
        Generates random electron temperature and ion saturation current
        within the specified ranges.
        
        Parameters
        ----------
        T_min : double, eV
            Lower bound for temperature.
        T_max : double, eV
            Upper bound for temperature.
        Is_min : double, A
            Lower bound for ion saturation current.
        Is_max : double, A
            Upper bound for ion saturation current.
        
        Returns
        -------
        T : double, eV
            Selected electron temperature.
        I : double, A
            Selected ion saturation current.
            
        '''
        # T = np.zeros((self.traces))
        # I = np.zeros((self.traces))
        
        # for i in range(self.traces):
        #     T[i] = np.random.uniform(T_min, T_max)
        #     I[i] = np.random.uniform(Is_min, Is_max)
        
        T = np.random.uniform(T_min, T_max, self.traces)
        I = np.random.uniform(Is_min, Is_max, self.traces)
        
        return T, I
    
    def V_raw(self, T_e):
        '''
        Generates a sweep voltage signal.
        
        Parameters
        ----------
        T_e : double, eV
            Plasma electron temperature.

        Returns
        -------
        tt : float[], s
            Evenly-spaced times on [0, self.window].
        vv : double[], V
            Voltage signals generated as a sawtooth wave, or repeated linear
            sweep from -5 V to +10 V.
        '''
        T = self.period # 3 s(?), 15.02 s between traces
        Vf = self.V_f(T_e)
        
        vv = np.zeros((self.traces, self.window))
        tt = np.zeroslike(vv)
        
        A = (np.random.rand() * 10) * T_e
        phi = (np.random.rand() * 10) * T_e
        delta = T_e / (0.1 + (np.random.rand() * 10))
        
        tt = np.linspace(0, T, self.window)
        vv = (2*A/np.pi) * np.arctan( np.cot((tt*np.pi/T) + phi) ) - np.abs(Vf) + delta # sawtooth sweep
        
        return tt, vv
    
    def I_raw(self, vv, T_e, I_is):
        T_joules = Q * T_e
        n_e = (-I_is*np.exp(0.5)*np.sqrt(self.m_i/T_joules)) / (Q*self.A_p)
        
        Ie = lang.Je * self.A_p
        Ii = lang.Ji * self.A_p
        I = Ie - Ii
        
        return I
    
    def V_f(self, T_e):
        '''
        Calculate floating potential of the current probe trace.
        
        Parameters
        ----------
        T_e : double, eV
            Plasma electron temperature.
            
        Returns
        -------
        V_f : double, V
            Plasma floating potential, found by Hutchinson Plasma Diagnostics
            eq. 3.2.35.
        '''
        return (T_e/2) * (np.log( (2*np.pi)*(M_E/m_i) ) - 1)
    
    def __call__(self):
        T_e, I_is = self.Te_Is_picker(self.T_min, self.T_max, self.Is_min, self.Is_max)
        
        V_f = self.V_f(T_e)
        
        tt, vol = self.V_noisy(T_e)
        cur = self.I_noisy(vol, T_e, I_is, vol)
        
        for i in range(self.traces):
            vol[i,:] = vol[i,:] + np.abs(V_f[i])
        
        return tt, vol, cur, T_e, I_is
    
import matplotlib.pyplot as plt;

def main():
    print("____")
    
    T_min = 0.2191 # eV
    T_max = 0.2191 # eV
    Is_min = -2.15E-7 # A
    Is_max = -2.15E-7 # A
    
    box = SpaceBox(T_min,T_max,Is_min,Is_max,mu_o,A_P,1000,3,2)
    tt, vol, cur, T_e, I_is = box()
    
    n = 1
    fig, ax = plt.subplots()
    ax.plot(vol[n,:], cur[n,:], color='r')
    plt.ylabel('Current (A)')
    plt.xlabel('Bias (V)')
    
    fig2, ax2 = plt.subplots()
    ax2.plot(tt[n, :], vol[n, :])
    plt.ylabel('Bias (V)')
    plt.xlabel('Time (s)')
    
    fig3, ax3 = plt.subplots()
    ax3.plot(tt[n, :], cur[n, :])
    plt.ylabel('Current (A)')
    plt.xlabel('Time (s)')
    
    plt.show()
