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
        # Experimentally-determined slope of ion current due to sheath expansion
        # self.beta_m = np.random.uniform(5.00E-10, 1.25E-7, traces)
        # self.beta_b = np.random.uniform(-5.80E-6, -4.58E-8, traces)
        self.beta_m = np.tile([1.423E-8],(self.traces,1))
        self.beta_b = np.tile([-1.761E-7],(self.traces,1))
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
        T = np.random.uniform(T_min, T_max, self.traces)
        I = np.random.uniform(Is_min, Is_max, self.traces)
        
        return T, I
    
    def stf1_sweep(self, t):
        v = np.zeros(t.shape[0])
        v[:11]    = 0.2*t[:11] - 5
        v[11:210] = 0.02*t[11:210] - 3.20
        v[210:]   = 0.1*t[210:] - 20
        return v
    
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
        # T = self.period # 3 s(?), 15.02 s between traces
        # Vf = self.V_f(T_e)
        
        # vv = np.zeros((self.traces, self.window))
        # tt = np.zeros((self.traces, self.window))
        
        # A = (np.random.rand() * 10) * T_e
        # phi = (np.random.rand() * 10) * T_e
        # delta = T_e / (0.1 + (np.random.rand() * 10))
        # tt = np.linspace(0, T, self.window)
        #vv = (2*A/np.pi) * np.arctan( np.cot((tt*np.pi/T) + phi) ) - np.abs(Vf) + delta # sawtooth sweep
        
        tt = np.linspace(0, self.window-1, self.window)
        vv = np.tile(self.stf1_sweep(tt), (self.traces, 1))
        
        return tt, vv
    
    def I_raw(self, tt, vv, T_e, I_is):
        n_i = (np.abs(I_is) / (0.5*Q*self.A_p)) * np.sqrt(mi_ev / T_e)
        print(n_i[0])
        I_es = ((n_i*Q*self.A_p) / (4*C)) * np.sqrt((8*T_e) / (np.pi*ME_EV))
        
        I_e = np.zeros((self.traces, self.window))
        I_i = np.zeros((self.traces, self.window))
        for i in range(self.traces):
            I_e[i] = I_es[i] * np.exp((vv[i])/(T_e[i]))
            I_i[i] = self.beta_m[i]*tt + self.beta_b[i]
        I = I_e - I_i
        
        fig4 = plt.figure()
        plt.plot(I_e[0],'c')
        plt.plot(I_i[0],'m')
        print(I_e[0])
        
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
        
        tt, vol = self.V_raw(T_e)
        cur = self.I_raw(tt, vol, T_e, I_is)
        
        return tt, vol, cur, T_e, I_is
    
import matplotlib.pyplot as plt;

def main():
    print("____")
    
    T_min = 0.991 # eV
    T_max = 0.991 # eV
    Is_min = -7.787E-8 # A
    Is_max = -7.787E-8 # A
    
    box = SpaceBox(T_min,T_max,Is_min,Is_max,mu_o,A_P,301,3,2)
    tt, vol, cur, T_e, I_is = box()
    
    n = 0
    fig, ax = plt.subplots()
    ax.plot(vol[n,:], cur[n,:], color='r')
    plt.ylabel('Current (A)')
    plt.xlabel('Bias (V)')
    
    # fig2, ax2 = plt.subplots()
    # ax2.plot(tt, vol[n, :])
    # plt.ylabel('Bias (V)')
    # plt.xlabel('Time (s)')
    
    fig3, ax3 = plt.subplots()
    ax3.plot(tt, cur[n, :])
    plt.ylabel('Current (A)')
    plt.xlabel('Time (s)')
    
    plt.show()

if __name__ == "__main__":
    main()
