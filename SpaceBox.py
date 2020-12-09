# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:16:14 2020
Generate synthetic Langmuir probe traces.

@author: mlazo
"""

import numpy as np
from scipy.special import lambertw

# Local imports
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
        # self.beta_m = np.random.uniform(5.00e-10, 1.25e-7, traces)
        # self.beta_b = np.random.uniform(-5.80e-6, -4.50e-8, traces)
        self.beta_m = np.random.uniform(5.00e-8, 1.25e-7, traces)
        self.beta_b = np.random.uniform(-5.80e-7, -4.50e-8, traces)
        # self.beta_m = np.tile([1.423e-8],(self.traces,1))
        # self.beta_b = np.tile([-1.761e-7],(self.traces,1))
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
        '''
        Generates a standard STF-1 voltage sweep from -5 V to +10 V.
        
        Parameters
        ----------
        t : float[], s
            Time series values or indices.
        
        Returns
        -------
        v : float[], V
            Swept voltage.
        '''
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
        tt = np.linspace(0, self.period, self.window)
        
        return tt, vv
    
    def I_raw(self, tt, vv, T_e, I_is):
        '''
        Generates a sweep of current measurements corresponding to voltage
        time series signals.
        
        Parameters
        ----------
        tt : float[][], s
            Time series indices.
        vv : float[][] V
            Time series voltage values.
        T_e : float[], eV
            Plasma electron temperatures.
        I_is : float[], A
            Plasma ion saturation currents.
        
        Returns
        -------
        I : float[][], A
            Time series current values, calculated from theoretical model.
        V : float[], V
            Plasma floating potentials.
        '''
        T_J = T_e * Q
        n_i = (np.abs(I_is) / (0.5*Q*self.A_p)) * np.sqrt(m_i / T_J)
        I_es = ((n_i*Q*self.A_p) / 4) * np.sqrt((8*T_J) / (np.pi*M_E))
        
        I_e = np.zeros((self.traces, self.window))
        I_i = np.zeros((self.traces, self.window))
        V_f = np.zeros(self.traces)
        for i in range(self.traces):
            I_i[i] = lang.Ii(vv[i], self.beta_m[i], self.beta_b[i])
            V_f[i] = (I_is[i] - self.beta_b[i] / self.beta_m[i])
            I_e[i] = lang.Ie(vv[i], V_f[i], I_es[i], T_e[i])
        I_e[I_e>=CUTOFF] = CUTOFF
        I = I_e - I_i
        
        # fig = plt.figure(figsize=(5,4))
        # plt.plot(vv[0],I_e[0],'c',label='electron')
        # plt.plot(vv[0],I_i[0],'m',label='ion')
        # plt.plot(vv[0],I[0],'r',label='total')
        # plt.plot(V_f[0],I_is[0],'o',color='r',label=r"V_f")
        # plt.xlabel('Bias (V)')
        # plt.ylabel('Current (A)')
        # plt.grid(True)
        # plt.legend()
        # print(I_i[0])
        
        return I, V_f
    
    def __call__(self):
        T_e, I_is = self.Te_Is_picker(self.T_min, self.T_max, self.Is_min, self.Is_max)
        
        tt, vol = self.V_raw(T_e)
        cur, V_f = self.I_raw(tt, vol, T_e, I_is)
        
        return tt, vol, cur, T_e, I_is

# # # # For testing # # # #
import matplotlib.pyplot as plt;

def main():
    print("____")
    
    # For testing trace 51 in file 0
    # T_min = 0.991 # eV
    # T_max = 0.991 # eV
    # Is_min = -7.787E-8 # A
    # Is_max = -7.787E-8 # A
    
    box = SpaceBox(T_min,T_max,Is_min,Is_max,mu_o,A_P,301,3,2)
    tt, vol, cur, T_e, I_is = box()
    
    n = 0
    fig2, ax2 = plt.subplots()
    ax2.plot(vol[n, :], cur[n, :], color='r')
    plt.ylabel('Current (A)')
    plt.xlabel('Bias (V)')
    
    # fig3, ax3 = plt.subplots()
    # ax3.plot(tt, vol[n, :])
    # plt.ylabel('Bias (V)')
    # plt.xlabel('Time (s)')
    
    # fig4, ax4 = plt.subplots()
    # ax4.plot(tt, cur[n, :])
    # plt.ylabel('Current (A)')
    # plt.xlabel('Time (s)')
    
    plt.show()

if __name__ == "__main__":
    main()
