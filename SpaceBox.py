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
        self.beta_m = np.random.uniform(5.00E-10, 1.25E-7, traces)
        self.beta_b = np.random.uniform(-5.80E-6, -4.58E-8, traces)
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
        v[:11] = 0.2*t[:11] - 5
        v[11:210] = 0.02*t[11:210] - 3.20
        v[210:] = 0.1*t[210:] - 20
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
        T = self.period # 3 s(?), 15.02 s between traces
        Vf = self.V_f(T_e)
        
        # vv = np.zeros((self.traces, self.window))
        # tt = np.zeros((self.traces, self.window))
        
        # A = (np.random.rand() * 10) * T_e
        # phi = (np.random.rand() * 10) * T_e
        # delta = T_e / (0.1 + (np.random.rand() * 10))
        # tt = np.linspace(0, T, self.window)
        #vv = (2*A/np.pi) * np.arctan( np.cot((tt*np.pi/T) + phi) ) - np.abs(Vf) + delta # sawtooth sweep
        
        vv = np.zeros((self.traces, self.window))
        tt = np.linspace(0, self.window-1, self.window)
        for i in range(self.traces):
            vv[i] = self.stf1_sweep(tt)
        
        return tt, vv
    
    def A_sheath(self, t, m, b):
        return m*t + b
    
    def I_raw(self, tt, vv, T_e, I_is):
        T_joules = Q * T_e
        # n_e = (-I_is*np.exp(0.5)*np.sqrt(self.m_i/T_joules)) / (Q*self.A_p)
        # cs = np.sqrt(m_i / T_joules)
        # alpha = 0.5 # Magnetization parameter
        # n_i = (np.abs(I_is)*cs) / (Q*alpha*self.A_p)
        
        As = np.zeros((self.traces,self.window))
        n_i = np.zeros(self.traces)
        const = np.zeros(self.traces)
        # const = n_i * Q * self.A_p * np.sqrt(T_e/m_i)
        # const = Q * n_i * np.sqrt(8 * T_joules / (np.pi * M_E))
        Ie = np.zeros((self.traces, self.window))
        Ii = np.zeros((self.traces, self.window))
        I = np.zeros((self.traces, self.window))
        for i in range(self.traces):
            # Ie[i] = const[i] * lang.Je(tt, vv[i], T_e[i])
            # Ii[i] = const[i] * lang.Ii(tt, n_i[i], T_e[i], self.beta_m[i], self.beta_b[i]) / self.A_p
            # I = Ie - Ii
            As[i] = self.A_sheath(tt, self.beta_m[i], self.beta_b[i])
            n_i[i] = (-I_is[i]*np.exp(0.5)*np.sqrt(self.m_i/T_joules[i])) / (Q*self.A_p)
            const[i] = n_i[i] * Q * self.A_p * np.sqrt(T_e[i]/m_i)
            Ii[i] = As[i]
            Ie[i] = const[i] * (0.5 * np.sqrt(2*m_i/np.pi*M_E) * np.exp(vv[i]))
            plt.plot(Ie[i])
            plt.title("ASDF")
        I = Ie - Ii
        # plt.plot(I[0])
        # print(I)
        # plt.title("AAAAAAAA")
        
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
        
        # for i in range(self.traces):
        #     vol[i,:] = vol[i,:] + np.abs(V_f[i])
        
        return tt, vol, cur, T_e, I_is
    
import matplotlib.pyplot as plt;

def main():
    print("____")
    
    T_min = 8.525 # eV
    T_max = 8.525 # eV
    Is_min = -7.188E-8 # A
    Is_max = -7.188E-8 # A
    
    box = SpaceBox(T_min,T_max,Is_min,Is_max,mu_o,A_P,301,3,2)
    tt, vol, cur, T_e, I_is = box()
    
    n = 0
    fig, ax = plt.subplots()
    ax.plot(vol[n,:], cur[n,:], color='r')
    plt.ylabel('Current (A)')
    plt.xlabel('Bias (V)')
    
    fig2, ax2 = plt.subplots()
    # ax2.plot(tt[n, :], vol[n, :])
    ax2.plot(tt, vol[n, :])
    plt.ylabel('Bias (V)')
    plt.xlabel('Time (s)')
    
    fig3, ax3 = plt.subplots()
    # ax3.plot(tt[n, :], cur[n, :])
    ax3.plot(tt, vol[n, :])
    plt.ylabel('Current (A)')
    plt.xlabel('Time (s)')
    
    plt.show()

if __name__ == "__main__":
    main()
