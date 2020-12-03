# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:30:35 2020
Fundamental physical and experimental constants.

@author: mlazo
"""

Q = 1.602E-19   # Fundamental charge, in C
M_P = 1.673E-27 # Proton mass, in kg
M_E = 9.109E-31 # Electron mass, in kg
mu_o = 16       # Mass number of oxygen, in amu

CUTOFF = 2.420E-06  # Physical cutoff electron current for CubeSat, in A
A_P = 7.7515E-3     # Probe area, in m^2

m_i = M_P * mu_o # Ion mass of singly ionized, monatomic oxygen, in kg

# VALUES BELOW ARE PLACEHOLDERS
T_min = 0.1 # Minimum electron temperature for modeling, in eV
T_max = 2   # Maximum electron temperature for modeling, in eV
Is_min = 0  # Minimum ion saturation current for modeling, in A
Is_max = 1  # Maximum ion saturation current for modeling, in A