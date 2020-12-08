# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:30:35 2020
Fundamental physical and experimental constants.

@author: mlazo
"""

# Physical constants
C = 2.998e+8             # Speed of light, in m/s
C2 = C**2                # Speed of light squared, in m^2/s^2
Q = 1.602e-19            # Fundamental charge, in C
M_P = 1.673e-27          # Proton mass, in kg
MP_EV = 938.3e+6 / C2    # Proton mass, in eV/c^2
M_E = 9.109e-31          # Electron mass, in kg
ME_EV = 0.511e+6 / C2    # Electron mass, in eV/c^2

mu_o = 16            # Mass number of oxygen, in amu
m_i = M_P * mu_o     # Ion mass of singly ionized, monatomic oxygen, in kg
mi_ev = MP_EV * mu_o # Ion mass of singly ionized, monatomic oxygen, in ev/c^2

# Probe parameters
CUTOFF = 2.420e-06  # Physical cutoff electron current for CubeSat, in A
A_P = 7.7515e-3     # Probe area, in m^2

# VALUES BELOW ARE PLACEHOLDERS
# Label parameters
T_min = 0.1 # Minimum electron temperature for modeling, in eV
T_max = 8.5 # Maximum electron temperature for modeling, in eV
Is_min = 0  # Minimum ion saturation current for modeling, in A
Is_max = 1  # Maximum ion saturation current for modeling, in A