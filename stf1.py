# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 18:35:31 2020
Functions to import, clean, and analyze Simulation-to-Flight 1 CubeSat
Langmuir probe data.

@author: mlazo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from constants import *

def read_data(file):
    '''
    Read STF-1 Langmuir probe data from one of several pre-processed data
    files and return voltage and current time series data in SI units.
    
    Parameters
    ----------
    file : int
        Integer selection of one of several pre-processed data files.
        Current available input : data file:
            1 : Jan. 16, 2019
            2 : Feb. 26, 2019
            3 : Apr. 05, 2019
            4 : Apr. 19, 2019
    
    Returns
    -------
    cdsec : ndarray[], s
        Timestamps marking the beginning of a voltage sweep.
        Time is recorded in CCSDS seconds, a count in seconds from a
        NASA-defined epoch. Can be converted to GPS seconds.
    vv : ndarray[][], V
        Voltage sweep values from -5V to +10V, where each row corresponds to
        a single probe trace.
    ii : ndarray[][], A
        Collected values of electron current, where each row corresponds to a
        single probe trace.
    traces : int
        Number of probe traces in file.
    '''
    # TODO: Add file selector in GUI    
    datasets = ["2019-01-18-14-55-59-SPW-lp-tlm-t",
                "2019_02_27_09_01_30_lp_tlm",
                "2019_04_12_06_01_34_lp_tlm",
                "2019_04_24_17_58_00_lp_tlm"
                ]
    filename = "data/{}.csv".format(datasets[file])
    data = pd.read_csv(filename, header=0)
    
    try: # New format
        drops = ["TARGET", "PACKET", "PACKET_TIMESECONDS", "PACKET_TIMEFORMATTED", "RECEIVED_TIMESECONDS", "RECEIVED_TIMEFORMATTED", "RECEIVED_COUNT", "STF1_SCID", "STF1_FIFO", "CCSDS_PKT_VER", "CCSDS_PKT_TYP", "CCSDS_SEC_FLG", "CCSDS_APID", "CCSDS_SEQ_FLAGS", "CCSDS_SEQ_COUNT", "CCSDS_LENGTH", "LP_HEADER_EXP_STATUS", "LP_HEADER_EXP_NUMBER", "LP_HEADER_SYNC_CFE_ELAPSED_SECONDS", "LP_HEADER_SYNC_CFE_ELAPSED_SUBSECONDS", "LP_HEADER_SYNC_FIRMWARE_TIME", "LP_HEADER_COMMAND_TIME", "LP_HEADER_PAYLOAD_SIZE", "LP_DATA_PLASM_START_TM", "LP_DATA_PLASM_STOP_TM", "LP_DATA_BIAS_2_MEAS_DL"]
        data = data.drop(drops, axis=1)
        data = data.drop(data.filter(regex="Unname"), axis=1)
    except: # Old format
        drops = ["TARGET", "PACKET",  "STF1_SCID", "STF1_FIFO", "CCSDS_PKT_VER", "CCSDS_PKT_TYP", "CCSDS_SEC_FLG", "CCSDS_APID", "CCSDS_SEQ_FLAGS", "CCSDS_SEQ_COUNT", "CCSDS_LENGTH", "LP_HEADER_EXP_STATUS", "LP_HEADER_EXP_NUMBER", "LP_HEADER_SYNC_CFE_ELAPSED_SECONDS", "LP_HEADER_SYNC_CFE_ELAPSED_SUBSECONDS", "LP_HEADER_SYNC_FIRMWARE_TIME", "LP_HEADER_COMMAND_TIME", "LP_HEADER_PAYLOAD_SIZE", "LP_DATA_PLASM_START_TM", "LP_DATA_PLASM_STOP_TM", "LP_DATA_BIAS_2_MEAS_DL"]
        data = data.drop(drops, axis=1)
        data = data.drop(data.filter(regex="Unname"), axis=1)
    
    # FILTER OUT BAD TRACES
    skips = {0:[378],
             1:[214],
             2:[0,59,60,61,62,63,93],
             3:[213,214,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,530,531]
             }
    data = data.drop(skips[file])
    
    cdsec = (data.iloc[:,0] + (data.iloc[:,1]/(2**32))).values # not needded for ML?
    
    vv = ((0.004 * data.iloc[:,2::2] - 2.048) * 5).values
    ii = ((0.001 * data.iloc[:,3::2] - 2.048) / 1000000).values
    traces = vv.shape[0]
    
    return cdsec, vv, ii, traces

def analyze_data(vv, ii, traces):
    '''
    Calculate floating potential V_f, electron temperature T_e, and ion
    density n_i for the provided Langmuir probe current-voltage characteristic.
    
    Parameters
    ----------
    vv : ndarray[][], V
        Voltage sweep values from -5V to +10V, where each row corresponds to
        a single probe trace.
    ii : ndarray[][], A
        Collected values of electron current, where each row corresponds to a
        single probe trace.
    traces : int
        Number of probe traces in file.
    
    Returns
    -------
    T_e : ndarray[], eV
        Plasma electron temperatures for each trace.
    I_isat : ndarray[], A
        Plasma ion saturation currents.
    n_i : ndarray[], m^-3
        Plasma ion densities.
    V_f : ndarray[], V
        Plasma floating potentials.
    Ii_m : ndarray[], C^2/Js
        Slopes which characterize ion current via plasma sheath expansion.
        Found from a linear fit to the ion saturation region of the I-V trace.
    Ii_b : ndarray[], A
        Intercepts which characterize ion current via plasma sheath expansion.
        Found from a linear fit to the ion saturation region of the I-V trace.
    '''
    # Floating potential is bias value at which total current is zero
    Vf_ind = [ np.argmin(np.abs(ii[i])) for i in range(traces) ]
    V_f = np.asarray( [ vv[i,Vf_ind[i]] for i in range(traces) ] )
    
    # Apply linear fit to ion saturation region and calculate I_isat
    I_i = np.zeros((traces,vv.shape[1]))
    I_isat = np.zeros(traces)
    Ii_m = np.zeros(traces)
    Ii_b = np.zeros(traces)
    for i in range(traces):
        z = np.polyfit(vv[i,:Vf_ind[i]], ii[i,:Vf_ind[i]], 1)
        p = np.poly1d(z)
        
        I_i[i] = p(vv[i])
        Ii_m[i] = p[1]
        Ii_b[i] = p[0]
        
        I_isat[i] = I_i[i,Vf_ind[i]]
    
    I_e = np.abs(ii - I_i)
        
    # Calculate electron temperature from linear fit to transition region
    log_Ie = np.log(I_e)
    T_e = np.zeros(traces)
    # Keep track of calculated values outside characteristic parameter regimes
    very_high_te = []
    very_high_ni = []
    for i in range(traces):
        z = np.polyfit(vv[i,Vf_ind[i]:], log_Ie[i,Vf_ind[i]:], 1)
        p = np.poly1d(z)
        
        try:
            T_e[i] = 1/p[1]
            if T_e[i] > 8.5:
                very_high_te.append(T_e[i])
                T_e[i] = 8.5 # filter out very high temperatures
            elif T_e[i] < 0:
                T_e[i] = 0.0001 # filter out unphysical temperatures
        except ZeroDivisionError:
            T_e[i] = 0.0001
    
    very_high_ni = []
    cs = np.sqrt(m_i / (T_e*Q)) # Bohm velocity
    alpha = 0.5 # Magnetization parameter
    n_i = (np.multiply(np.abs(I_isat),cs)) / (Q*alpha*A_P)
    
    very_high_ni = n_i[n_i>1E+12]
    n_i[n_i >= 1E+12] = 1E+12 # Filter out very high densities
    print('Uncharacteristic Electron Temperatures:\n', very_high_te,
          '\nUncharacteristic Ion Densities:\n',very_high_ni)
    print('\nNumber of unchar. temps.:', len(very_high_te),
          '\nNumber of unchar. densities.:', len(very_high_ni))
    
    ## For testing
    fig = plt.figure()
    n = 2 # 50 in set 1 is interesting
    plt.plot(vv[n], ii[n], '.', label='total')
    plt.plot(vv[n], I_i[n], '.', label='ion')
    plt.plot(vv[n], I_e[n], '.', label='electron')
    plt.plot(V_f[n], I_isat[n], 'o', color='r')
    
    return T_e, I_isat, n_i, V_f, Ii_m, Ii_b

## For testing
cdsec,vv,ii,traces = read_data(0)
T_e,I_is,n_i,V_f,Ii_m,Ii_b = analyze_data(vv, ii, traces)

fig2,axes = plt.subplots(2,1, figsize=(8,4.5), dpi=144)
axes[0].plot(cdsec, T_e)
axes[1].plot(cdsec, n_i)
axes[0].set_ylabel('Electron Temp. (eV)')
axes[1].set_ylabel('Ion Density (m^-3')
axes[1].set_xlabel('CCSDS Time (s)')

print('sample Te, Is:', T_e[50], I_is[50])
print('sample ni:', n_i[50])
print('sample beta_m, beta_b:', Ii_m[50],Ii_b[50])

plt.tight_layout()

# fig3 = plt.figure()
# plt.plot(vv[2])
# print(vv[2])