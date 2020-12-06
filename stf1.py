# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 18:35:31 2020
Import and clean STF-1 data.

@author: tonyf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from constants import *

# TODO: Add file selector in GUI
file = 3 # 0-3

datasets = ["2019-01-18-14-55-59-SPW-lp-tlm-t", "2019_02_27_09_01_30_lp_tlm", "2019_04_12_06_01_34_lp_tlm", "2019_04_24_17_58_00_lp_tlm"]
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
# TODO: Add date selector to GUI
skips = {"jan16-19":[378],
         "feb27-19":[214],
         "apr05-19":[0,59,60,61,62,63,93],
         "apr19-19":[213,214,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,530,531]
         }
data = data.drop(skips["apr19-19"])

cdsec = (data.iloc[:,0] + (data.iloc[:,1]/(2**32))).values # not needded for ML?

vv = ((0.004 * data.iloc[:,2::2] - 2.048) * 5).values
ii = ((0.001 * data.iloc[:,3::2] - 2.048) / 1000000).values
traces = vv.shape[0]

# Floating potential is bias value at which total current is zero
Vf_ind = [ np.argmin(np.abs(ii[i])) for i in range(traces) ]
V_f = np.asarray( [ vv[i,Vf_ind[i]] for i in range(traces) ] )
print('floating potential:', V_f)

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

I_e = np.abs(ii - I_i) #+ 0.2E-6
# ION CURRENT PARAMETERIZATION
print('ion current slope, beta:', Ii_b[2])
print('I_is/V_f:', (I_isat[2]+Ii_b[2])/V_f[2])
print('ion current slope range:', max(Ii_b)-min(Ii_b))
print('max {}\nmin {}'.format(max(Ii_b),min(Ii_b)))

# print('ion saturation:', I_isat, I_isat.shape)

# Calculate electron temperature from linear fit to transition region
log_Ie = np.log(I_e)
T_e = np.zeros(traces)
for i in range(traces):
    z = np.polyfit(vv[i,Vf_ind[i]:], log_Ie[i,Vf_ind[i]:], 1)
    p = np.poly1d(z)
    
    try:
        T_e[i] = 1/p[1]
        if T_e[i] > 15:
            T_e[i] = 15 # filter out very high temperatures
        elif T_e[i] < 0:
            T_e[i] = 0.0001 # filter out unphysical temperatures
    except ZeroDivisionError:
        T_e[i] = 0.0001
# print('electron temperature:', T_e)

cs = np.sqrt(m_i / (T_e*Q)) # Bohm velocity
alpha = 0.5 # Magnetization parameter
n_i = (np.multiply(np.abs(I_isat),cs)) / (Q*alpha*A_P)
n_i[n_i > 3E+11] = 3E+11 # Filter out very high densities

fig = plt.figure()
n = 2 # 50 in set 1 is interesting
plt.plot(vv[n], ii[n], '.', label='total')
plt.plot(vv[n], I_i[n], '.', label='ion')
plt.plot(vv[n], I_e[n], '.', label='electron')
plt.plot(V_f[n], I_isat[n], 'o', color='r')

fig2,axes = plt.subplots(2,1, figsize=(8,4.5), dpi=144)
axes[0].plot(cdsec, T_e)
axes[1].plot(cdsec, n_i)
axes[0].set_ylabel('Electron Temp. (eV)')
axes[1].set_ylabel('Ion Density (m^-3')
axes[1].set_xlabel('CCSDS Time (s)')

plt.tight_layout()



## PUT ALL THE TRACES INTO TWO 1D ARRAYS FOR SAWTOOTH WAVE
fig3 = plt.figure()
a = vv.shape[1]
x = np.linspace(0, a, a)
plt.plot(x,vv[2])
# print(vv[2])