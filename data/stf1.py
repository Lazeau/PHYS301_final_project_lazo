# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 18:35:31 2020
Import and clean STF-1 data.

@author: tonyf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TODO: Add file selector in GUI
filename = "2019-01-18-14-55-59-SPW-lp-tlm-t.csv"

data = pd.read_csv(filename, header=0)
drops = ["TARGET", "PACKET", "STF1_SCID", "STF1_FIFO", "CCSDS_PKT_VER", "CCSDS_PKT_TYP", "CCSDS_SEC_FLG", "CCSDS_APID", "CCSDS_SEQ_FLAGS", "CCSDS_SEQ_COUNT", "CCSDS_LENGTH", "LP_HEADER_EXP_STATUS", "LP_HEADER_EXP_NUMBER", "LP_HEADER_SYNC_CFE_ELAPSED_SECONDS", "LP_HEADER_SYNC_CFE_ELAPSED_SUBSECONDS", "LP_HEADER_SYNC_FIRMWARE_TIME", "LP_HEADER_COMMAND_TIME", "LP_HEADER_PAYLOAD_SIZE", "LP_DATA_PLASM_START_TM", "LP_DATA_PLASM_STOP_TM", "LP_DATA_BIAS_2_MEAS_DL"]
data = data.drop(drops, axis=1)
data = data.drop(data.filter(regex="Unname"), axis=1)
# FILTER OUT BAD TRACES
data = data.drop([378])

cdsec = data.iloc[:,0] + (data.iloc[:,1]/(2**32)) # not needded for ML

vv = (0.004 * data.iloc[:,2::2] - 2.048) * 5
ii = (0.001 * data.iloc[:,3::2] - 2.048) / 1000000

# Floating potential is bias value at which total current is zero
ii_mins = [ np.argmin(np.abs(ii.values[i])) for i in range(ii.shape[0]) ]
V_f = np.asarray( [ vv.values[i,ii_mins[i]] for i in range(vv.shape[0]) ] )
print(V_f[0])

# Apply linear fit to ion saturation region and calculate I_isat
z = np.polyfit(vv.values[0,:ii_mins[0]], ii.values[0,:ii_mins[0]], 1)
p = np.poly1d(z)
print(p[1]) #slope
print(p)

I_i = p(vv.values[0])
print(I_i)
I_isat = I_i[ii_mins[0]]
I_e = ii.values[0] - I_i
print(I_isat)

plt.plot(vv.values[0], ii.values[0])
plt.plot(vv.values[0], I_i)
plt.plot(vv.values[0], I_e)



## PUT ALL THE TRACES INTO TWO 1D ARRAYS FOR SAWTOOTH WAVE