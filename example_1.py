# -*- coding: utf-8 -*-
"""
Example 1 to demonstarte how to do mTRL calibration
This is a 2nd-tier mTRL calibration
The .s2p files already went through a 1st-tier calibration
"""

# part of python standard library
import os  

# need to be installed via pip
import skrf as rf      # for RF stuff
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt   # for plotting

# my script (MultiCal.py and TUGmTRL must also be in same folder)
from mTRL import mTRL

def plot_2x2(NW, f_units='ghz', title='mTRL'):
    fig, axs = plt.subplots(2,2)
    NW.frequency.unit = f_units
    for inx in NW.port_tuples:
        m = inx[0]
        n = inx[1]
        NW.plot_s_db(m=m, n=n, ax=axs[inx])
    fig.suptitle(title)
    fig.tight_layout(pad=1.08)

# main script
if __name__ == '__main__':
    
    # load the measurements
    # files' path are reference to script's path
    s2p_path = os.path.dirname(os.path.realpath(__file__)) + '\\s2p_example_1\\'
    
    # Calibration standards
    L1    = rf.Network(s2p_path + 'Cascade_line_0200u.s2p')
    L2    = rf.Network(s2p_path + 'Cascade_line_0450u.s2p')
    L3    = rf.Network(s2p_path + 'Cascade_line_0900u.s2p')
    L4    = rf.Network(s2p_path + 'Cascade_line_1800u.s2p')
    SHORT = rf.Network(s2p_path + 'Cascade_short.s2p')
    
    # Verification lines (not part of mTRL)
    L5    = rf.Network(s2p_path + 'Cascade_line_3500u.s2p')
    L6    = rf.Network(s2p_path + 'Cascade_line_5250u.s2p')
    
    lines = [L1, L2, L3, L4]
    line_lengths = [200e-6, 450e-6, 900e-6, 1800e-6]
    reflect = [SHORT]
    reflect_est = [-1]
    reflect_offset = [0]
    
    cal = mTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, ereff_est=5+0j)
    
    # using NIST MultiCal mTRL
    cal.run_multical()
    L5_cal_nist = cal.apply_cal(L5)
    plot_2x2(L5_cal_nist, title='MultiCal mTRL')
    
    # using TUG mTRL
    cal.run_tug()
    L5_cal_tug = cal.apply_cal(L5)
    plot_2x2(L5_cal_tug, title='TUG mTRL')

    plt.show()
    
# EOF