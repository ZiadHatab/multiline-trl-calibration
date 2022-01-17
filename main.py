# -*- coding: utf-8 -*-
"""
Example to show you how to do the calibration
"""

import os

from mTRL import mTRL

import skrf as rf      # for RF stuff
import numpy as np
import matplotlib.pyplot as plt   # for plotting


# main script
if __name__ == '__main__':
    c0 = 299792458
    
    # measurements path assumed to be same as script's path
    path = os.path.dirname(os.path.realpath(__file__))
    # load the measurements and return the mean network
    lower_ghz   = 10
    upper_ghz   = 150
    # Calibration standards
    L0    = rf.NetworkSet(rf.read_all(path, contains='1ps_200u')).mean_s[f'{lower_ghz}-{upper_ghz}ghz']
    L1    = rf.NetworkSet(rf.read_all(path, contains='3ps_450u')).mean_s[f'{lower_ghz}-{upper_ghz}ghz']
    L2    = rf.NetworkSet(rf.read_all(path, contains='7ps_900u')).mean_s[f'{lower_ghz}-{upper_ghz}ghz']
    L3    = rf.NetworkSet(rf.read_all(path, contains='14ps_1800u')).mean_s[f'{lower_ghz}-{upper_ghz}ghz']
    SHORT = rf.NetworkSet(rf.read_all(path, contains='short')).mean_s[f'{lower_ghz}-{upper_ghz}ghz']
    
    # Verification lines (not part of mTRL)
    L4    = rf.NetworkSet(rf.read_all(path, contains='27ps_3500u')).mean_s[f'{lower_ghz}-{upper_ghz}ghz']
    L5    = rf.NetworkSet(rf.read_all(path, contains='40ps_5250u')).mean_s[f'{lower_ghz}-{upper_ghz}ghz']
        
    freq = L0.frequency
    f = freq.f
    
    lines = [L0, L1, L2, L3]
    line_lengths = [200e-6, 450e-6, 900e-6, 1800e-6]
    reflect = [SHORT]
    reflect_est = [-1]
    reflect_offset = [0]
    
    cal = mTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, ereff_est=1+0j)
    
    cal.run_multical()
    
    A = cal.apply_cal(L0)
    
    cal.run_tug()
    
    B = cal.apply_cal(L0)
    
    plt.figure()
    A.s11.plot_s_db()
    B.s11.plot_s_db()


    plt.show()
    
# EOF