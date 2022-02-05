# -*- coding: utf-8 -*-
"""
Example 4 to demonstarte how to analyze the statistical performance of mTRL 
calibration using measured data (for simplicity, I'm unisg data from ex-1)
"""

# part of python standard library
import os
import timeit

# need to be installed via pip
import skrf as rf      # for RF stuff
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt   # for plotting

# my script (MultiCal.py and TUGmTRL.py must also be in same folder)
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

def plot_2x2_2(NW, fig, axs, f_units='ghz', name='mTRL', title='mTRL'):
    NW.frequency.unit = f_units
    NW.name = name
    for inx in NW.port_tuples:
        m = inx[0]
        n = inx[1]
        NW.plot_s_db(m=m, n=n, ax=axs[inx])
    fig.suptitle(title)
    fig.tight_layout(pad=1.08)

def add_white_noise(NW, sigma=0.01):
    # add white noise to a network's S-paramters
    freq  = NW.frequency
    noise = (np.random.standard_normal((len(freq.f),2,2)) 
             + 1j*np.random.standard_normal((len(freq.f),2,2)))*sigma
    S = NW.s + noise
    return rf.Network(frequency=freq, s=S)

def add_phase_error(NW, sigma=1):
    # add gaussian phase noise (in degrees) to a network's S-paramters
    freq  = NW.frequency
    phase_noise = np.random.standard_normal((len(freq.f),2,2))*sigma
    S = abs(NW.s)*np.exp(1j*np.deg2rad(np.angle(NW.s, deg=True) + phase_noise))
    
    return rf.Network(frequency=freq, s=S)

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
    
    # DUT noisless
    cal = mTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
                   reflect_est=reflect_est, reflect_offset=reflect_offset, ereff_est=5+0j)
    dut_meas = L5
    cal.run_tug()
    dut = cal.apply_cal(dut_meas)
    print('\n Noiseless case done.\n\n')
    
    # Monte Carlo Analysis
    M = 10 # number of trials
    sigma_phi = 15
    sigma_noise = 0.1
    NIST_ntws = []
    TUG_ntws  = []
    for inx in range(M):
        # addituve noise
        #lines_n   = [add_white_noise(NW, sigma_noise) for NW in lines]
        #reflect_n = [add_white_noise(NW, sigma_noise) for NW in reflect]
        
        # phase error
        lines_n   = [add_phase_error(NW, sigma_phi) for NW in lines]
        reflect_n = [add_phase_error(NW, sigma_phi) for NW in reflect]
        
        # calibration object
        cal = mTRL(lines=lines_n, line_lengths=line_lengths, reflect=reflect_n, 
                   reflect_est=reflect_est, reflect_offset=reflect_offset, ereff_est=5+0j)
            
        # cal using NIST MultiCal mTRL
        cal.run_multical()    
        NIST_ntws.append( cal.apply_cal(dut_meas) )
        
        # cal using TUG mTRL
        cal.run_tug()
        TUG_ntws.append( cal.apply_cal(dut_meas) )
        
        print(f'\n Index: {inx+1} out of {M} done.')
    
    NIST_ntws = rf.NetworkSet(NIST_ntws)
    TUG_ntws = rf.NetworkSet(TUG_ntws)
    
    # mean error NIST vs. TUG
    fig, axs = plt.subplots(2,2)
    plot_2x2_2((NIST_ntws - dut).mean_s, fig, axs, 
               name='NIST', title='Mean-Error: NIST vs. TUG mTRL')
    plot_2x2_2((TUG_ntws - dut).mean_s, fig, axs, 
               name='TUG', title='Mean-Error: NIST vs. TUG mTRL')
    
    plt.show()
    
# EOF