# -*- coding: utf-8 -*-
"""
Example 3 to demonstarte how to do mTRL calibration
using simulated data (for simplicity, switch-terms not considered)

simulated data adapted from:
    https://scikit-rf.readthedocs.io/en/latest/examples/metrology/Multiline%20TRL.html
"""

import timeit

# need to be installed via pip
import skrf as rf      # for RF stuff
from skrf.media import CPW, Coaxial
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

# main script
if __name__ == '__main__':
    
    # define frequency range
    freq = rf.F(0.1, 300, 3000, unit='GHz')
    
    # 1.0 mm coaxial media for calibration error boxes
    coax1mm = Coaxial(freq, z0=50, Dint=0.44e-3, Dout=1.0e-3, sigma=1e8)
    A = coax1mm.line(1, 'm', z0=58, name='X', embed=True) # left
    B = coax1mm.line(1.1, 'm', z0=40, name='Y', embed=True) # right
    
    # CPW media used for DUT and the calibration standards
    cpw = CPW(freq, z0=50, w=40e-6, s=25e-6, ep_r=12.9*(1-0.001j), t=5e-6, rho=2e-8)
    
    # line standards
    line_lengths = [0,  150e-6,  600e-6, 1500e-6, 1800e-6, 2550e-6]
    lines = [A**cpw.line(l, 'm')**B for l in line_lengths]
    
    # reflect standard
    SHORT = rf.two_port_reflect( cpw.delay_short(0, 'm') )
    reflect = [A**SHORT**B]
    reflect_est = [-1]
    reflect_offset = [0]
    
    # embedded DUT
    dut = cpw.line(3500e-6, 'm', embed=True, z0=100)
    dut_meas = A**dut**B
    
    # calibration object
    cal = mTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, ereff_est=5+0j)
        
    # cal using NIST MultiCal mTRL
    tic = timeit.default_timer()
    cal.run_multical()
    toc = timeit.default_timer()
    print(f'\n NIST MultiCal Elapsed Time: {toc-tic:.6f} seconds')
    dut_cal_nist = cal.apply_cal(dut_meas)
    plot_2x2(dut_cal_nist, title='MultiCal mTRL')
    
    # cal using TUG mTRL
    tic = timeit.default_timer()
    cal.run_tug()
    toc = timeit.default_timer()
    print(f'\n TUG mTRL Elapsed Time: {toc-tic:.6f} seconds')
    dut_cal_tug = cal.apply_cal(dut_meas)
    plot_2x2(dut_cal_tug, title='TUG mTRL')
    
    # original dut
    plot_2x2(dut, title='Original dut')
    
    # difference w.r.t. NIST-MultiCal
    diff = dut_cal_nist - dut
    plot_2x2(diff, title='error diff MultiCal')
    
    # difference w.r.t. TUG mTRL
    diff = dut_cal_tug - dut
    plot_2x2(diff, title='error diff TUG')
    
    plt.show()
    
# EOF