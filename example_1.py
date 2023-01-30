"""
Example to demonstrate how to do mTRL calibration. This is a 2nd-tier mTRL calibration (aka de-embedding)
The .s2p files already went through a 1st-tier calibration.
"""
import os

# need to be installed via pip
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt   # for plotting

# my script (MultiCal.py and TUGmTRL must also be in same folder)
from mTRL import mTRL

class PlotSettings:
    # to make plots look better for publication
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    def __init__(self, font_size=10, latex=False): 
        self.font_size = font_size 
        self.latex = latex
    def __enter__(self):
        plt.style.use('seaborn-v0_8-paper')
        # make svg output text and not curves
        plt.rcParams['svg.fonttype'] = 'none'
        # fontsize of the axes title
        plt.rc('axes', titlesize=self.font_size*1.2)
        # fontsize of the x and y labels
        plt.rc('axes', labelsize=self.font_size)
        # fontsize of the tick labels
        plt.rc('xtick', labelsize=self.font_size)
        plt.rc('ytick', labelsize=self.font_size)
        # legend fontsize
        plt.rc('legend', fontsize=self.font_size*1)
        # fontsize of the figure title
        plt.rc('figure', titlesize=self.font_size)
        # controls default text sizes
        plt.rc('text', usetex=self.latex)
        #plt.rc('font', size=self.font_size, family='serif', serif='Times New Roman')
        plt.rc('lines', linewidth=1.5)
    def __exit__(self, exception_type, exception_value, traceback):
        plt.style.use('default')

def plot_2x2(NW, fig, axs, f_units='ghz', name='mTRL', title='mTRL'):
    NW.frequency.unit = f_units
    NW.name = name
    for inx in NW.port_tuples:
        m = inx[0]
        n = inx[1]
        NW.plot_s_db(m=m, n=n, ax=axs[inx])
    fig.suptitle(title)
    fig.tight_layout(pad=1.08)

# main script
if __name__ == '__main__':
    # useful functions
    c0 = 299792458   # speed of light in vacuum (m/s)
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(complex).eps))  # eps to ensure positive square-root
    gamma2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))  # losses dB/mm
    # load the measurements
    # files' path are reference to script's path
    s2p_path = os.path.dirname(os.path.realpath(__file__)) + '\\s2p_example_1\\'
    
    # Calibration standards
    L1    = rf.Network(s2p_path + 'Cascade_line_0200u.s2p')
    L2    = rf.Network(s2p_path + 'Cascade_line_0450u.s2p')
    L3    = rf.Network(s2p_path + 'Cascade_line_0900u.s2p')
    L4    = rf.Network(s2p_path + 'Cascade_line_1800u.s2p')
    L5    = rf.Network(s2p_path + 'Cascade_line_3500u.s2p')
    L6    = rf.Network(s2p_path + 'Cascade_line_5250u.s2p') # used as well as DUT
    SHORT = rf.Network(s2p_path + 'Cascade_short.s2p')
    
    lines = [L1, L2, L3, L4, L5, L6]
    line_lengths = [200e-6, 450e-6, 900e-6, 1800e-6, 3500e-6, 5250e-6]
    reflect = [SHORT]
    reflect_est = [-1]
    reflect_offset = [0]
    
    cal = mTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, ereff_est=5+0j)
    
    DUT = L6
    # using NIST MultiCal
    cal.run_multical()
    dut_cal_nist = cal.apply_cal(DUT)
    gamma_mul = cal.gamma
    ereff_mul = cal.ereff
    
    # using TUG mTRL
    cal.run_tug()
    dut_cal_tug = cal.apply_cal(DUT)
    gamma_tug = cal.gamma
    ereff_tug = cal.ereff
    
    # using skrf 
    line_lengths = line_lengths
    offset = line_lengths[0]
    line_lengths = [i - offset for i in line_lengths]  # to set the reference the same as my code
    measured = [L1, SHORT, L2, L3, L4, L5, L6]
    cal_skrf = rf.NISTMultilineTRL(
        measured = measured,
        Grefls = [-1],
        l = line_lengths,
        er_est = 5+0j,
    )
    cal_skrf.run()
    dut_cal_skrf = cal_skrf.apply_cal(DUT)
    gamma_skrf = cal_skrf.gamma
    ereff_skrf = cal_skrf.er_eff

    with PlotSettings(14):
        fig, axs = plt.subplots(2,2, figsize=(10,7))
        fig.set_dpi(600)
        plot_2x2(dut_cal_nist, fig, axs, name='NIST MultiCal', title='Calibrated DUT (Line)')
        plot_2x2(dut_cal_tug, fig, axs, name='TUG mTRL', title='Calibrated DUT (Line)')
        plot_2x2(dut_cal_skrf, fig, axs, name='skrf', title='Calibrated DUT (Line)')

    f = L1.frequency.f
    with PlotSettings(14):
        fig, axs = plt.subplots(1,2, figsize=(10,3.8))        
        fig.set_dpi(600)
        fig.tight_layout(pad=2)
        ax = axs[0]
        ax.plot(f*1e-9, ereff_mul.real, lw=2, label='NIST MultiCal', 
                marker='^', markevery=50, markersize=10)
        ax.plot(f*1e-9, ereff_tug.real, lw=2, label='TUG mTRL', 
                marker='v', markevery=50, markersize=10)
        ax.plot(f*1e-9, ereff_skrf.real, lw=2, label='skrf', 
                marker='>', markevery=50, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Relative effective permittivity')
        ax.set_ylim([4.5, 6])
        ax.set_yticks(np.arange(4.5, 6.01, 0.3))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.legend()
        ax = axs[1]
        ax.plot(f*1e-9, gamma2dbmm(gamma_mul), lw=2, label='NIST MultiCal', 
                marker='^', markevery=50, markersize=10)
        ax.plot(f*1e-9, gamma2dbmm(gamma_tug), lw=2, label='TUG mTRL', 
                marker='v', markevery=50, markersize=10)
        ax.plot(f*1e-9, gamma2dbmm(gamma_skrf), lw=2, label='skrf', 
                marker='>', markevery=50, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Loss (dB/mm)')
        ax.set_ylim([0, 1.5])
        ax.set_yticks(np.arange(0, 1.51, 0.3))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.legend()
    
    plt.show()
    
# EOF