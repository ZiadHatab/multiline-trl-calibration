"""
Example comparing the statistical performance of different mTRL calibrations.

A Monte Carlo analysis adds random noise to the measured standards and compares
the mean absolute error (MAE) of NIST MultiCal, TUG mTRL, and scikit-rf, both for
the calibration coefficients and for the extracted line parameters (effective
permittivity and loss per unit length). The trials are independent, so they are
run in parallel across CPU cores to speed up the (otherwise slow) analysis.
"""
import os
import io
import time
import contextlib
import warnings
from functools import partial
from multiprocessing import Pool

# need to be installed via pip
import skrf as rf
from skrf.calibration import NISTMultilineTRL
import numpy as np
import matplotlib.pyplot as plt

# my script (MultiCal.py and TUGmTRL.py must also be in same folder)
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

def add_white_noise(NW, sigma=0.01):
    # add white noise to a network's S-parameters
    freq  = NW.frequency
    noise = (np.random.standard_normal((len(freq.f),2,2))
             + 1j*np.random.standard_normal((len(freq.f),2,2)))*sigma
    S = NW.s + noise
    return rf.Network(frequency=freq, s=S)

def add_uniform_noise(NW, lower=-0.01, upper=0.01):
    # add uniform noise to a network's S-parameters
    freq  = NW.frequency
    noise  = np.random.uniform(lower, upper, (len(freq.f),2,2)) + \
        1j*np.random.uniform(lower, upper, (len(freq.f),2,2))
    S = NW.s + noise
    return rf.Network(frequency=freq, s=S)

def add_phase_error(NW, lower=-5, upper=5):
    # add uniform phase noise (in degrees) to a network's S-parameters
    freq  = NW.frequency
    noise = np.random.uniform(lower, upper, (len(freq.f),2,2))
    S = abs(NW.s)*np.exp(1j*np.deg2rad(np.angle(NW.s, deg=True) + noise))
    return rf.Network(frequency=freq, s=S)

def coef_MAE(coef_MC, coefs_ideal, name, name2=None):
    # mean absolute error (over the trials) of a calibration coefficient
    name2 = name if name2 is None else name2
    return np.array([ abs(x[name]-coefs_ideal[name2]) for x in coef_MC ]).mean(axis=0)

def gamma_MAE(gamma_MC, gamma_ideal, transform):
    # mean absolute error (over the trials) of a real-valued line parameter,
    # i.e. transform(gamma) such as Re(ereff) or loss in dB/mm.
    ref = transform(gamma_ideal)
    return np.array([ abs(transform(g)-ref) for g in gamma_MC ]).mean(axis=0)

def run_trial(seed, lines, reflect, line_lengths, reflect_est, reflect_offset, ereff_est, sigma_noise):
    # one Monte Carlo trial: add noise, then calibrate with all three methods.
    # the seed makes each trial use a different (but reproducible) noise realization.
    np.random.seed(seed)
    lines_n   = [add_white_noise(NW, sigma_noise) for NW in lines]
    reflect_n = [add_white_noise(NW, sigma_noise) for NW in reflect]

    # silence the per-frequency progress prints and warnings coming from the workers
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')

        cal = mTRL(lines=lines_n, line_lengths=line_lengths, reflect=reflect_n,
                   reflect_est=reflect_est, reflect_offset=reflect_offset, ereff_est=ereff_est)
        cal.run_multical()                    # NIST MultiCal
        coefs_NIST, gamma_NIST = cal.coefs, cal.gamma
        cal.run_tug()                         # TUG mTRL
        coefs_TUG, gamma_TUG = cal.coefs, cal.gamma

        # scikit-rf (uses the line list with the reflect inserted after the thru)
        cal_skrf = NISTMultilineTRL(
            measured = [lines_n[0], reflect_n[0], *lines_n[1:]],
            Grefls = [-1],
            l = [i - line_lengths[0] for i in line_lengths],
            refl_offset = reflect_offset,
            er_est = ereff_est)
        cal_skrf.run()
        coefs_skrf, gamma_skrf = cal_skrf.coefs, cal_skrf.gamma

    return coefs_NIST, gamma_NIST, coefs_TUG, gamma_TUG, coefs_skrf, gamma_skrf

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
    script_dir = os.path.dirname(os.path.realpath(__file__))
    s2p_path = script_dir + '\\s2p_example_1\\'
    fig_dir  = script_dir + '\\images\\'

    # Calibration standards
    L1    = rf.Network(s2p_path + 'Cascade_line_0200u.s2p')
    L2    = rf.Network(s2p_path + 'Cascade_line_0450u.s2p')
    L3    = rf.Network(s2p_path + 'Cascade_line_0900u.s2p')
    L4    = rf.Network(s2p_path + 'Cascade_line_1800u.s2p')
    L5    = rf.Network(s2p_path + 'Cascade_line_3500u.s2p')
    L6    = rf.Network(s2p_path + 'Cascade_line_5250u.s2p') # used as well as DUT
    SHORT = rf.Network(s2p_path + 'Cascade_short.s2p')
    freq = L1.frequency
    f = freq.f

    lines = [L1, L2, L3, L4, L5, L6]
    line_lengths = [200e-6, 450e-6, 900e-6, 1800e-6, 3500e-6, 5250e-6]
    reflect = [SHORT]
    reflect_est = [-1]
    reflect_offset = [-100e-6]
    ereff_est = 6.2-0.0001j

    # DUT noiseless
    cal = mTRL(lines=lines, line_lengths=line_lengths, reflect=reflect,
               reflect_est=reflect_est, reflect_offset=reflect_offset, ereff_est=ereff_est)
    print('\nNoiseless case...')
    with contextlib.redirect_stdout(io.StringIO()):
        cal.run_multical()   # use MultiCal as reference
    coefs_ideal = cal.coefs
    gamma_ideal = cal.gamma

    # Monte Carlo Analysis
    M = 100            # number of trials
    sigma_noise = 0.2  # noise standard deviation
    print(f'\n\nWith noise... ({M} trials, scikit-rf {rf.__version__})')

    # the trials are independent, so run them in parallel across all CPU cores.
    # imap_unordered yields each result as soon as a trial finishes, so we can print
    # progress (order does not matter: the trials are averaged together afterwards).
    worker = partial(run_trial, lines=lines, reflect=reflect, line_lengths=line_lengths,
                     reflect_est=reflect_est, reflect_offset=reflect_offset,
                     ereff_est=ereff_est, sigma_noise=sigma_noise)
    results = []
    t0 = time.time()
    with Pool() as pool:
        for i, res in enumerate(pool.imap_unordered(worker, range(M)), start=1):
            results.append(res)
            elapsed = time.time() - t0
            print(f'  {i}/{M} trials done | elapsed {elapsed:.0f} s, eta {elapsed/i*(M-i):.0f} s   ',
                  end='\r', flush=True)
    print()
    coefs_NIST, gamma_NIST, coefs_TUG, gamma_TUG, coefs_skrf, gamma_skrf = map(list, zip(*results))

    EDF_NIST = coef_MAE(coefs_NIST, coefs_ideal, 'EDF')
    ESF_NIST = coef_MAE(coefs_NIST, coefs_ideal, 'ESF')
    ERF_NIST = coef_MAE(coefs_NIST, coefs_ideal, 'ERF')
    EDR_NIST = coef_MAE(coefs_NIST, coefs_ideal, 'EDR')
    ESR_NIST = coef_MAE(coefs_NIST, coefs_ideal, 'ESR')
    ERR_NIST = coef_MAE(coefs_NIST, coefs_ideal, 'ERR')
    ETF_NIST = coef_MAE(coefs_NIST, coefs_ideal, 'ETF')
    ETR_NIST = coef_MAE(coefs_NIST, coefs_ideal, 'ETR')

    EDF_TUG = coef_MAE(coefs_TUG, coefs_ideal, 'EDF')
    ESF_TUG = coef_MAE(coefs_TUG, coefs_ideal, 'ESF')
    ERF_TUG = coef_MAE(coefs_TUG, coefs_ideal, 'ERF')
    EDR_TUG = coef_MAE(coefs_TUG, coefs_ideal, 'EDR')
    ESR_TUG = coef_MAE(coefs_TUG, coefs_ideal, 'ESR')
    ERR_TUG = coef_MAE(coefs_TUG, coefs_ideal, 'ERR')
    ETF_TUG = coef_MAE(coefs_TUG, coefs_ideal, 'ETF')
    ETR_TUG = coef_MAE(coefs_TUG, coefs_ideal, 'ETR')

    EDF_skrf = coef_MAE(coefs_skrf, coefs_ideal, 'forward directivity', 'EDF')
    ESF_skrf = coef_MAE(coefs_skrf, coefs_ideal, 'forward source match', 'ESF')
    ERF_skrf = coef_MAE(coefs_skrf, coefs_ideal, 'forward reflection tracking', 'ERF')
    EDR_skrf = coef_MAE(coefs_skrf, coefs_ideal, 'reverse directivity', 'EDR')
    ESR_skrf = coef_MAE(coefs_skrf, coefs_ideal, 'reverse source match', 'ESR')
    ERR_skrf = coef_MAE(coefs_skrf, coefs_ideal, 'reverse reflection tracking', 'ERR')

    # MAE of the extracted line parameters (same definitions as in example 1 and 2):
    # effective permittivity (real part) and loss per unit length (dB/mm).
    ereff = lambda g: gamma2ereff(g, f).real
    ereff_MAE_NIST = gamma_MAE(gamma_NIST, gamma_ideal, ereff)
    ereff_MAE_TUG  = gamma_MAE(gamma_TUG,  gamma_ideal, ereff)
    ereff_MAE_skrf = gamma_MAE(gamma_skrf, gamma_ideal, ereff)
    loss_MAE_NIST  = gamma_MAE(gamma_NIST, gamma_ideal, gamma2dbmm)
    loss_MAE_TUG   = gamma_MAE(gamma_TUG,  gamma_ideal, gamma2dbmm)
    loss_MAE_skrf  = gamma_MAE(gamma_skrf, gamma_ideal, gamma2dbmm)

    # filename suffix encoding the noise level, e.g. sigma=0.2 -> 'std_0_2'
    suffix = f'std_{sigma_noise}'.replace('.', '_')

    with PlotSettings(14):
        fig, axs = plt.subplots(3,2, figsize=(10,11))
        fig.set_dpi(600)
        fig.tight_layout(pad=2)
        ax = axs[0,0]
        ax.plot(f*1e-9, mag2db(EDF_NIST), lw=2, label='NIST MultiCal',
                marker='^', markevery=50, markersize=10)
        ax.plot(f*1e-9, mag2db(EDF_TUG), lw=2, label='TUG mTRL',
                marker='v', markevery=50, markersize=10)
        ax.plot(f*1e-9, mag2db(EDF_skrf), lw=2, label=f'NIST skrf v{rf.__version__}',
                marker='>', markevery=50, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Forward directivity')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))

        ax = axs[1,0]
        ax.plot(f*1e-9, mag2db(ESF_NIST), lw=2, label='NIST MultiCal',
                marker='^', markevery=50, markersize=10)
        ax.plot(f*1e-9, mag2db(ESF_TUG), lw=2, label='TUG mTRL',
                marker='v', markevery=50, markersize=10)
        ax.plot(f*1e-9, mag2db(ESF_skrf), lw=2, label=f'NIST skrf v{rf.__version__}',
                marker='>', markevery=50, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Forward source match')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))

        ax = axs[2,0]
        ax.plot(f*1e-9, mag2db(ERF_NIST), lw=2, label='NIST MultiCal',
                marker='^', markevery=50, markersize=10)
        ax.plot(f*1e-9, mag2db(ERF_TUG), lw=2, label='TUG mTRL',
                marker='v', markevery=50, markersize=10)
        ax.plot(f*1e-9, mag2db(ERF_skrf), lw=2, label=f'NIST skrf v{rf.__version__}',
                marker='>', markevery=50, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Forward reflection tracking')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))

        ax = axs[0,1]
        ax.plot(f*1e-9, mag2db(EDR_NIST), lw=2, label='NIST MultiCal',
                marker='^', markevery=50, markersize=10)
        ax.plot(f*1e-9, mag2db(EDR_TUG), lw=2, label='TUG mTRL',
                marker='v', markevery=50, markersize=10)
        ax.plot(f*1e-9, mag2db(EDR_skrf), lw=2, label=f'NIST skrf v{rf.__version__}',
                marker='>', markevery=50, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Reverse directivity')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))

        ax = axs[1,1]
        ax.plot(f*1e-9, mag2db(ESR_NIST), lw=2, label='NIST MultiCal',
                marker='^', markevery=50, markersize=10)
        ax.plot(f*1e-9, mag2db(ESR_TUG), lw=2, label='TUG mTRL',
                marker='v', markevery=50, markersize=10)
        ax.plot(f*1e-9, mag2db(ESR_skrf), lw=2, label=f'NIST skrf v{rf.__version__}',
                marker='>', markevery=50, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Reverse source match')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))

        ax = axs[2,1]
        ax.plot(f*1e-9, mag2db(ERR_NIST), lw=2, label='NIST MultiCal',
                marker='^', markevery=50, markersize=10)
        ax.plot(f*1e-9, mag2db(ERR_TUG), lw=2, label='TUG mTRL',
                marker='v', markevery=50, markersize=10)
        ax.plot(f*1e-9, mag2db(ERR_skrf), lw=2, label=f'NIST skrf v{rf.__version__}',
                marker='>', markevery=50, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Reverse reflection tracking')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.98),
                   loc='lower center', ncol=3, borderaxespad=0
                   )
        plt.suptitle(f"Mean Absolute Error (MAE) in dB of calibration coefficients. Noise std = {sigma_noise:.2f}", verticalalignment='bottom').set_y(1.02)
        #fig.savefig(fig_dir + f'example_3_{suffix}.png', dpi=600, bbox_inches='tight')

    # MAE of the extracted line parameters, in the same format as the coefficient plots.
    with PlotSettings(14):
        fig, axs = plt.subplots(1,2, figsize=(10,3.8))
        fig.set_dpi(600)
        fig.tight_layout(pad=2)
        ax = axs[0]
        ax.plot(f*1e-9, ereff_MAE_NIST, lw=2, label='NIST MultiCal',
                marker='^', markevery=50, markersize=10)
        ax.plot(f*1e-9, ereff_MAE_TUG, lw=2, label='TUG mTRL',
                marker='v', markevery=50, markersize=10)
        ax.plot(f*1e-9, ereff_MAE_skrf, lw=2, label=f'NIST skrf v{rf.__version__}',
                marker='>', markevery=50, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('MAE of effective permittivity')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylim(0, 1.5)  # clip the low-frequency blow-up (1/f^2) so the band of interest is visible
        ax.set_yticks(np.arange(0, 1.51, 0.3))

        ax = axs[1]
        ax.plot(f*1e-9, loss_MAE_NIST, lw=2, label='NIST MultiCal',
                marker='^', markevery=50, markersize=10)
        ax.plot(f*1e-9, loss_MAE_TUG, lw=2, label='TUG mTRL',
                marker='v', markevery=50, markersize=10)
        ax.plot(f*1e-9, loss_MAE_skrf, lw=2, label=f'NIST skrf v{rf.__version__}',
                marker='>', markevery=50, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('MAE of loss (dB/mm)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylim(0, 1.2)
        ax.set_yticks(np.arange(0, 1.21, 0.3))

        handles, labels = ax.get_legend_handles_labels()
        # the figure is short, so place the title clear above the top legend.
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.0),
                   loc='lower center', ncol=3, borderaxespad=0
                   )
        plt.suptitle(f"Mean Absolute Error (MAE) of effective permittivity and loss. Noise std = {sigma_noise:.2f}", verticalalignment='bottom').set_y(1.12)
        #fig.savefig(fig_dir + f'example_3_ereff_loss_{suffix}.png', dpi=600, bbox_inches='tight')

    plt.show()

# EOF
