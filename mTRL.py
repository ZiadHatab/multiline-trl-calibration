import MultiCal
import TUGmTRL

import skrf as rf
import numpy as np

c0 = 299792458   # speed of light in vacuum (m/s)

def correct_sw(NW, gamma_f, gamma_r):
    # gamma_f forward
    # gamma_r reverse
    
    G21 = gamma_f.s.squeeze()
    G12 = gamma_r.s.squeeze()
    
    freq  = NW.frequency
    SS = []
    for S,g21,g12 in zip(NW.s,G21,G12):
        S11 = (S[0,0]-S[0,1]*S[1,0]*g21)/(1-S[0,1]*S[1,0]*g21*g12)
        S12 = (S[0,1]-S[0,0]*S[0,1]*g12)/(1-S[0,1]*S[1,0]*g21*g12)
        S21 = (S[1,0]-S[1,1]*S[1,0]*g21)/(1-S[0,1]*S[1,0]*g21*g12)
        S22 = (S[1,1]-S[0,1]*S[1,0]*g12)/(1-S[0,1]*S[1,0]*g21*g12)
        SS.append([[S11,S12],[S21,S22]])
    SS = np.array(SS)
    return rf.Network(frequency=freq, s=SS)

class mTRL:
    """
    Multiline TRL calibration.
    
    Two algorithms implemented here: 
        1. The classical mTRL from NIST (MultiCal) [2,3]
        2. Improved implementation based on [1]

    [1] Ziad Hatab, Michael Gadringer, Wolfgang Boesch, "Improving the Reliability 
    of the Multiline TRL Calibration Algorithm," 98th ARFTG Conference, Jan. 2022
    (accepted for presentation)
    
    [2] D. C. DeGroot, J. A. Jargon and R. B. Marks, "Multiline TRL revealed," 
    60th ARFTG Conference Digest, Fall 2002., 
    2002, pp. 131-155, doi: 10.1109/ARFTGF.2002.1218696.
    
    [3] R. B. Marks, "A multiline method of network analyzer calibration", 
    IEEE Transactions on Microwave Theory and Techniques, 
    vol. 39, no. 7, pp. 1205-1215, July 1991.
    
    """
    
    def __init__(self, lines, line_lengths, reflect, 
                 reflect_est=[-1], reflect_offset=[0], ereff_est=1+0j, switch_term=None):
        """
        mTRL initializer.
        
        Parameters
        --------------
        lines : list of :class:`~skrf.network.Network`
             Measured lines. The first one is defined as Thru, 
             and by default calibration is defined in its middel.
                
        line_lengths : list of float
            Lengths of the line. In the same order as the paramter 'lines'
        
        reflect : list of :class:`~skrf.network.Network`
            Measured reflect standards (2-port device)
            
        reflect_est : list of float
            Estimated reflection coefficient of the reflect standard.
            In the same order as the parameter 'reflect'.
            E.g., if you have a short : [-1]
        
        reflect_offset : list of float
            Offsets of the reflect standards from the reference plane (mid of Thru standard)
            Negative: towards the port
            Positive: away from port
            Units in meters.
        
        ereff_est : complex
            Estimated effective permittivity.
        
        switch_term : list of :class:`~skrf.network.Network`
            list of 1-port networks. Holds 2 elements:
                1. network for forward switch term.
                2. network of reverse switch term.
        """
        
        self.lines = lines
        self.line_lengths = line_lengths
        self.reflect = reflect
        self.reflect_est = reflect_est
        self.reflect_offset = reflect_offset
        self.ereff_est = ereff_est
        self.switch_term = switch_term
        
        # correct for switch terms
        if self.switch_term is not None:
            self.lines = [correct_sw(NT, switch_term[0], switch_term[1]) for NT in self.lines]
            self.reflect = [correct_sw(NT, switch_term[0], switch_term[1]) for NT in self.reflect]
                
    def run_multical(self):
        print('\nMultiCal mTRL in progress:')
        f = self.lines[0].frequency.f
    
        # measurements 
        T_lines = [ rf.s2t(x.s) for x in self.lines]
        S_short = [ x.s for x in self.reflect]
        
        line_lengths = self.line_lengths
        reflect_est  = self.reflect_est
        reflect_offset = self.reflect_offset
        
        # initial arrays to fill
        gamma_full = []
        X_full     = []
        K_full     = []
        
        # initial estimate
        ereff_0  = self.ereff_est
        gamma_0  = 2*np.pi*f[0]/c0*np.sqrt(-ereff_0)
        gamma_0  = abs(gamma_0.real) + 1j*abs(gamma_0.imag)
        # perform the calibration
        for inx, ff in enumerate(f):
            meas_lines_T = [ x[inx] for x in T_lines ]
            meas_reflect_S = [ x[inx] for x in S_short ]
            
            # MultiCal
            X, K, gamma = MultiCal.mTRL(meas_lines_T, line_lengths, meas_reflect_S, 
                                        gamma_0, reflect_est, reflect_offset)
            if inx+1 < len(f):
                gamma_0 = gamma.real + 1j*gamma.imag*f[inx+1]/ff
                
            X_full.append(X)
            K_full.append(K)
            gamma_full.append(gamma)
            print(f'Frequency: {(ff*1e-9).round(4)} GHz done!', end='\r', flush=True)
            
        self.X = np.array(X_full)
        self.K = np.array(K_full)
        self.gamma = np.array(gamma_full)
        
    def run_tug(self):
        print('\nTUG mTRL in progress:')
        f = self.lines[0].frequency.f
    
        # measurements 
        T_lines = [ rf.s2t(x.s) for x in self.lines]
        S_short = [ x.s for x in self.reflect]
        
        line_lengths = self.line_lengths
        reflect_est  = self.reflect_est
        reflect_offset = self.reflect_offset
        
        # initial arrays to fill
        gamma_full = []
        X_full     = []
        K_full     = []
        abs_lambda_full = []
        
        # initial estimate
        ereff_0  = self.ereff_est
        # perform the calibration
        for inx, ff in enumerate(f):
            meas_lines_T = [ x[inx] for x in T_lines ]
            meas_reflect_S = [ x[inx] for x in S_short ]
            
            # TUG mTRL
            X, K, ereff_0, gamma, abs_lambda = TUGmTRL.mTRL(meas_lines_T, line_lengths, 
                                              meas_reflect_S, ereff_0, reflect_est, 
                                              reflect_offset, ff)
            
            X_full.append(X)
            K_full.append(K)
            gamma_full.append(gamma)
            abs_lambda_full.append(abs_lambda)
            print(f'Frequency: {(ff*1e-9).round(4)} GHz done!', end='\r', flush=True)
            
        self.X = np.array(X_full)
        self.K = np.array(K_full)
        self.gamma = np.array(gamma_full)
        self.abs_lambda = np.array(abs_lambda_full)
        
    def apply_cal(self, NW):
        # apply calibration to a 1- or 2-port network
        X = self.X
        K = self.K
                
        if self.switch_term is not None:
            NW = correct_sw(NW, self.switch_term[0], self.switch_term[1])
        
        freq  = NW.frequency
        T     = rf.s2t(NW.s)
        T_cal = np.array([(1/k*np.linalg.pinv(x)@t.T.flatten()).reshape((2,2)).T for x,k,t in zip(X, K, T)])
        
        return rf.Network(frequency=freq, s=rf.t2s(T_cal))
    
    def shift_plane(self):
        # shift calibration plane
        # i will code this later
        pass
    
    def renorm_impedance(self):
        # re-normalize calibration plane
        # i will code this later
        pass
    