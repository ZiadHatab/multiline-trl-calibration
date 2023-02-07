import MultiCal
import TUGmTRL

import skrf as rf
import numpy as np

c0 = 299792458   # speed of light in vacuum (m/s)

def correct_switch_term(S, GF, GR):
    '''
    correct switch terms of measured S-parameters at a single frequency point
    GF: forward (sourced by port-1)
    GR: reverse (sourced by port-2)
    '''
    S_new = S.copy()
    S_new[0,0] = (S[0,0]-S[0,1]*S[1,0]*GF)/(1-S[0,1]*S[1,0]*GF*GR)
    S_new[0,1] = (S[0,1]-S[0,0]*S[0,1]*GR)/(1-S[0,1]*S[1,0]*GF*GR)
    S_new[1,0] = (S[1,0]-S[1,1]*S[1,0]*GF)/(1-S[0,1]*S[1,0]*GF*GR)
    S_new[1,1] = (S[1,1]-S[0,1]*S[1,0]*GR)/(1-S[0,1]*S[1,0]*GF*GR)
    return S_new

def sqrt_unwrapped(z):
    '''
    Take the square root of a complex number with unwrapped phase.
    '''
    return np.sqrt(abs(z))*np.exp(0.5*1j*np.unwrap(np.angle(z)))

    
class mTRL:
    """
    Multiline TRL calibration.
    
    Two algorithms implemented here: 
        1. The classical mTRL from NIST (MultiCal) [2,3]
        2. Improved implementation based on [1]

    [1] Ziad Hatab, Michael Gadringer, Wolfgang Boesch, "Improving the Reliability 
    of the Multiline TRL Calibration Algorithm," 98th ARFTG Conference, Jan. 2022
    
    [2] D. C. DeGroot, J. A. Jargon and R. B. Marks, "Multiline TRL revealed," 
    60th ARFTG Conference Digest, Fall 2002, pp. 131-155
    
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
            Lengths of the line. In the same order as the parameter 'lines'
        
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
                2. network for reverse switch term.
        """
        
        self.f  = lines[0].frequency.f
        self.Slines = np.array([x.s for x in lines])
        self.lengths = np.array(line_lengths)
        self.Sreflect = np.array([x.s for x in (reflect if isinstance(reflect, list) else [reflect]) ])
        self.reflect_est = np.atleast_1d(reflect_est)
        self.reflect_offset = np.atleast_1d(reflect_offset)
        self.ereff_est = ereff_est
        
        if switch_term is not None:
            self.switch_term = np.array([x.s.squeeze() for x in switch_term])
        else:
            self.switch_term = np.array([self.f*0 for x in range(2)])
        
    def run_multical(self):
        # MultiCal
        print('\nMultiCal mTRL in progress:')
        
        # initial arrays to fill
        gammas  = []
        Xs      = []
        ks      = []
        
        lengths = self.lengths
        reflect_est  = self.reflect_est
        reflect_offset = self.reflect_offset
        
        # initial estimate
        ereff0  = self.ereff_est
        gamma0  = 2*np.pi*self.f[0]/c0*np.sqrt(-ereff0)
        
        # perform the calibration
        for inx, f in enumerate(self.f):
            Slines = self.Slines[:,inx,:,:]
            Sreflect = self.Sreflect[:,inx,:,:]
            sw = self.switch_term[:,inx]
            
            # correct switch term
            Slines = [correct_switch_term(x,sw[0],sw[1]) for x in Slines] if np.any(sw) else Slines
            Sreflect = [correct_switch_term(x,sw[0],sw[1]) for x in Sreflect] if np.any(sw) else Sreflect # this is actually not needed!
            
            X, k, gamma = MultiCal.mTRL(Slines, lengths, Sreflect, 
                                        gamma0, reflect_est, reflect_offset)
            if inx+1 < len(self.f):
                gamma0 = gamma.real + 1j*gamma.imag*self.f[inx+1]/f
                
            Xs.append(X)
            ks.append(k)
            gammas.append(gamma)
            print(f'Frequency: {(f*1e-9).round(4)} GHz done!', end='\r', flush=True)
            
        self.X = np.array(Xs)
        self.k = np.array(ks)
        self.gamma = np.array(gammas)
        self.ereff = -(c0/2/np.pi/self.f*self.gamma)**2
        self.error_coef()
        
    def run_tug(self):
        # TUG mTRL
        print('\nTUG mTRL in progress:')
        
        # initial arrays to fill
        gammas  = []
        Xs      = []
        ks      = []
        lambds  = []
        
        lengths = self.lengths
        
        # initial estimate
        ereff0  = self.ereff_est
        reflect0 = self.reflect_est
        reflect_offset = self.reflect_offset
        # perform the calibration
        for inx, f in enumerate(self.f):
            Slines = self.Slines[:,inx,:,:]
            Sreflect = self.Sreflect[:,inx,:,:]
            sw = self.switch_term[:,inx]
            
            # correct switch term
            Slines = [correct_switch_term(x,sw[0],sw[1]) for x in Slines] if np.any(sw) else Slines
            Sreflect = [correct_switch_term(x,sw[0],sw[1]) for x in Sreflect] if np.any(sw) else Sreflect
            
            X, k, ereff0, gamma, _, lambd = TUGmTRL.mTRL(Slines, lengths, Sreflect, ereff0, 
                                                                    reflect0, reflect_offset, f)
            
            Xs.append(X)
            ks.append(k)
            gammas.append(gamma)
            lambds.append(lambd)
            print(f'Frequency: {(f*1e-9).round(4)} GHz done!', end='\r', flush=True)
            
        self.X = np.array(Xs)
        self.k = np.array(ks)
        self.gamma = np.array(gammas)
        self.ereff = -(c0/2/np.pi/self.f*self.gamma)**2
        self.lambd = np.array(lambds)
        self.error_coef()  # compute the 12 error terms
        
    def apply_cal(self, NW, left=True):
        '''
        Apply calibration to a 1-port or 2-port network.
        NW:   the network to be calibrated (1- or 2-port).
        left: boolean: define which port to use when 1-port network is given. If left is True, left port is used; otherwise right port is used.
        '''
        nports = np.sqrt(len(NW.port_tuples)).astype('int') # number of ports
        # if 1-port, convert to 2-port (later convert back to 1-port)
        if nports < 2:
            NW = rf.two_port_reflect(NW)
        # apply cal
        S_cal = []
        for x,k,s,sw in zip(self.X, self.k, NW.s, self.switch_term.T):
            s    = correct_switch_term(s, sw[0], sw[1]) if np.any(sw) else s
            xinv = np.linalg.pinv(x)
            M_ = np.array([-s[0,0]*s[1,1]+s[0,1]*s[1,0], -s[1,1], s[0,0], 1])
            T_ = xinv@M_
            s21_cal = k*s[1,0]/T_[-1]
            T_ = T_/T_[-1]
            S_cal.append([[T_[2], (T_[0]-T_[2]*T_[1])/s21_cal],[s21_cal, -T_[1]]])
        S_cal = np.array(S_cal)
        freq  = NW.frequency
        
        # revert to 1-port device if the input was a 1-port device
        if nports < 2:
            if left: # left port
                S_cal = S_cal[:,0,0]
            else:  # right port
                S_cal = S_cal[:,1,1]
        return rf.Network(frequency=freq, s=S_cal.squeeze())
    
    def error_coef(self):
        '''
        This function returns the conventional 12 error terms from the error-box model. The conversion equations used are adapted from references [4] and [5].
        Initially, only the 3 error terms from each port were included. However, due to feedback from @Zwelckovich, the function has been updated to now return 
        all 12 error terms. Additionally, for the sake of completeness, the switch terms have also been included. 
        Furthermore, the function also includes a consistency test between the 8-terms and 12-terms models, as discussed in reference [4].
        
        [4] R. B. Marks, "Formulations of the Basic Vector Network Analyzer Error Model including Switch-Terms," 50th ARFTG Conference Digest, 1997, pp. 115-126.
        [5] Dunsmore, J.P.. Handbook of Microwave Component Measurements: with Advanced VNA Techniques.. Wiley, 2020.

        The following list includes the full error term abbreviations. In reference [4], Marks used the abbreviations without providing their full forms, 
        which can be challenging to understand for those unfamiliar with VNA calibration terminology. 
        For a comprehensive understanding of VNAs, I recommend consulting the book by Dunsmore [5], where all the terms are listed in full.
        
        Left port error terms (forward direction):
        EDF: forward directivity
        ESF: forward source match
        ERF: forward reflection tracking
        ELF: forward load match
        ETF: forward transmission tracking
        EXF: forward crosstalk
        
        Right port error terms (reverse direction):
        EDR: reverse directivity
        ESR: reverse source match
        ERR: reverse reflection tracking
        ELR: reverse load match
        ETR: reverse transmission tracking
        EXR: reverse crosstalk
        
        Switch terms:
        GF: forward switch term
        GR: reverse switch term

        NOTE: the k in my notation is equivalent to Marks' notation [4] by this relationship: k = (beta/alpha)*(1/ERR).
        '''

        self.coefs = {}
        # forward 3 error terms. These equations are directly mapped from eq. (3) in [4]
        EDF =  self.X[:,2,3]
        ESF = -self.X[:,3,2]
        ERF =  self.X[:,2,2] - self.X[:,2,3]*self.X[:,3,2]
        
        # reverse 3 error terms. These equations are directly mapped from eq. (3) in [4]
        EDR = -self.X[:,1,3]
        ESR =  self.X[:,3,1]
        ERR =  self.X[:,1,1] - self.X[:,3,1]*self.X[:,1,3]
        
        # switch terms
        GF = self.switch_term[0]
        GR = self.switch_term[1]

        # remaining forward terms
        ELF = ESR + ERR*GF/(1-EDR*GF)  # eq. (36) in [4].
        ETF = 1/self.k/(1-EDR*GF)      # eq. (38) in [4], after substituting eq. (36) in eq. (38) and simplifying.
        EXF = 0*ESR  # setting it to zero, since we assumed no cross-talk in the calibration. (update if known!)

        # remaining reverse terms
        ELR = ESF + ERF*GR/(1-EDF*GR)    # eq. (37) in [4].
        ETR = self.k*ERR*ERF/(1-EDF*GR)  # eq. (39) in [4], after substituting eq. (37) in eq. (39) and simplifying.
        EXR = 0*ESR  # setting it to zero, since we assumed no cross-talk in the calibration. (update if known!)

        # forward direction
        self.coefs['EDF'] = EDF
        self.coefs['ESF'] = ESF
        self.coefs['ERF'] = ERF
        self.coefs['ELF'] = ELF
        self.coefs['ETF'] = ETF
        self.coefs['EXF'] = EXF
        self.coefs['GF']  = GF

        # reverse direction
        self.coefs['EDR'] = EDR
        self.coefs['ESR'] = ESR
        self.coefs['ERR'] = ERR
        self.coefs['ELR'] = ELR
        self.coefs['ETR'] = ETR
        self.coefs['EXR'] = EXR
        self.coefs['GR']  = GR

        # consistency check between 8-terms and 12-terms model. Based on eq. (35) in [4].
        # This should equal zero, otherwise there is inconsistency between the models (can arise from bad switch term measurements).
        self.coefs['check'] = abs( ETF*ETR - (ERR + EDR*(ELF-ESR))*(ERF + EDF*(ELR-ESF)) )
        return self.coefs 

    def reciprocal_ntwk(self):
        '''
        Return left and right error-boxes as skrf networks, assuming they are reciprocal.
        '''
        freq = rf.Frequency.from_f(self.f, unit='hz')
        freq.unit = 'ghz'

        # left error-box
        S11 = self.coefs['EDF']
        S22 = self.coefs['ESF']
        S21 = sqrt_unwrapped(self.coefs['ERF'])
        S12 = S21
        S = np.array([ [[s11,s12],[s21,s22]] for s11,s12,s21,s22 
                                in zip(S11,S12,S21,S22) ])
        left_ntwk = rf.Network(s=S, frequency=freq, name='Left error-box')
        
        # right error-box
        S11 = self.coefs['EDR']
        S22 = self.coefs['ESR']
        S21 = sqrt_unwrapped(self.coefs['ERR'])
        S12 = S21
        S = np.array([ [[s11,s12],[s21,s22]] for s11,s12,s21,s22 
                                in zip(S11,S12,S21,S22) ])
        right_ntwk = rf.Network(s=S, frequency=freq, name='Right error-box')
        right_ntwk.flip()
        return left_ntwk, right_ntwk
    
    def shift_plane(self, d=0):
        '''
        Shift calibration plane by a distance d.
        Negative d value shifts toward port, while positive d value shift away from port.
        For example, if your Thru has a length of L, then d=-L/2 shifts the plane backward to the edges of the Thru.
        '''
        X_new = []
        K_new = []
        for x,k,g in zip(self.X, self.k, self.gamma):
            z = np.exp(-g*d)
            KX_new = k*x@np.diag([z**2, 1, 1, 1/z**2])
            X_new.append(KX_new/KX_new[-1,-1])
            K_new.append(KX_new[-1,-1])
        self.X = np.array(X_new)
        self.k = np.array(K_new)
    
    def renorm_impedance(self, Z_new, Z0=50):
        '''
        Re-normalize reference calibration impedance. by default, the ref impedance is the characteristic 
        impedance of the line standards (even if you don'y know it!).
        Z_new: new ref. impedance (can be array if frequency dependent)
        Z0: old ref. impedance (can be array if frequency dependent)
        '''
        # ensure correct array dimensions if scalar is given (frequency independent).
        N = len(self.k)
        Z_new = Z_new*np.ones(N)
        Z0    = Z0*np.ones(N)
        
        G = (Z_new-Z0)/(Z_new+Z0)
        X_new = []
        K_new = []
        for x,k,g in zip(self.X, self.k, G):
            KX_new = k*x@np.kron([[1, -g],[-g, 1]],[[1, g],[g, 1]])/(1-g**2)
            X_new.append(KX_new/KX_new[-1,-1])
            K_new.append(KX_new[-1,-1])

        self.X = np.array(X_new)
        self.k = np.array(K_new)

# EOF