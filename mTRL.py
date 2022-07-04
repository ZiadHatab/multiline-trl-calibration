import MultiCal
import TUGmTRL

import skrf as rf
import numpy as np

c0 = 299792458   # speed of light in vacuum (m/s)

def correct_switch_term(S, G21, G12):
    # correct switch terms of measured S-parameters at a single frequency point
    # G21: forward (sourced by port-1)
    # G12: reverse (sourced by port-2)
    S_new = S.copy()
    S_new[0,0] = (S[0,0]-S[0,1]*S[1,0]*G21)/(1-S[0,1]*S[1,0]*G21*G12)
    S_new[0,1] = (S[0,1]-S[0,0]*S[0,1]*G12)/(1-S[0,1]*S[1,0]*G21*G12)
    S_new[1,0] = (S[1,0]-S[1,1]*S[1,0]*G21)/(1-S[0,1]*S[1,0]*G21*G12)
    S_new[1,1] = (S[1,1]-S[0,1]*S[1,0]*G12)/(1-S[0,1]*S[1,0]*G21*G12)
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
            
            X, K, gamma = MultiCal.mTRL(Slines, lengths, Sreflect, 
                                        gamma0, reflect_est, reflect_offset)
            if inx+1 < len(self.f):
                gamma0 = gamma.real + 1j*gamma.imag*self.f[inx+1]/f
                
            Xs.append(X)
            ks.append(K)
            gammas.append(gamma)
            print(f'Frequency: {(f*1e-9).round(4)} GHz done!', end='\r', flush=True)
            
        self.X = np.array(Xs)
        self.K = np.array(ks)
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
        gamma0  = 2*np.pi*self.f[0]/c0*np.sqrt(-ereff0)
        reflect_est0 = np.array([ x*np.exp(-2*gamma0*y) for x,y in zip(self.reflect_est, self.reflect_offset) ])
        
        # perform the calibration
        for inx, f in enumerate(self.f):
            Slines = self.Slines[:,inx,:,:]
            Sreflect = self.Sreflect[:,inx,:,:]
            sw = self.switch_term[:,inx]
            
            # correct switch term
            Slines = [correct_switch_term(x,sw[0],sw[1]) for x in Slines] if np.any(sw) else Slines
            Sreflect = [correct_switch_term(x,sw[0],sw[1]) for x in Sreflect] if np.any(sw) else Sreflect # this is actually not needed!
            
            # the reflect standard is recursivly updated after the first point
            if inx < 1:
                X, K, ereff0, gamma, reflect_est0, abs_lambda = TUGmTRL.mTRL(Slines, lengths, Sreflect, ereff0, reflect_est0, f)
                reflect_est0 = np.array([ x*np.exp(-2*gamma*y) for x,y in zip(self.reflect_est, self.reflect_offset) ])
            
            X, K, ereff0, gamma, reflect_est0, abs_lambda = TUGmTRL.mTRL(Slines, lengths, 
                                                                         Sreflect, ereff0, 
                                                                         reflect_est0, f)
            
            Xs.append(X)
            ks.append(K)
            gammas.append(gamma)
            lambds.append(abs_lambda)
            print(f'Frequency: {(f*1e-9).round(4)} GHz done!', end='\r', flush=True)
            
        self.X = np.array(Xs)
        self.K = np.array(ks)
        self.gamma = np.array(gammas)
        self.ereff = -(c0/2/np.pi/self.f*self.gamma)**2
        self.abs_lambda = np.array(lambds)
        self.error_coef()
        
    def apply_cal(self, NW, left=True):
        # apply calibration to a 1-port or 2-port network.
        # NW:   the network to be calibrated (1- or 2-port).
        # left: boolean: define which port to use when 1-port network is given
        # if left is True, left port is used; otherwise right port is used.
        
        nports = np.sqrt(len(NW.port_tuples)).astype('int') # number of ports
        # if 1-port, convert to 2-port (later convert back to 1-port)
        if nports < 2:
            NW = rf.two_port_reflect(NW)
        
        # apply cal
        S_cal = []
        for x,k,s,sw in zip(self.X, self.K, NW.s, self.switch_term.T):
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
        # return the 3 error terms of each port
        #
        # R. B. Marks, "Formulations of the Basic Vector Network Analyzer Error Model including Switch-Terms," 
        # 50th ARFTG Conference Digest, 1997, pp. 115-126.
        #
        # left port:
        # ERF: forward reflection tracking
        # EDF: forward directivity
        # ESF: forward source match
        # 
        # right port:
        # ERR: reverse reflection tracking
        # EDR: reverse directivity
        # ESR: reverse source match
        
        X = self.X
        self.coefs = {}
        # forward errors
        self.coefs['ERF'] =  X[:,2,2] - X[:,2,3]*X[:,3,2]
        self.coefs['EDF'] =  X[:,2,3]
        self.coefs['ESF'] = -X[:,3,2]
        
        # reverse errors
        self.coefs['ERR'] =  X[:,1,1] - X[:,3,1]*X[:,1,3]
        self.coefs['EDR'] = -X[:,1,3]
        self.coefs['ESR'] =  X[:,3,1]
        
    def receprical_ntwk(self):
        # left and right error-boxes, assuming they are reciprocal
        freq = rf.Frequency.from_f(self.f)
        
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
        # shift calibration plane by distance d
        # negative: shift toward port
        # positive: shift away from port
        # e.g., if your Thru has a length of L, 
        # then d=-L/2 to shift the plane backward 
        
        X_new = []
        K_new = []
        for x,k,g in zip(self.X, self.K, self.gamma):
            z = np.exp(-g*d)
            KX_new = k*x@np.diag([z**2, 1, 1, 1/z**2])
            X_new.append(KX_new/KX_new[-1,-1])
            K_new.append(KX_new[-1,-1])
            
        self.X = np.array(X_new)
        self.K = np.array(K_new)
    
    def renorm_impedance(self, Z_new, Z0=50):
        # re-normalize reference calibration impedance
        # by default, the ref impedance is the characteristic 
        # impedance of the line standards.
        # Z_new: new ref. impedance (can be array if frequency dependent)
        # Z0: old ref. impedance (can be array if frequency dependent)
        
        # ensure correct array dimensions (if not, you get an error!)
        N = len(self.K)
        Z_new = Z_new*np.ones(N)
        Z0    = Z0*np.ones(N)
        
        G = (Z_new-Z0)/(Z_new+Z0)
        X_new = []
        K_new = []
        for x,k,g in zip(self.X, self.K, G):
            KX_new = k*x@np.kron([[1, -g],[-g, 1]],[[1, g],[g, 1]])/(1-g**2)
            X_new.append(KX_new/KX_new[-1,-1])
            K_new.append(KX_new[-1,-1])

        self.X = np.array(X_new)
        self.K = np.array(K_new)
    