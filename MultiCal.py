# -*- coding: utf-8 -*-
"""
@author: Ziad Hatab (zi.hatab@gmail.com)

This is an implementation of the MultiCal multiline TRL calibration algorithm 
discussed in [1]. The original algorithm is from [2].

Some concerns regarding [1]:
    1. on page 139, the paragraph after Eq. (11). It says that "MultiCal sets 
    phieff to 0-deg if the argument of the arcsine happens to exceed 1 due 
    to measurement noise". This sounds counter-intuitive to me, because you 
    would think that phieff should be set to 90-deg if the argument of arcsine 
    exceeds 1 (i.e., saturation). And when you look at [2], on page 1209, 
    the paragraph after Eq. (53) reads "We stipulate that phieff = 90-deg if 
    the argument of the arcsin is greater than 1". This statement makes more 
    sense to me than that in [1]. The question now: is the statement about 
    phieff in [1] a typo? I implemented both, see the function commonLine().
    
    2. On page 148, at Eq. (48). In the numerator of the first sum term 
    it says: exp(-g*(lm-lc))*exp(-g*(lm-lc)).conj(). This is definitely a typo. 
    The correct term should be exp(-g*(lm-lc))*exp(-g*(ln-lc)).conj(). 
    The typo is the line length in the second term, it should be ln not lm. 
    The original equation can be found in [2] at page 1211, Eq. (76). There it 
    is written correctly.
    
    3. On page 150, Eq. (57), the equation has a sign error. The second 
    fraction should have a plus sign in both numerator and denominator. 

    4. On page 151, Eqs. (62) and (63) are used to separate the 7-th error 
    term into two parts for each port. I tried using Eqs. (62) and (63) multiple 
    times and it never worked properly. When I take their product to compute R1R2 
    and use it in the calibration, the result of the calibration is wrong.
    Am I missing something here? I ended up deriving an equation for R1R2 
    directly from the Thru measurements (which works quite good).
    
    
    
[1] D. C. DeGroot, J. A. Jargon and R. B. Marks, "Multiline TRL revealed," 
60th ARFTG Conference Digest, Fall 2002, pp. 131-155

[2] R. B. Marks, "A multiline method of network analyzer calibration", 
IEEE Transactions on Microwave Theory and Techniques, 
vol. 39, no. 7, pp. 1205-1215, July 1991.

##########-NOTE-##########
This script is written to process only one frequency point. Therefore, you need 
to call this script in your main script and iterate through all frequency points.
##########-END-##########

"""

import numpy as np

c0 = 299792458   # speed of light in vacuum (m/s)

def S2T(S):
    # convert S- to T-parameters at a single frequency point
    T = S.copy()
    T[0,0] = -(S[0,0]*S[1,1]-S[0,1]*S[1,0])
    T[0,1] = S[0,0]
    T[1,0] = -S[1,1]
    T[1,1] = 1
    return T/S[1,0]

def T2S(T):
    # convert T- to S-parameters at a single frequency point
    S = T.copy()
    S[0,0] = T[0,1]
    S[0,1] = T[0,0]*T[1,1]-T[0,1]*T[1,0]
    S[1,0] = 1
    S[1,1] = -T[1,0]
    return S/T[1,1]

def deleteDiag(A):
    # delete diagonal elements
    return A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)

def commonLine(gamma, line_lengths):
    # select a common line
    exps = np.exp(gamma*line_lengths)
    A = abs(np.outer(exps,1/exps) - np.outer(1/exps,exps))/2
    A = deleteDiag(A)  # remove the same line differences
    
    # for this, please read comment #1 in the header of this file!
    # A[A > 1] = 0         # anything above 1 is set to 0 (see [1]): arcsin(0)=0
    A[A > 1] = 1         # anything above 1 is set to 1 (see [2]): arcsin(1)=pi/2
    
    phieff = np.arcsin(A)
    min_phieff = np.min(phieff, axis=1)  # find minimum phase in every group
    
    return np.argmax(min_phieff)  # the group that has maximum minimum phase is chosen

def BLUE(x,y,Vinv):
    # perform the Gauss-Markov linear estimation
    
    #return (x.transpose().dot(Vinv).dot(y))/(x.transpose().dot(Vinv).dot(x))
    return (x.conj()*(Vinv@y)).sum()/(x.conj()*(Vinv@x)).sum()

def VGamma(N):
    # inverse covariance matrix for gamma
    return np.eye(N-1, dtype=complex) - (1/N)*np.ones(shape=(N-1, N-1), dtype=complex)

def VCA(gamma, cline, lines):
    # inverse covariance matrix for C/A coefficient
    g = gamma
    l  = lines
    lc = cline
    dl = l-lc
    
    lm, ln = np.meshgrid(lines, lines, sparse=False, indexing='ij')
    
    V = ( 1/(np.exp(-g*(lm-lc))*np.exp(-g*(ln-lc)).conj())
         + 1/(abs(np.exp(-g*lc))**2*np.exp(-g*lm)*np.exp(-g*ln).conj()) )/( np.exp(-g*(lm-lc)) - np.exp(g*(lm-lc)) )/( np.exp(-g*(ln-lc)) - np.exp(g*(ln-lc)) ).conj()
    
    # Account for the terms with kronecker-delta (see [2], Eq.(77))
    diag = (abs(np.exp(-g*dl))**2 + 1/(abs(np.exp(-g*l))*abs(np.exp(-g*lc)))**2)/abs(np.exp(-g*dl) - np.exp(g*dl))**2

    V = V + np.diag(diag)
    
    return np.linalg.pinv(V)

def VB(gamma, cline, lines):
    # inverse covariance matrix for B coefficient
    g = gamma
    l  = lines
    lc = cline
    dl = l-lc

    lm, ln = np.meshgrid(lines, lines, sparse=False, indexing='ij')
    
    V = (np.exp(-g*(lm-lc))*np.exp(-g*(ln-lc)).conj() 
         + abs(np.exp(-g*lc))**2*np.exp(-g*lm)*np.exp(-g*ln).conj())/( np.exp(-g*(lm-lc)) - np.exp(g*(lm-lc)) )/( np.exp(-g*(ln-lc)) - np.exp(g*(ln-lc)) ).conj()
    
    # Account for the terms with kronecker-delta (see [2], Eq.(76))
    diag = (1/abs(np.exp(-g*dl))**2 + (abs(np.exp(-g*l))*abs(np.exp(-g*lc)))**2)/abs(np.exp(-g*dl) - np.exp(g*dl))**2

    V = V + np.diag(diag)
    
    return np.linalg.pinv(V)

def computeGL(Mc, Mlines, cline, lines, g_est):
    # Perform root choice procedure as described in [1]
    # Compute the vectors G and L used to estimate gamma (see [1], Eq. (24))
    
    # The work at https://github.com/simonary/MultilineTRL was very helpful.
    
    gdl = []
    dls = []
    for inx, (linej, Mj) in enumerate(zip(lines, Mlines)): # For every other line
        Mjc = Mj@np.linalg.pinv(Mc)
        
        Eij = np.linalg.eigvals(Mjc)
        
        dl = linej - cline
        
        # case 1  Eij[0] = exp(-gamma*l), Eij[1] = exp(gamma*l)
        Ea1 = (Eij[0]+1/Eij[1])/2 # exp(-gamma*l)
        Eb1 = (Eij[1]+1/Eij[0])/2 # exp(gamma*l)
        
        Pa1 = np.round( (g_est*dl + np.log(Ea1)).imag/2/np.pi )
        g_a1 = (-np.log(Ea1) + 1j*2*np.pi*Pa1)/dl
        Da1 = abs(g_a1/g_est - 1)
        
        Pb1 = np.round( -(g_est*dl - np.log(Eb1)).imag/2/np.pi )
        g_b1 = (-np.log(Eb1) + 1j*2*np.pi*Pb1)/dl
        Db1 = abs(g_b1/g_est + 1)
        
        # case 2  Eij[1] = exp(-gamma*l), Eij[0] = exp(gamma*l)
        Ea2 = (Eij[1] + 1/Eij[0])/2 # exp(-gamma*l)
        Eb2 = (Eij[0] + 1/Eij[1])/2 # exp(gamma*l)
        
        Pa2 = np.round( (g_est*dl + np.log(Ea2)).imag/2/np.pi )
        g_a2 = (-np.log(Ea2) + 1j*2*np.pi*Pa2)/dl
        Da2 = abs(g_a2/g_est - 1)
        
        Pb2 = np.round( -(g_est*dl - np.log(Eb2)).imag/2/np.pi )
        g_b2 = (-np.log(Eb2) + 1j*2*np.pi*Pb2)/dl
        Db2 = abs(g_b2/g_est + 1) 

        dls.append(dl)
        
        # Determin the assignment of eigenvalue
        if (Da1 + Db1) <= 0.1*(Da2 + Db2):
            gdl.append(-np.log(Ea1) + 1j*2*np.pi*Pa1)
            
        elif (Da2 + Db2) <= 0.1*(Da1 + Db1):
            gdl.append(-np.log(Ea2) + 1j*2*np.pi*Pa2)

        else:
            if (g_a1 + g_b1).real >= 0 and (g_a2 + g_b2).real < 0:
                gdl.append(-np.log(Ea1) + 1j*2*np.pi*Pa1)
                
            elif (g_a1 + g_b1).real < 0 and (g_a2 + g_b2).real >= 0:
                gdl.append(-np.log(Ea2) + 1j*2*np.pi*Pa2) 

            else: # sign of real part not the same
                if (Da1 + Db1) <= (Da2 + Db2):
                    gdl.append(-np.log(Ea1) + 1j*2*np.pi*Pa1)
                else:
                    gdl.append(-np.log(Ea2) + 1j*2*np.pi*Pa2)
            
    return np.array(gdl), np.array(dls)  # vectors G and L

def compute_B_CA(Mc, Mlines,  cline, lines, g_est, direction='forward'):
    Mcinv = np.linalg.pinv(Mc)
    Bs  = []
    CAs = []
    for inx, Mj in enumerate(Mlines): # For every other line
        if direction == 'forward':
            Eij, V = np.linalg.eig(Mj@Mcinv)
        elif direction == 'backward':
            Eij, V = np.linalg.eig((Mcinv@Mj).T)
        
        dl = lines[inx] - cline
        
        mininx = np.argmin( abs(Eij-np.exp(-g_est*dl)) )
        v1 = V[:,mininx]
        v2 = V[:,~mininx]
        
        CA = v1[1]/v1[0]
        B  = v2[0]/v2[1]
        
        CAs.append(CA)
        Bs.append(B)
    
    return np.array(CAs), np.array(Bs)  # vectors B and C/A

def mTRL(Slines, lengths, Sreflect, gamma_est, reflect_est, reflect_offset, override_gamma=-1):
    # 
    # Slines        : 3D array of 2D S-paramters of line measurements (first is set to Thru)
    # lengths       : 1D array containing line lengths in same order of measurements
    # Sreflect      : 3D array of 2D S-parameters of the measured reflects
    # gamma_est     : Scalar of estimated gamma
    # reflect_est   : 1D array of reference reflection coefficients (e.g., short=-1, open=1)
    # reflect_offset: 1D array of offset lengths of the reflect standards (reference to Thru)
    # 
    
    #  make sure all inputs have proper shape
    Slines         = np.atleast_3d(Slines).reshape((-1,2,2))
    lengths        = np.atleast_1d(lengths)
    Sreflect       = np.atleast_3d(Sreflect).reshape((-1,2,2))
    reflect_est    = np.atleast_1d(reflect_est)
    reflect_offset = np.atleast_1d(reflect_offset)
    
    lengths = lengths - lengths[0]  # setting the first line Thru
    
    Mi     = np.array([S2T(x) for x in Slines]) # convert to T-parameters    
    thru_T = Mi[0]
    N = len(lengths)
    gamma = abs(gamma_est.real) + 1j*abs(gamma_est.imag)
    
    cline_inx = commonLine(gamma, lengths)
    
    # extract the common line out from the list
    line_length_com = lengths[cline_inx]
    line_meas_com   = Mi[cline_inx]
    
    # delete the common line from the list
    line_meas_T  = np.delete(Mi, cline_inx, axis=0)
    line_lengths = np.delete(lengths, cline_inx, axis=0)
    
    # estimate gamma
    if override_gamma != -1:
        gamma = override_gamma
    else:
        G,L = computeGL(line_meas_com, line_meas_T, line_length_com, line_lengths, gamma)
        gamma = BLUE(L, G, VGamma(N))
        # gamma = abs(gamma.real) + 1j*abs(gamma.imag)

    
    # estimate B and C/A for forward direction
    CAs, Bs = compute_B_CA(line_meas_com, line_meas_T, line_length_com,
                           line_lengths, gamma, direction='forward')
    CA1 = BLUE(np.ones(N-1), CAs, VCA(gamma, line_length_com, line_lengths))
    B1  = BLUE(np.ones(N-1), Bs,   VB(gamma, line_length_com, line_lengths))
    
    # estimate B and C/A for backward direction
    CAs, Bs = compute_B_CA(line_meas_com, line_meas_T, line_length_com,
                          line_lengths, gamma, direction='backward')
    CA2 = BLUE(np.ones(N-1), CAs, VCA(gamma, line_length_com, line_lengths))
    B2  = BLUE(np.ones(N-1), Bs,   VB(gamma, line_length_com, line_lengths))
    
    '''
    # The original method MultiCal uses to compute A1A2 and R1R2... not recommended!
    # solve for A1A2 and R1R2 from Thru measurements
    meas_Thru_S = Slines[0]  # S-paramters of the Thru standard
    S11,S12,S21,S22 = meas_Thru_S.flatten()
    A1A2 = -(B1*B2-B1*S22-B2*S11+(S11*S22-S21*S12))/(1-CA1*S11-CA2*S22+CA1*CA2*(S11*S22-S21*S12))
    R1R2 = 1/S21/(CA1*CA2*A1A2 + 1)
    '''
    
    '''
    Below is a mathematical rewriting of the error-box model using 
    Kronecker product formulation. This has no influence on MultiCal procedures, 
    this is just for my convenience (see the NOTE at the end of this file!). 
    You can also write the problem in terms of the conventional 2 error-boxes 
    matrices, if you wish...
    '''
    # normalized calibration matrix
    X_ = np.array([[1,       B1,     B2,     B1*B2], 
                   [CA1,     1,      B2*CA1, B2   ], 
                   [CA2,     B1*CA2, 1,      B1   ], 
                   [CA1*CA2, CA2,    CA1,    1    ]])
    
    # solve for A1A2 and K, simultaneously (this gives much better results)
    KA1A2,_,_,K = np.linalg.pinv(X_)@thru_T.T.flatten()
    A1A2 = KA1A2/K
    
    # solve for A1/A2, A1 and A2
    A1 = []
    A2 = []
    for reflect_S, reflect_ref, offset in zip(Sreflect, reflect_est, reflect_offset):
        G1 = reflect_S[0,0]
        G2 = reflect_S[1,1]
        A1_A2 = (G1 - B1)/(1 - G1*CA1)*(1 + G2*CA2)/(G2 + B2)
        a1 = np.sqrt(A1_A2*A1A2)
        
        # choose correct answer for A1 and A2
        ref_est = reflect_ref*np.exp(-2*gamma*offset)
        G1 = (G1 - B1)/(1 - G1*CA1)/a1
        if abs(G1 - ref_est) > abs(-G1 - ref_est):
            a1 = -a1
        A1.append(a1)
        A2.append(A1A2/a1)
    
    A1 = np.array(A1).mean()
    A2 = np.array(A2).mean()
    
    # de-normalize
    X = X_@np.diag([A1A2, A2, A1, 1])
    
    '''
    X: is 4x4 matrix and is defined as X = (B.T kron A), where A is the 
    forward error-box and B is the reverse error-box (T-parameters). 
    This matrix holds the 6 error terms.
    
    K: is the 7-th term of the error-box model
    
    ############-NOTE-############
    Background on Kronecker product:
        Given three matrices X, Y, and Z, the vectorization of their triplet 
        product is given as:
            vec(XZY) = (Y.T kron X)*vec(Z)
        where vec() flatten a matrix into a vector.
        
        In our case if the error-box model is given as:
            M = K*A*T*B   
        where K is scalar, A and B are forward and backward error-boxes, and 
        T is the DUT. Then using Kronecker product description, we have:
            vec(M) = K*(B.T kron A)*vec(T)
        Therefore, the calibrated DUT is solved as:
            vec(T) = (1/K)*(B.T kron A)^(-1)*vec(M)
        
    https://en.wikipedia.org/wiki/Kronecker_product
    https://en.wikipedia.org/wiki/Vectorization_(mathematics)
    ############-END-############
    '''
    
    return X, K, gamma
    
    # EOF