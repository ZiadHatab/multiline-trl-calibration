"""
@author: Ziad Hatab (zi.hatab@gmail.com)

This is an implementation of multiline TRL calibration algorithm discussed in [1].
The implementation is different from the MultiCal algorithm from NIST [2,3], which solves N-1 
eigenvalue problems and combines their results using a Gauss-Markov linear estimator. 

The computation of the weighting matrix via Takagi decomposition was discussed in [4].

TUGmTRL:
    - Combines all lines measurement linearly using an optimal weighting matrix.
    - Solves a single eigenvalue problem.
    - Does not make assumptions about noise perturbation and aims to minimize 
    error perturbation on eigenvectors.

MultiCal:
    - Chooses a common line and creates N-1 pairs (N-1 eigenvalue problems).
    - Solves for the eigenvectors of the N-1 eigenvalue problems.
    - Combines the results using a Gauss-Markov linear estimator. 
    Linearity of the eigenvectors with respect to any perturbation must remain linear for this to work!

[1] Z. Hatab, M. Gadringer and W. Bösch, 
"Improving The Reliability of The Multiline TRL Calibration Algorithm," 
2022 98th ARFTG Microwave Measurement Conference (ARFTG), 
Las Vegas, NV, USA, 2022, pp. 1-5, doi: 10.1109/ARFTG52954.2022.9844064.

[2] D. C. DeGroot, J. A. Jargon and R. B. Marks, "Multiline TRL revealed," 
60th ARFTG Conference Digest, Fall 2002., 
2002, pp. 131-155, doi: 10.1109/ARFTGF.2002.1218696.

[3] R. B. Marks, "A multiline method of network analyzer calibration", 
IEEE Transactions on Microwave Theory and Techniques, 
vol. 39, no. 7, pp. 1205-1215, July 1991.

[4] Z. Hatab, M. Gadringer, and W. Bösch, 
"Propagation of Linear Uncertainties through Multiline Thru-Reflect-Line Calibration," 
e-print: https://arxiv.org/abs/2301.09126

##########-NOTE-##########
This script is written to process only one frequency point. Therefore, you need 
to call this script in your main script and iterate through all frequency points.
##########-END-##########
"""

# python -m pip install numpy -U
import numpy as np 

# constants
c0 = 299792458
Q = np.array([[0,0,0,1], [0,-1,0,0], [0,0,-1,0], [1,0,0,0]])
P = np.array([[1,0,0,0], [0, 0,1,0], [0,1, 0,0], [0,0,0,1]])

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

def compute_G_with_takagi(A):
    # implementation of Takagi decomposition to compute the matrix G used to determine the weighting matrix.
    # Singular value decomposition for the Takagi factorization of symmetric matrices
    # https://www.sciencedirect.com/science/article/pii/S0096300314002239
    u,s,vh = np.linalg.svd(A)
    u,s,vh = u[:,:2],s[:2],vh[:2,:]  # low-rank truncated (Eckart-Young-Mirsky theorem)
    phi = np.sqrt( s*np.diag(vh@u.conj()) )
    G = u@np.diag(phi)
    lambd = s[0]*s[1]    # this is the eigenvalue of the weighted eigenvalue problem (interesting, right?)
    return G, lambd

def WLS(x,y,w=1):
    # Weighted least-squares for a single parameter estimation
    x = x*(1+0j) # force x to be complex type 
    # return (x.conj().transpose().dot(w).dot(y))/(x.conj().transpose().dot(w).dot(x))
    return np.dot(x.conj().transpose().dot(w), y)/np.dot(x.conj().transpose().dot(w), x)

def Vgl(N):
    # inverse covariance matrix for propagation constant computation
    return np.eye(N-1, dtype=complex) - (1/N)*np.ones(shape=(N-1, N-1), dtype=complex)

def compute_gamma(X_, M, lengths, gamma_est, inx=0):
    # gamma = alpha + 1j*beta is determined through linear weighted least-squares    
    lengths = lengths - lengths[inx]
    del_inx = np.arange(len(lengths))!=inx  # get rid of the reference line (i.e., thru)
    EX = (np.linalg.inv(X_)@M)[[0,-1],:]   # extract z and y columns
    EX = np.diag(1/EX[:,inx])@EX            # normalize to a reference line based on index `inx` (can be any)
    # solve for alpha
    l = -2*lengths[del_inx]
    gamma_l = np.log(EX[0,:]/EX[-1,:])[del_inx]
    alpha =  WLS(l, gamma_l, Vgl(len(l)+1)).real
    # solve for beta
    l = -lengths[del_inx]
    gamma_l = np.log((EX[0,:] + 1/EX[-1,:])/2)[del_inx]
    n = np.round( (gamma_l - gamma_est*l).imag/np.pi/2 )
    gamma_l = gamma_l - 1j*2*np.pi*n # unwrap
    beta = WLS(l, gamma_l, Vgl(len(l)+1)).imag
    return alpha + 1j*beta 

def mTRL(Slines, lengths, Sreflect, ereff_est, reflect_est, reflect_offset, f):
    '''  
    Slines      : 3D array of 2D S-parameters of line measurements (first is set to Thru)
    lengths     : 1D array containing line lengths in same order of measurements
    Sreflect    : 3D array of 2D S-parameters of the measured reflects (can be multiple)
    ereff_est   : Scalar of estimated ereff 
    reflect_est : 1D array of reference reflection coefficients
    f           : Scalar, single frequency point (Hz)
    '''
    #  make sure all inputs have proper shape
    Slines         = np.atleast_3d(Slines).reshape((-1,2,2))
    lengths        = np.atleast_1d(lengths)
    Sreflect       = np.atleast_3d(Sreflect).reshape((-1,2,2))
    reflect_est    = np.atleast_1d(reflect_est)
    
    lengths = lengths - lengths[0]  # set the first line Thru
    
    # measurements
    Mi   = np.array([S2T(x) for x in Slines]) # convert to T-parameters
    M    = np.array([x.flatten('F') for x in Mi]).T
    Dinv = np.diag([1/np.linalg.det(x) for x in Mi])

    ## Compute W via Takagi decomposition (also the eigenvalue lambda is computed)
    G, lambdG = compute_G_with_takagi(Dinv@M.T@P@Q@M)
    W = (G@np.array([[0,1j],[-1j,0]])@G.T).conj()

    gamma_est = 2*np.pi*f/c0*np.sqrt(-ereff_est)
    gamma_est = abs(gamma_est.real) + 1j*abs(gamma_est.imag)  # this to avoid sign inconsistancies 
        
    z_est = np.exp(-gamma_est*lengths)
    y_est = 1/z_est 
    W_est = (np.outer(y_est,z_est) - np.outer(z_est,y_est)).conj()
    W = -W if abs(W-W_est).sum() > abs(W+W_est).sum() else W # resolve the sign ambiguity

    ## weighted eigenvalue problem
    F = M@W@Dinv@M.T@P@Q
    eigval, eigvec = np.linalg.eig(F)
    inx = np.argsort(abs(eigval))
    inx_null = inx[:2]
    inx = inx[-2:]
    lambd = (eigval[inx[0]] - eigval[inx[-1]])/2
    if abs(lambd - lambdG) > abs(lambd + lambdG):
        lambd = -lambd
        inx = np.flip(inx)
    x4_eig  = eigvec[:,inx[0]]/eigvec[-1,inx[0]]
    x4_eig[0]  = x4_eig[1]*x4_eig[2]
    x1__eig = eigvec[:,inx[-1]]/eigvec[0,inx[-1]]
    x1__eig[-1] = x1__eig[1]*x1__eig[2]
    x2__eig = np.array([x4_eig[2], 1, x4_eig[2]*x1__eig[2], x1__eig[2]])
    x3__eig = np.array([x4_eig[1], x4_eig[1]*x1__eig[1], 1, x1__eig[1]])
    X__eig = np.array([x1__eig, x2__eig, x3__eig, x4_eig]).T  # estimated normalized calibration matrix
    
    # nullspace basis for x2_ and x3_
    v1 = eigvec[:,inx_null[0]]
    v2 = eigvec[:,inx_null[1]]
    v11,v12,v13,v14 = v1
    v21,v22,v23,v24 = v2
    # find x2_ = x2/b11 (2 solutions)
    if abs(v12)+abs(v23) > abs(v22)+abs(v13):  # to avoid dividing by small numbers
        k2 = -v11*v22*v24/v12 + v11*v14*v22**2/v12**2 + v21*v24 - v14*v21*v22/v12
        k1 = v11*v24/v12 - 2*v11*v14*v22/v12**2 - v23 + v13*v22/v12 + v14*v21/v12
        k0 = v11*v14/v12**2 - v13/v12
        c2 = np.roots([k2,k1,k0])*np.array([1,1])
        c1 = (1 - c2*v22)/v12
    else:
        k2 = -v11*v12*v24/v22 + v11*v14 + v12**2*v21*v24/v22**2 - v12*v14*v21/v22
        k1 = v11*v24/v22 - 2*v12*v21*v24/v22**2 + v12*v23/v22 - v13 + v14*v21/v22
        k0 = v21*v24/v22**2 - v23/v22
        c1 = np.roots([k2,k1,k0])*np.array([1,1])
        c2 = (1 - c1*v12)/v22        
    x2_ = np.array( [v1*x + v2*y for x,y in zip(c1,c2)] )  # 2 solutions
    if abs(v12)+abs(v23) < abs(v22)+abs(v13):  # to avoid dividing by small numbers
        k2 = -v11*v23*v24/v13 + v11*v14*v23**2/v13**2 + v21*v24 - v14*v21*v23/v13
        k1 = v11*v24/v13 - 2*v11*v14*v23/v13**2 + v12*v23/v13 - v22 + v14*v21/v13
        k0 = v11*v14/v13**2 - v12/v13
        c4 = np.roots([k2,k1,k0])*np.array([1,1])
        c3 = (1 - c4*v23)/v13
    else:
        k2 = -v11*v13*v24/v23 + v11*v14 + v13**2*v21*v24/v23**2 - v13*v14*v21/v23
        k1 = v11*v24/v23 - v12 - 2*v13*v21*v24/v23**2 + v13*v22/v23 + v14*v21/v23
        k0 = v21*v24/v23**2 - v22/v23
        c3 = np.roots([k2,k1,k0])*np.array([1,1])
        c4 = (1 - c3*v13)/v23    
    x3_ = np.array( [v1*x + v2*y for x,y in zip(c3,c4)] )   # 2 solutions
    # choose the correct answer for x2_ and x3_
    mininx = np.argmin( abs(x2_ - x2__eig).sum(axis=1) )
    x2__null = x2_[mininx]
    mininx = np.argmin( abs(x3_ - x3__eig).sum(axis=1) )
    x3__null = x3_[mininx]
    # build x1_ and x4 from x2_ and x3_
    x1__null = np.array([1, x3__null[3], x2__null[3], x3__null[3]*x2__null[3]])
    x4_null  = np.array([x3__null[0]*x2__null[0], x3__null[0], x2__null[0], 1])
    X__null  = np.array([x1__null, x2__null, x3__null, x4_null]).T  # normalized calibration matrix
    
    X_  = (X__null + X__eig)/2
    x1_ = X_[:,0]
    x2_ = X_[:,1]
    x3_ = X_[:,2]
    x4  = X_[:,3]
    
    ## Compute propagation constant
    gamma = compute_gamma(X_, M, lengths, gamma_est)
    ereff = -(c0/2/np.pi/f*gamma)**2
    
    # solve for a11b11 and K from Thru measurement
    ka11b11,_,_,k = np.linalg.pinv(X_)@M[:,0]
    a11b11 = ka11b11/k
    
    # solve for a11/b11, a11 and b11 (use redundant reflect measurement, if available)
    a11s = []
    b11s = []
    reflect_est_new = []
    reflect_est = reflect_est*np.exp(-2*gamma*reflect_offset)
    for reflect_S, ref_est in zip(Sreflect, reflect_est):
        Ga = reflect_S[0,0]
        Gb = reflect_S[1,1]
        a11_b11 = (Ga - x2_[0])/(1 - Ga*x3_[3])*(1 + Gb*x2_[3])/(Gb + x3_[0])
        a11 = np.sqrt(a11_b11*a11b11)
        b11 = a11b11/a11
        # choose correct answer for a11 and b11
        G0 = ( (Ga - x2_[0])/(1 - Ga*x3_[3])/a11 + (Gb + x3_[0])/(1 + Gb*x2_[3])/b11 )/2  # average
        if abs(G0 - ref_est) > abs(G0 + ref_est):
            a11 = -a11
            b11 = -b11
            G0  = -G0
        a11s.append(a11)
        b11s.append(b11)
        reflect_est_new.append(G0)
    a11 = np.array(a11s).mean()
    b11 = np.array(b11s).mean()
    reflect_est = np.array(reflect_est_new)*np.exp(2*gamma*reflect_offset)

    X  = X_@np.diag([a11b11, b11, a11, 1]) # build the calibration matrix (de-normalize)
        
    return X, k, ereff, gamma, reflect_est, lambd

# EOF