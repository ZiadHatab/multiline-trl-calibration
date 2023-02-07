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

def s2t(S, pseudo=False):
    T = S.copy()
    T[0,0] = -(S[0,0]*S[1,1]-S[0,1]*S[1,0])
    T[0,1] = S[0,0]
    T[1,0] = -S[1,1]
    T[1,1] = 1
    return T if pseudo else T/S[1,0]

def t2s(T, pseudo=False):
    S = T.copy()
    S[0,0] = T[0,1]
    S[0,1] = T[0,0]*T[1,1]-T[0,1]*T[1,0]
    S[1,0] = 1
    S[1,1] = -T[1,0]
    return S if pseudo else S/T[1,1]

def compute_G_with_takagi(A):
    # implementation of Takagi decomposition to compute the matrix G used to determine the weighting matrix.
    # Singular value decomposition for the Takagi factorization of symmetric matrices
    # https://www.sciencedirect.com/science/article/pii/S0096300314002239
    u,s,vh = np.linalg.svd(A)
    u,s,vh = u[:,:2],s[:2],vh[:2,:]  # low-rank truncated (Eckart-Young-Mirsky theorem)
    phi = np.sqrt( s*np.diag(vh@u.conj()) )
    G = u@np.diag(phi)
    lambd = s[0]*s[1]  # this is the eigenvalue of the weighted eigenvalue problem (squared Frobenius norm of W)
    return G, lambd

def WLS(x,y,w=1):
    # Weighted least-squares for a single parameter estimation
    x = x*(1+0j) # force x to be complex type 
    return (x.conj().dot(w).dot(y))/(x.conj().dot(w).dot(x))

def Vgl(N):
    # inverse covariance matrix for propagation constant computation
    return np.eye(N-1, dtype=complex) - (1/N)*np.ones(shape=(N-1, N-1), dtype=complex)

def compute_gamma(X_inv, M, lengths, gamma_est, inx=0):
    # gamma = alpha + 1j*beta is determined through linear weighted least-squares    
    lengths = lengths - lengths[inx]
    EX = (X_inv@M)[[0,-1],:]                  # extract z and y columns
    EX = np.diag(1/EX[:,inx])@EX              # normalize to a reference line based on index `inx` (can be any)
    
    del_inx = np.arange(len(lengths)) != inx  # get rid of the reference line (i.e., thru)
    
    # solve for alpha
    l = -2*lengths[del_inx]
    gamma_l = np.log(EX[0,:]/EX[-1,:])[del_inx]
    alpha =  WLS(l, gamma_l.real, Vgl(len(l)+1))
    
    # solve for beta
    l = -lengths[del_inx]
    gamma_l = np.log((EX[0,:] + 1/EX[-1,:])/2)[del_inx]
    n = np.round( (gamma_l - gamma_est*l).imag/np.pi/2 )
    gamma_l = gamma_l - 1j*2*np.pi*n # unwrap
    beta = WLS(l, gamma_l.imag, Vgl(len(l)+1))
    
    return alpha + 1j*beta 

def solve_quadratic(v1, v2, inx, x_est):
    # inx contain index of the unit value and product 
    v12,v13 = v1[inx]
    v22,v23 = v2[inx]
    mask = np.ones(v1.shape, bool)
    mask[inx] = False
    v11,v14 = v1[mask]
    v21,v24 = v2[mask]
    if abs(v12) > abs(v22):  # to avoid dividing by small numbers
        k2 = -v11*v22*v24/v12 + v11*v14*v22**2/v12**2 + v21*v24 - v14*v21*v22/v12
        k1 = v11*v24/v12 - 2*v11*v14*v22/v12**2 - v23 + v13*v22/v12 + v14*v21/v12
        k0 = v11*v14/v12**2 - v13/v12
        c2 = np.array([(-k1 - np.sqrt(-4*k0*k2 + k1**2))/(2*k2), (-k1 + np.sqrt(-4*k0*k2 + k1**2))/(2*k2)])
        c1 = (1 - c2*v22)/v12
    else:
        k2 = -v11*v12*v24/v22 + v11*v14 + v12**2*v21*v24/v22**2 - v12*v14*v21/v22
        k1 = v11*v24/v22 - 2*v12*v21*v24/v22**2 + v12*v23/v22 - v13 + v14*v21/v22
        k0 = v21*v24/v22**2 - v23/v22
        c1 = np.array([(-k1 - np.sqrt(-4*k0*k2 + k1**2))/(2*k2), (-k1 + np.sqrt(-4*k0*k2 + k1**2))/(2*k2)])
        c2 = (1 - c1*v12)/v22
    x = np.array( [v1*x + v2*y for x,y in zip(c1,c2)] )  # 2 solutions
    mininx = np.argmin( abs(x - x_est).sum(axis=1) )
    return x[mininx]

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
    Mi   = np.array([s2t(x) for x in Slines]) # convert to T-parameters
    M    = np.array([x.flatten('F') for x in Mi]).T
    Dinv = np.diag([1/np.linalg.det(x) for x in Mi])

    ## Compute W via Takagi decomposition (also the eigenvalue lambda is computed)
    G, lambd = compute_G_with_takagi(Dinv@M.T@P@Q@M)
    W = (G@np.array([[0,1j],[-1j,0]])@G.T).conj()

    gamma_est = 2*np.pi*f/c0*np.sqrt(-ereff_est)
    gamma_est = abs(gamma_est.real) + 1j*abs(gamma_est.imag)  # this to avoid sign inconsistencies 
        
    z_est = np.exp(-gamma_est*lengths)
    y_est = 1/z_est
    W_est = (np.outer(y_est,z_est) - np.outer(z_est,y_est)).conj()
    W = -W if abs(W-W_est).sum() > abs(W+W_est).sum() else W # resolve the sign ambiguity
    
    ## weighted eigenvalue problem
    F = M@W@Dinv@M.T@P@Q
    eigval, eigvec = np.linalg.eig(F+lambd*np.eye(4))
    inx = np.argsort(abs(eigval))
    v1 = eigvec[:,inx[0]]
    v2 = eigvec[:,inx[1]]
    v3 = eigvec[:,inx[2]]
    v4 = eigvec[:,inx[3]]
    x1__est = v1/v1[0]
    x1__est[-1] = x1__est[1]*x1__est[2]
    x4_est = v4/v4[-1]
    x4_est[0] = x4_est[1]*x4_est[2]
    x2__est = np.array([x4_est[2], 1, x4_est[2]*x1__est[2], x1__est[2]])
    x3__est = np.array([x4_est[1], x4_est[1]*x1__est[1], 1, x1__est[1]])
    
    # solve quadratic equation for each column
    x1_ = solve_quadratic(v1, v4, [0,3], x1__est)
    x2_ = solve_quadratic(v2, v3, [1,2], x2__est)
    x3_ = solve_quadratic(v2, v3, [2,1], x3__est)
    x4  = solve_quadratic(v1, v4, [3,0], x4_est)
    
    # build the normalized cal coefficients (average the answers from range and null spaces)    
    a12 = (x2_[0] + x4[2])/2
    b21 = (x3_[0] + x4[1])/2
    a21_a11 = (x1_[1] + x3_[3])/2
    b12_b11 = (x1_[2] + x2_[3])/2
    X_  = np.kron([[1,b21],[b12_b11,1]], [[1,a12],[a21_a11,1]])
    
    X_inv = np.linalg.inv(X_)
    ## Compute propagation constant
    gamma = compute_gamma(X_inv, M, lengths, gamma_est)
    ereff = -(c0/2/np.pi/f*gamma)**2
    
    # solve for a11b11 and K from Thru measurement
    ka11b11,_,_,k = X_inv@M[:,0]
    a11b11 = ka11b11/k
    
    # solve for a11/b11, a11 and b11 (use redundant reflect measurement, if available)
    reflect_est = reflect_est*np.exp(-2*gamma*reflect_offset)
    Mr = np.array([s2t(x, pseudo=True).flatten('F') for x in Sreflect]).T
    T  = X_inv@Mr
    a11_b11 = -T[2,:]/T[1,:]
    a11 = np.sqrt(a11_b11*a11b11)
    b11 = a11b11/a11
    G_cal = ( (Sreflect[:,0,0] - a12)/(1 - Sreflect[:,0,0]*a21_a11)/a11 + (Sreflect[:,1,1] + b21)/(1 + Sreflect[:,1,1]*b12_b11)/b11 )/2  # average
    for inx,(Gcal,Gest) in enumerate(zip(G_cal, reflect_est)):
        if abs(Gcal - Gest) > abs(Gcal + Gest):
            a11[inx]   = -a11[inx]
            b11[inx]   = -b11[inx]
            G_cal[inx] = -G_cal[inx]
    a11 = a11.mean()
    b11 = b11.mean()
    reflect_est = G_cal*np.exp(2*gamma*reflect_offset)

    X  = X_@np.diag([a11b11, b11, a11, 1]) # build the calibration matrix (de-normalize)

    return X, k, ereff, gamma, reflect_est, lambd

# EOF