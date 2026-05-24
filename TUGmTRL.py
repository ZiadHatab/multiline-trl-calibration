"""
@author: Ziad Hatab (zi.hatab@gmail.com)

TUG Multiline TRL Calibration (TUGmTRL)
=======================================

This module implements the improved TUG multiline TRL calibration algorithm
as described in [1,4]. It is designed for optimal combination of all line
measurements using a single, weighted eigenvalue problem. The weighting
matrix is derived via low-rank Takagi decomposition, maximizing the eigengap
and minimizing eigenvector sensitivity. No assumptions are made about the
type of statistical error in the measurements, and no common line is
required---all measurements are combined at once.

Additionally, I modified the Thru normalization step, which is now done with S-parameters 
instead of T-parameters as in done in [5]. Multiple reflect measurements are supported 
and handled via rank-1 recovery. There is also an option to apply scaling to the line measurements 
with repeated lengths and change the L-norm weighting in the eigenvalue problem (see appendix section in [6])

This implementation is distinct from the classical MultiCal algorithm from
NIST [2,3], which solves N-1 eigenvalue problems and combines their results
using a Gauss-Markov estimator. In MultiCal, weights are applied to the
eigenvectors, not directly to the measurements, and a common line is selected
during calibration, which can change across frequencies and cause
discontinuities.

Key Features (TUGmTRL):
- No assumptions about measurement error statistics.
- All line measurements are optimally combined into a single 4×4 weighted
    eigenvalue problem.
- The weighting matrix is derived via Takagi decomposition [4].
- The propagation constant is estimated via linear least squares.
- No common line selection is required.

References:
[1] Z. Hatab, M. Gadringer and W. Bösch, "Improving The Reliability of The
    Multiline TRL Calibration Algorithm," 2022 98th ARFTG Microwave
    Measurement Conference (ARFTG), 2022, pp. 1-5,
    doi: 10.1109/ARFTG52954.2022.9844064.
[2] D. C. DeGroot, J. A. Jargon and R. B. Marks, "Multiline TRL revealed,"
    60th ARFTG Conference Digest, Fall 2002, pp. 131-155,
    doi: 10.1109/ARFTGF.2002.1218696.
[3] R. B. Marks, "A multiline method of network analyzer calibration,"
    IEEE Transactions on Microwave Theory and Techniques, vol. 39, no. 7,
    pp. 1205-1215, July 1991, doi: 10.1109/22.85388.
[4] Z. Hatab, M. E. Gadringer, and W. Bösch, "Propagation of Linear
    Uncertainties through Multiline Thru-Reflect-Line Calibration,"
    IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1-9,
    2023, doi: 10.1109/TIM.2023.3296123.
[5] Z. Hatab, M. E. Gadringer and W. Bösch, "A Thru-Free Multiline Calibration," 
    in IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1-9, 2023, 
    Art no. 1008709, doi: 10.1109/TIM.2023.3308226.
[6] Z. Hatab, M. E. Gadringer, and W. Bösch, "The Choice of Line Lengths in 
    Multiline Thru-Reflect-Line Calibration," 
    arXiv e-print: https://arxiv.org/abs/2512.18641

Note:
-----
This script processes only one frequency point at a time. To calibrate across
a frequency sweep, call this function in your main script and iterate over all
frequency points.
"""

# python -m pip install numpy -U
import numpy as np 

# constants
c0 = 299792458
Q  = np.array([[0,0,0,1], [0,-1,0,0], [0,0,-1,0], [1,0,0,0]])
P  = np.array([[1,0,0,0], [0, 0,1,0], [0,1, 0,0], [0,0,0,1]])
P2 = np.array([[0,1],[1,0]])  # 2x2 permutation matrix

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

def LFT(E, S):
    # Linear fractional transformation
    # R. A. Speciale, "Projective Matrix Transformations in Microwave Network Theory," 
    #        1981 IEEE MTT-S International Microwave Symposium Digest, Los Angeles, CA, USA.
    N = S.shape[0]
    E11, E12, E21, E22 = E[:N,:N], E[:N,N:], E[N:,:N], E[N:,N:]
    return (E11@S + E12)@np.linalg.inv(E21@S + E22)

def LFTinv(E, S):
    # inverse linear fractional transformation
    # R. A. Speciale, "Projective Matrix Transformations in Microwave Network Theory," 
    #        1981 IEEE MTT-S International Microwave Symposium Digest, Los Angeles, CA, USA.
    N = S.shape[0]
    E11, E12, E21, E22 = E[:N,:N], E[:N,N:], E[N:,:N], E[N:,N:]
    return np.linalg.inv(S@E21 - E11)@(E12 - S@E22)

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

def compute_gamma(z, y, lengths, gamma_est, inx=0):
    # gamma = alpha + 1j*beta is determined through linear weighted least-squares    
    lengths = lengths - lengths[inx]
    z = z/z[inx]
    y = y/y[inx]
    del_inx = np.arange(len(lengths)) != inx  # get rid of the reference line (i.e., thru)

    l = -lengths[del_inx]
    gamma_l = np.log((z + 1/y)/2)[del_inx]
    n = np.round( (gamma_l - gamma_est*l).imag/np.pi/2 )
    gamma_l = gamma_l - 1j*2*np.pi*n # unwrap
    gamma = WLS(l, gamma_l, Vgl(len(l)+1))
    
    return gamma.real + 1j*abs(gamma.imag) # ensure delay is positive (causality) in case unwrapping is not perfect.

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
        c2 = np.roots([k2,k1,k0])*np.ones(2)
        c1 = (1 - c2*v22)/v12
    else:
        k2 = -v11*v12*v24/v22 + v11*v14 + v12**2*v21*v24/v22**2 - v12*v14*v21/v22
        k1 = v11*v24/v22 - 2*v12*v21*v24/v22**2 + v12*v23/v22 - v13 + v14*v21/v22
        k0 = v21*v24/v22**2 - v23/v22
        c1 = np.roots([k2,k1,k0])*np.ones(2)
        c2 = (1 - c1*v12)/v22
    x = np.array( [v1*x + v2*y for x,y in zip(c1,c2)] )  # 2 solutions
    mininx = np.argmin( abs(x - x_est).sum(axis=1) )
    return x[mininx]

def mTRL(Slines, lengths, Sreflect, gamma_est, reflect_est, reflect_offset,
         compensate_repeated_lines, lnorm):
    '''  
    Slines         : 3D array of 2D S-parameters of line measurements (first is set to Thru)
    lengths        : 1D array containing line lengths in same order of measurements
    Sreflect       : 3D array of 2D S-parameters of the measured reflects (can be multiple)
    gamma_est      : Scalar, estimated propagation constant.
    reflect_est    : 1D array of reference reflection coefficients
    reflect_offset : Scalar, the offset distance for the reflect measurement relative to the first line (thru).
    compensate_repeated_lines : boolean, apply scaling to the line measurements with repeated lengths.
    lnorm          : int, specify the norm-weighting in the eigenvalue problem. Default is 1 (L1 norm).
    '''
    #  make sure all inputs have proper shape
    Slines         = np.atleast_3d(Slines).reshape((-1,2,2))
    lengths        = np.atleast_1d(lengths)
    Sreflect       = np.atleast_3d(Sreflect).reshape((-1,2,2))
    reflect_est    = np.atleast_1d(reflect_est)
    
    # set the first line Thru
    lengths = lengths - lengths[0]
    
    # measurements
    Mi   = np.array([s2t(x) for x in Slines]) # convert to T-parameters
    M    = np.array([x.flatten('F') for x in Mi]).T
    Dinv = np.diag([1/np.linalg.det(x) for x in Mi])

    ## compute W via Takagi decomposition (also the eigenvalue lambda is computed)
    G, lambd = compute_G_with_takagi(Dinv@M.T@P@Q@M)
    W = (G@np.array([[0,1j],[-1j,0]])@G.T).conj()
    kappa = 2*lambd/abs(W).sum() # this is the normalized eigenvalue without scaling (for effective phase computation)

    ## compute z = exp(-gamma*length) and y = 1/z from matrix G.
    zy = G@np.array([[1,-1j],[1j,1]])@G.T  # could be np.outer(z,y) or np.outer(y,z) depending on the sign of W
    u,_,vh = np.linalg.svd(zy)  # rank-1 recovery
    z = u[:,0]   # ambiguous up to a scaling factor
    y = vh[0,:]  # ambiguous up to a scaling factor
    
    ## pick the sign of W and swap z and y if needed.
    z_est = np.exp(-gamma_est*lengths)
    y_est = 1/z_est
    lambd_est = y_est.dot(W).dot(z_est)  # projection of the estimated z and y onto the weighting matrix W. This is how lambda is defined.
    if abs(lambd_est - lambd) > abs(lambd_est + lambd):
        W = -W
        y, z = z, y  # swap z and y if the sign of W is flipped
    
    ## incorporate scaling to the weighting matrix. See [6] for details.
    # S1: Percentage of occurrence for redundant (duplicate) lengths:
    # e.g., [0, 2, 3, 4, 3] -> [1, 1, 0.5, 1, 0.5]
    _, inv, counts = np.unique(lengths, return_inverse=True, return_counts=True)
    q  = 1/counts[inv]
    S1 = np.outer(q, q) if compensate_repeated_lines else 1
    # S2: change L-norm weighting of the eigenvalue problem (e.g., L1, L2, etc.)
    S2 = abs(W)**(lnorm-1)
    S  = S1*S2 # combined scaling to account for both repeated lines and norm-weighting
    WS  = W*S  # new weighting matrix scaled by S.
    
    # new scaled eigenvalue and normalized eigenvalue after scaling the weighting matrix by S.
    lambd_S = 0.5*abs(WS.conj()*W).sum()
    kappa_S = 2*lambd_S/abs(WS).sum()

    ## weighted eigenvalue problem
    F = M@WS@Dinv@M.T@P@Q
    eigval, eigvec = np.linalg.eig(F)
    inx = np.argsort(abs(eigval))
    # null space
    v2 = eigvec[:,inx[0]]
    v3 = eigvec[:,inx[1]]
    # range space
    v1 = eigvec[:,inx[2]]
    v4 = eigvec[:,inx[3]]
    # eigenvalue from the eigenvalue problem should be same to the one computed from Takagi decomposition.
    lambd_eigval = (eigval[inx[3]] - eigval[inx[2]])/2  
    if abs(lambd_eigval - lambd_S) > abs(lambd_eigval + lambd_S):
        v1, v4 = v4, v1  # swap if assumed order is wrong.
    # build estimates for x1_, x2_, x3_, and x4 from the eigenvectors 
    # these are used as initial estimates for solving the quadratic equations below
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
    # normalized error terms
    A_ = np.array([[1,a12],[a21_a11,1]])
    B_ = np.array([[1,b12_b11],[b21,1]])
    X_ = np.kron(B_.T, A_)
    Zero = np.zeros_like(A_)
    E_ = P.T@np.block([[A_, Zero],[Zero, P2@np.linalg.inv(B_)@P2]])@P # 16-term structure
    
    # recovers S21 = exp(-gamma*length) and factor k^2*a11*b11 from rank-1 recovery from all lines.
    # k^2*a11*b11 not used, as transmission normalization is enforced by the thru measurement.
    Slines_cal = np.array([LFTinv(E_, s) for s in Slines])
    R = np.vstack(( Slines_cal[:, 1, 0], Slines_cal[:, 0, 1]))
    u,_,vh = np.linalg.svd(R)  # rank-1 recovery
    s21 = vh[0,:]/vh[0,0]  # this is exp(-gamma*length) (normalized to the thru)
    # k2a11b11 = u[1,0]/u[0,0]   # k^2*a11*b11

    ## compute propagation constant using error terms
    # method-1: use z, y from the takagi decomposition (matrix G)
    gamma1 = compute_gamma(z, y, lengths, gamma_est)
    # method-2: use z, y from de-embedding the lines
    z, y = s21, 1/s21
    gamma2 = compute_gamma(z, y, lengths, gamma_est)

    # choose which gamma solution approach is more consistent with known lambda.
    z1 = np.exp(-gamma1*lengths)
    y1 = 1/z1
    lambd1 = y1.dot(W).dot(z1)
    z2 = np.exp(-gamma2*lengths)
    y2 = 1/z2
    lambd2 = y2.dot(W).dot(z2)
    if abs(lambd1 - lambd) < abs(lambd2 - lambd):
        gamma = gamma1
    else:
        gamma = gamma2
    
    ## solve for a11b11 and K from Thru measurement. 
    # using S-parameter formulation [5]. Forces S21=S12=1. 
    k = 1/Slines_cal[0,1,0]
    a11b11  = Slines_cal[0,0,1]/k
    
    ## solve for a11 and b11 using the reflect measurement, if available. 
    # otherwise, set a11 = b11 = sqrt(a11b11).
    if np.isnan(Sreflect[0,0,0]):
        # no reflect measurement available
        a11 = np.sqrt(a11b11) 
        b11 = a11
    else:
        # use redundant reflect measurement, if available
        reflect_est = reflect_est*np.exp(-2*gamma*reflect_offset)
        Sreflect_cal = np.array([LFTinv(E_, s) for s in Sreflect])
        R = np.vstack((Sreflect_cal[:, 0, 0], Sreflect_cal[:, 1, 1]))
        u,_,_ = np.linalg.svd(R)   # rank-1 recovery from all reflect measurements (if multiple)
        a11_b11 = u[0,0]/u[1,0]    # this is a11/b11
        a11 = np.sqrt(a11_b11*a11b11)
        b11 = a11b11/a11

        # resolve the sign by comparing estimate to measured reflect.
        G_cal = (Sreflect_cal[:,0,0]/a11 + Sreflect_cal[:,1,1]/b11)/2
        if np.abs(G_cal + reflect_est).sum() < np.abs(G_cal - reflect_est).sum():
            G_cal = -G_cal
            a11   = -a11
            b11   = -b11
        # new reflect estimate for next frequency point.
        reflect_est = G_cal*np.exp(2*gamma*reflect_offset)

    X  = X_@np.diag([a11b11, b11, a11, 1]) # build the calibration matrix (de-normalize)

    return X, k, gamma, reflect_est, lambd, kappa, lambd_S, kappa_S

# EOF