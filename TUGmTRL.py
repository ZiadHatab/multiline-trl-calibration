"""
@author: Ziad Hatab (zi.hatab@gmail.com)

This is an implementation of multiline TRL calibration algorithm discussed in [1]. 
Developed at:
    Institute of Microwave and Photonic Engineering,
    Graz University of Technology (TU-Graz), Austria 
    
The core principle of this algorithm is to combine all lines in one matrix 
and optimally weight them. This yields finally to a single eigenvalue problem.

This procedure is different from the one done in MultiCal [2,3], where N-1 eigenvalue 
problems are solved and their result is summed using Gauss-Markov linear estimator.

In summery, we can compare our approach and MultiCal as follow:
    TUGmTRL [1]: 
        Combine all lines measurement linearly using an optimal weighting matrix 
            --> Solve a single eigenvalue problem
            --> No assumption made about noise perturbation. 
                The weighting matrix was derived to maximize the "eigengap" of 
                the eigenvalue problem, hence minimizing error perturbation 
                on eigenvectors, in general.
                https://en.wikipedia.org/wiki/Eigenvalue_perturbation
                https://en.wikipedia.org/wiki/Eigengap
            
    MultiCal [2,3]: 
        Choose a common line and create N-1 pairs (N-1 eigenvalue problems)
            --> Solve the N-1 eigenvalue problems
            --> Combine the result of all N-1 pairs using Gauss-Markov linear 
                estimator (requires inverse of a covariance matrix).
            --> The Gauss-Markov linear estimator is only valid under assumption 
                of small noise perturbation (first-order). That is why MultiCal 
                can only operate reliably under impact of low disturbances.
                (MultiCal suffers especially from phase error).

[1] Ziad Hatab, Michael Gadringer, Wolfgang Boesch, "Improving the Reliability 
of the Multiline TRL Calibration Algorithm," 98th ARFTG Conference, Jan. 2022

[2] D. C. DeGroot, J. A. Jargon and R. B. Marks, "Multiline TRL revealed," 
60th ARFTG Conference Digest, Fall 2002., 
2002, pp. 131-155, doi: 10.1109/ARFTGF.2002.1218696.

[3] R. B. Marks, "A multiline method of network analyzer calibration", 
IEEE Transactions on Microwave Theory and Techniques, 
vol. 39, no. 7, pp. 1205-1215, July 1991.

##########-NOTE-##########
This script is written to process only one frequency point. Therefore, you need 
to call this script in your main script and iterate through all frequency points.
##########-END-##########
"""

# third party, need to be installed via pip install ... 
import numpy as np     # for array and math stuff (similar to matlab)
import scipy.optimize as so   # optimization package (for gamma)
import numpy.polynomial.polynomial as poly # to solve polynomial equation

# constants
c0 = 299792458
Q = np.array([[0,0,0,1], [0,-1,0,0], [0,0,-1,0], [1,0,0,0]])
P = np.array([[1,0,0,0], [0, 0,1,0], [0,1, 0,0], [0,0,0,1]])

def findereff(x, *argv):
    # objective function to estimate ereff
    meas = argv[0]
    line_lengths = argv[1]
    w = 2*np.pi*argv[2]
    ereff = x[0] + 1j*x[1]
    gamma = w/c0*np.sqrt(-ereff)
    ex = np.exp(gamma*line_lengths)
    E = np.outer(ex, 1/ex)
    model = E + E.T
    error = meas - model
    return (error*error.conj()).real.sum() # np.linalg.norm(error, ord='fro')**2
    #return np.linalg.norm(error, ord='nuc')

def mTRL(meas_lines_T, line_lengths, meas_reflect_S, ereff_est, reflect_est, reflect_offset, f, override_gamma=-1):
    # 
    # meas_lines_T  : 3D array of 2D T-paramters of line measurements (first is set to Thru)
    # line_lengths  : 1D array containing line lengths in same order of measurements
    # meas_reflect_S: 3D array of 2D S-parameters of the measured reflects
    # ereff_est     : Scalar of estimated ereff
    # reflect_est   : 1D array of reference reflection coefficients (e.g., short=-1, open=1)
    # reflect_offset: 1D array of offset lengths of the reflect standards (reference to Thru)
    # f             : Scalar, single frequency point (Hz)
    
    #  make sure all inputs have proper shape
    meas_lines_T   = np.atleast_3d(meas_lines_T).reshape((-1,2,2))
    line_lengths   = np.atleast_1d(line_lengths)
    meas_reflect_S = np.atleast_3d(meas_reflect_S).reshape((-1,2,2))
    reflect_est    = np.atleast_1d(reflect_est)
    reflect_offset = np.atleast_1d(reflect_offset)
    
    line_lengths = np.array(line_lengths) - line_lengths[0]  # set first to Thru
    
    # measurements
    M = np.array([x.T.flatten() for x in meas_lines_T]).T
    Dinv = np.diag([1/np.linalg.det(x) for x in meas_lines_T])
    
    # estimate gamma
    if override_gamma != -1:
        gamma = override_gamma
        ereff = -(c0/2/np.pi/f*gamma)**2
    else:
        #options={'rhobeg': 1.0, 'maxiter': 1000, 'disp': False, 'catol': 1e-8}
        xx = so.minimize(findereff, [ereff_est.real, ereff_est.imag], 
                         method='COBYLA', #options=options, tol=1e-8,
                       args=(Dinv@M.T@P@Q@M, line_lengths, f))
        ereff = xx.x[0] + 1j*xx.x[1]
        gamma = 2*np.pi*f/c0*np.sqrt(-ereff)
        gamma = abs(gamma.real) + 1j*abs(gamma.imag)
        
    # lines model
    L = np.array([[np.exp(-gamma*l),0,0,np.exp(gamma*l)] for l in line_lengths]).T
    
    # weighting matrix
    exps = np.exp(gamma*line_lengths)
    W = (np.outer(exps,1/exps) - np.outer(1/exps,exps)).conj()
    
    H = L@W@L.T@P@Q          # model (weighted)
    F = M@W@Dinv@M.T@P@Q     # measurements (weighted)
    
    abs_lambda = max(abs(np.linalg.eigvals(F)))  # abs(lambda) for debugging!
    
    # nullspace basis for x2_ and x3_
    u, s, vh = np.linalg.svd(F, full_matrices=True)
    F = (u[:,:-2]*s[:-2])@vh[:-2,:]   # Low-rank approximation (Eckart–Young)
    _, _, vh = np.linalg.svd(F, full_matrices=True)
    v1 = vh[-1,:].T.conj()
    v2 = vh[-2,:].T.conj()
    v11,v12,v13,v14 = v1
    v21,v22,v23,v24 = v2
            
    # find x2_ = x2/b11 (2 solutions)
    if abs(v12) > abs(v22):  # to avoid dividing by small numbers
        k2 = -v11*v22*v24/v12 + v11*v14*v22**2/v12**2 + v21*v24 - v14*v21*v22/v12
        k1 = v11*v24/v12 - 2*v11*v14*v22/v12**2 - v23 + v13*v22/v12 + v14*v21/v12
        k0 = v11*v14/v12**2 - v13/v12
        c2 = poly.polyroots([k0,k1,k2])*np.array([1,1])
        c1 = (1 - c2*v22)/v12
    else:
        k2 = -v11*v12*v24/v22 + v11*v14 + v12**2*v21*v24/v22**2 - v12*v14*v21/v22
        k1 = v11*v24/v22 - 2*v12*v21*v24/v22**2 + v12*v23/v22 - v13 + v14*v21/v22
        k0 = v21*v24/v22**2 - v23/v22
        c1 = poly.polyroots([k0,k1,k2])*np.array([1,1])
        c2 = (1 - c1*v12)/v22
    x2_ = np.array( [v1*x + v2*y for x,y in zip(c1,c2)] )  # 2 solutions
    
    # find x3_ = x3/a11 (2 solutions)
    if abs(v13) > abs(v23):  # to avoid dividing by small numbers
        k2 = -v11*v23*v24/v13 + v11*v14*v23**2/v13**2 + v21*v24 - v14*v21*v23/v13
        k1 = v11*v24/v13 - 2*v11*v14*v23/v13**2 + v12*v23/v13 - v22 + v14*v21/v13
        k0 = v11*v14/v13**2 - v12/v13
        c4 = poly.polyroots([k0,k1,k2])*np.array([1,1])
        c3 = (1 - c4*v23)/v13
    else:
        k2 = -v11*v13*v24/v23 + v11*v14 + v13**2*v21*v24/v23**2 - v13*v14*v21/v23
        k1 = v11*v24/v23 - v12 - 2*v13*v21*v24/v23**2 + v13*v22/v23 + v14*v21/v23
        k0 = v21*v24/v23**2 - v22/v23
        c3 = poly.polyroots([k0,k1,k2])*np.array([1,1])
        c4 = (1 - c3*v13)/v23
    x3_ = np.array( [v1*x + v2*y for x,y in zip(c3,c4)] )   # 2 solutions
    
    # estimates for x1/a11b11 and x4 from F-+lambda
    J = F-H[0,0]*np.eye(4)   # F+lambda
    u, s, vh = np.linalg.svd(J, full_matrices=True)
    J = (u[:,:-1]*s[:-1])@vh[:-1,:]   # Low-rank approximation (Eckart–Young)
    _, _, vh = np.linalg.svd(J, full_matrices=True)
    v = vh[-1,:].T.conj()
    x1__est = v/v[0]   # x1_ = x1/a11b11
    
    J = F-H[-1,-1]*np.eye(4)  # F-lambda
    u, s, vh = np.linalg.svd(J, full_matrices=True)
    J = (u[:,:-1]*s[:-1])@vh[:-1,:]  # Low-rank approximation (Eckart–Young)
    _, _, vh = np.linalg.svd(J, full_matrices=True)
    v = vh[-1,:].T.conj()
    x4_est = v/v[-1]   # x4
    
    x2__est = np.array([x4_est[2], 1, x4_est[2]*x1__est[2], x1__est[2]])
    x3__est = np.array([x4_est[1], x4_est[1]*x1__est[1], 1, x1__est[1]])
        
    # choose the correct answer for x2_ and x3_
    mininx = np.argmin( abs(x2_ - x2__est).sum(axis=1) )
    x2_ = x2_[mininx]
    mininx = np.argmin( abs(x3_ - x3__est).sum(axis=1) )
    x3_ = x3_[mininx]
    
    # build x1_ and x4 from x2_ and x3_
    x1_ = np.array([1, x3_[3], x2_[3], x3_[3]*x2_[3]])
    x4 = np.array([x3_[0]*x2_[0], x3_[0], x2_[0], 1])
    
    # Normalized calibration matrix 
    X_ = np.array([x1_, x2_, x3_, x4]).T
    
    # solve for a11b11 and K from Thru measurement
    Ka11b11,_,_,K = np.linalg.pinv(X_)@M[:,0]
    a11b11 = Ka11b11/K
    
    # solve for a11/b11, a11 and b11 (use redundant reflect measurement, if available)
    a11 = []
    b11 = []
    for reflect_S, reflect_ref, offset in zip(meas_reflect_S,reflect_est,reflect_offset):
        Ga = reflect_S[0,0]
        Gb = reflect_S[1,1]
        a11_b11 = (Ga - x2_[0])/(1 - Ga*x3_[3])*(1 + Gb*x2_[3])/(Gb + x3_[0])
        a1 = np.sqrt(a11_b11*a11b11)
        
        # choose correct answer for a11 and b11
        ref_est = reflect_ref*np.exp(-2*gamma*offset)
        G0 = (Ga - x2_[0])/(1 - Ga*x3_[3])/a1
        if abs(G0 - ref_est) > abs(-G0 - ref_est):
            a1 = -a1
        a11.append(a1)
        b11.append(a11b11/a1)
    
    a11 = np.array(a11).mean()
    b11 = np.array(b11).mean()
    
    # build the calibration matrix (de-normalize)
    X  = X_@np.diag([a11b11, b11, a11, 1])
    
    ## comment this block if you want gamma from the optimization solution
    # Compute a better gamma and ereff from the longest line standard
    inxmax = np.argmax(line_lengths)
    l = line_lengths[inxmax]
    _,_,_,z = np.linalg.pinv(X)@M[:,inxmax]/K
    n = np.round( (-np.log(z)+gamma*l).imag/2/np.pi ) # phase unwrap factor
    gamma = (np.log(z) + 1j*2*np.pi*n)/l
    ereff = -(c0/2/np.pi/f*gamma)**2
    
    
    return X, K, ereff, gamma, abs_lambda

# EOF