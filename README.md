# Multiline TRL Calibration

Implementation of multiline TRL calibration. Two algorithms are included here:

1. An “improved” implementation based on a paper of mine [1].
2. The classical MultiCal implementation from NIST [2,3].

## Difference between implementations

NIST MultiCal [2,3]:

1. Assumes that error in the measurements can be modeled linearly in the solution of the calibration coefficients (eigenvectors).
2. Given that linearity assumption holds, then one can solve multiple eigenvalue problems and combine their result using the Gauss-Markov BLUE estimator (weighted sum) to obtain a final solution.
3. The weighing is done on the solutions (eigenvectors) and not the measurements.

TUG mTRL [1]:

1. Makes no assumption on the type of error propagated through the calibration coefficients.
2. A weighting matrix is derived to combine the measurements optimally (minimize eigenvectors sensitivity).
3. The weighting is done on the measurements to create a single eigenvalue problem to solve.

**So, why the method I’m presenting (TUG mTRL [1]) is better?**

The simple answer is that it does not enforces any assumption on the quality of your measurements. It just give the best solution it can deliver given what you provided it.

A more involved answer is it computes a “true mean value” of your calibration even if high noise is present in your measurements. 
So, the linearity imposed in NIST MultiCal can only hold true if the error in the measurement is low. 
Generally, if we have clean data (minimum error), both mTRL methods will give same result. But, the moment your measurements have some error, the result between TUG mTRL and NIST MultiCal will differ differently from the true solution (noiseless solution), depending how high the error is. 
If the error is small, then TUG mTRL and NIST MultiCal will equally deviate from the true solution. But, the moment the error in the measurements is too high, then NIST MultiCal will differ quite dramatically from the true solution. This is because linearity assumption was violated and thus Gauss-Markov estimator is not valid anymore.
So, if you perform a statistical experiment (assume noise is too high) and take the mean value of both NIST MultiCal, and TUG mTRL. You will find that TUG mTRL will converge to the true solution (un-biased), while NIST MultiCal will converge to a wrong solution (biased).

## Code requirements

First, these three files need to be in the same folder `mTRL.py`, `MultiCal.py`, `TUGmTRL.py` , and you will load the file `mTRL.py` in your main script (also, in same folder).

Secondly, you need to have `numpy`, `scipy`, `matplotlib`, and `scikit-rf` installed in your python environment.

```batch
python -m pip install -U numpy scipy matplotlib scikit-rf
```

## Examples

### example 1:

This example demonstrate how to do a simple 2nd tier calibration (de-embedding), where the s2p data were captured using an already calibrated VNA.

### example 2:

This example demonstrate how to do a full 1st tier calibration (including switch terms). The s2p data are the raw data from the VNA. 

### example 3:

I will include this later, but the idea is use simulated data. An example of simulated data can be found in skrf website: [https://scikit-rf.readthedocs.io/en/latest/examples/metrology/Multiline TRL.html](https://scikit-rf.readthedocs.io/en/latest/examples/metrology/Multiline%20TRL.html)

Note: the NIST MultiCal implementation I used is not imported from skrf, but I wrote it myself based on [2,3].

## Work in Progress

By no means is this a final release. There are still functions I need to implement, like shifting the reference plane (now it is in the middle of the Thru standard). Also, to re-normalize the calibration impedance... and many other small tweaks.

Also, I will try to inlcude a more proper documentation...

## References

- [1] Ziad Hatab, Michael Gadringer, Wolfgang Boesch, "Improving the Reliability
of the Multiline TRL Calibration Algorithm," presented at 98th ARFTG Conference, Jan. 2022
    
    (I will include a url when the paper is listed in [https://ieeexplore.ieee.org/](https://ieeexplore.ieee.org/))
    
- [2] D. C. DeGroot, J. A. Jargon and R. B. Marks, "Multiline TRL revealed,"
60th ARFTG Conference Digest, Fall 2002., pp. 131-155
    
    [Multiline TRL revealed](https://ieeexplore.ieee.org/document/1218696)
    
- [3] R. B. Marks, "A multiline method of network analyzer calibration",
IEEE Transactions on Microwave Theory and Techniques,
vol. 39, no. 7, pp. 1205-1215, July 1991.
    
    [A multiline method of network analyzer calibration](https://ieeexplore.ieee.org/document/85388)
