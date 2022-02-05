# Multiline TRL Calibration

Implementation of multiline TRL calibration. Two algorithms are included here:

1. An “improved” implementation based on a work of mine [1].
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

I will include a documentation in the future...

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

This example demonstrate how to do a mTRL calibration using simulated data from skrf package. You can adjust the simulated data to test various scenarios. 

### example 4:

This example demonstrate how to do a statistical analysis on mTRL calibration via Monte-Carlo method. For this example I used the data from example_1, where I provided options (uncomment to use) to analyze the cases of additive noise, or phase error. You can create other types of analysis.

## Work in Progress

By no means is this a final release. There are still functions I want to implement, and also I will try to inlcude a more proper documentation...

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
