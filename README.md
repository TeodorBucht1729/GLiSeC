# GLiSeC
GLiSeC stands for Geometric Limit Set Calculator, and it is a tool for approximating the limit set of eigenvalues for banded Toeplitz matrices. The algorithm is presented in the paper available at https://arxiv.org/abs/2308.00829. 

The main algorithm, corresponding to Algorithm 3 in the paper is the file `polygon.py`, this file also contains examples of how to run the algorithm that have been used to create the results and Figures presented in the paper. 

Note that the file `mpsolve.py` is directly taken from MPSolve repository: https://github.com/robol/MPSolve, and it is added to this repositiry to make the code runnable. Also note that the package `mpsolve` needs to be installed in order to run `limit_calc_mpsolve.py`, which is used in one of the examples. It can be installed here: https://numpi.dm.unipi.it/scientific-computing-libraries/mpsolve/. 