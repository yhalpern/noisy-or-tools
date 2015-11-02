import pyximport; pyximport.install()
import numpy as np
import scipy.optimize as opt
import maximum_likelihood_helpers
from multiprocessing import Pool


def learn_failures((X, Y, L)):
    def func(failures):
        ret = -maximum_likelihood_helpers.func(X,Y,failures)
        return ret

    def grad(failures):
        ret = -maximum_likelihood_helpers.grad(X,Y,failures)
        return ret

    X0 = np.log(np.ones(L+1, dtype=float)*(1-1e-6)) #initialize a small distance away from the bound
    bounds = [(None, 0) for _ in xrange(L+1)] 
    bounds[-1] = (None, np.log(1-1e-6)) #never allow the leak to go too close to 0
    failures, _, _ = opt.fmin_l_bfgs_b(func, X0, grad, bounds=bounds, disp=0)
    failures = np.exp(failures) #optimization was in log space. Exponentiate to get values.
    return failures

if __name__ == '__main__':

    pool = Pool(20)
    L = 50 # latent variables
    M = 300 # observations
    N = 1000 # patients
    X_matrix = np.array(np.random.rand(N, M) > 0.7, dtype=int)
    Y_matrix = np.array(np.random.rand(N, L) > 0.8, dtype=int)
    Y = [np.nonzero(y)[0].tolist() for y in Y_matrix]
    targets = ((X_matrix[:, j].tolist(), Y, L) for j in xrange(M))
    failures = np.array(pool.map(learn_failures, targets))
    print failures.shape

