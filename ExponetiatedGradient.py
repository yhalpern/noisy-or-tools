import numpy as np
from helpers import logsumexp


def expGrad(_f, x0, eps, max_steps=1000, lower=float('-inf'), verbose=False):
    #work in logspace
    x = np.log(x0)
    l = lower
    step = 1
    while step < max_steps:
        assert np.abs(np.exp(x).sum()- 1) < 1e-9, x.sum()
        old_f, g = _f(np.exp(x), withGrad=True)
        eta = 1.0
        while eta > 0:
            old_x = x
            x = x -eta*g.T
            x = x - logsumexp(x)
            f = _f(np.exp(x))
            if eta < 1e-10:
                eta = 0
                break

            if f < old_f:
                break
            else:
                x = old_x
                eta /= 2.0
        #check convergence
        if verbose:
            #print 'g', g
            print 'eta', eta,
            print 'f', f,
        lam = g
        lam -= lam.min()
        gap = np.dot(lam,np.exp(x))
        l = max(l, f-gap)

        assert f >= l, "f < l? "+str(f)+':'+str(l)

        if verbose:
            print 'gap', f-l

        if (f-l < eps) or step > max_steps or eta == 0:
            f = _f(np.exp(x), withGrad=False)
            break

        step += 1

    return np.exp(x), f, step, gap

if __name__ == "__main__":
    x0 = np.ones((5,1)) / 5.0
    x0[0] = 0.9
    x0 = x0 / x0.sum()

    def _f(x, withGrad=0):
        val = np.dot(x.T,x)
        if withGrad:
            g = 2*x.T
            return val, g
        return val

    expGrad(_f, x0, eps=1e-5, verbose=True)
