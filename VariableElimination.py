import numpy as np

def do_padding(factor, indices):
    #print 'padding factor', factor
    factor[1] = factor[1].transpose(np.argsort(factor[0]))
    shape = tuple([2 if i in factor[0] else 1 for i in indices])
    #print shape
    
    factor[1] = factor[1].reshape(shape)

    shape = tuple([1 if i in factor[0] else 2 for i in indices])
    new_factor = np.tile(factor[1], shape)
    #new_factor = new_factor / new_factor.sum()
    return new_factor

def do_product(phi):
    #print 'doing a product over', phi
    indices = set()
    
    for f in phi:
        indices |= set(f[0])

    indices = sorted(indices)

    #print 'indices are', indices
    prod = np.ones((2,)*len(indices), dtype=float)
    for f in phi:
        pad = do_padding(f, indices)
        #print 'padded version of', f, 'is', pad
        prod *= pad

    return indices, prod

def eliminate(L, factors):
    #print 'eliminating', L, 'from', factors
    phi = [f for f in factors if L in f[0]]
    phi_prime = [f for f in factors if not L in f[0]]


    #print 'phi is', phi
    order, psi = do_product(phi)

    #print 'psi is', psi, order
    #print 'summing over', L
    L_index = order.index(L)

    #print 'summing over index', L_index
    tau = np.sum(psi, axis=L_index)
    #print 'tau is', tau
    order.pop(L_index)
    phi_prime.append([order, tau])
    return phi_prime
