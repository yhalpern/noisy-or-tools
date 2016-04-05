from __future__ import division
import shelve
import sys
from collections import defaultdict
import numpy as np
import itertools
from operator import mul
import subprocess
import os
from copy import deepcopy

def sample(p, N):
    shape = p.shape
    p = p.flatten() / float(p.sum())
    A = np.random.multinomial(N, p)
    A += 1.0 #add some extra noise
    return A.reshape(shape)

def logsumexp(L):
    A = np.max(L)
    L = L-A
    return A + np.log(np.exp(L).sum())

def get_counts(marginals, header):
    counts = {}
    for k in marginals:
        #order = index_sort([header[z] for z in k])
        order = np.argsort([header[z] for z in k])
        #print [header[z] for z in k], 'order', order
        counts[tuple(sorted([header[z] for z in k]))] = marginals[k].transpose(tuple(order))
    return counts

def create_CPD(m, smoothing=0, verbose=False): #first dimension is the variable itself
    r = deepcopy(m)
    nVars = len(m.shape)
    r += smoothing
    for K in itertools.product([0,1], repeat=nVars-1):
        key = tuple([slice(None)] + list(K))
        r[key] = m[key] / m[key].sum() 
        if verbose:
            print 'r[', key, '] =', m[key], '/', m[key].sum()
            print 'just checking m[:,0,0]', m[:,0,0]

    return r

def extract_edges(structure_file):
    #columns are parents
    adj = np.loadtxt(structure_file)
    parents_of = defaultdict(list)
    edges = zip(*np.nonzero(adj))
    #for i,j in edges:
    #    parents_of[j].append(i)

    return edges

def makedirs(name):
    try:
        os.makedirs(name)
    except:
        pass

def index_sort(L):
    L_sorted = sorted(L)
    return [L_sorted.index(l) for l in L]


def flatten(L):
    ret = []
    for l in L:
        ret += l
    return ret

def row_normalize(M):
    M = np.array(M, dtype=float)
    for i in xrange(M.shape[0]):
        if float(M[i,:].sum()) > 0:
            M[i,:] = M[i,:] / float(M[i,:].sum())
    return M

def stack(L):
    #print [np.array(l).shape for l in L]
    r = np.vstack([np.array(l) for l in L]) 
    return r
    
def calculate_adjustment(noise_list):
    n = len(noise_list)
    A = np.matrix(np.zeros((2**n, 2**n)))
    for r_index, r in enumerate(itertools.product([0,1], repeat=n)):
        for s_index, s in enumerate(itertools.product([0,1], repeat=n)):
            A[r_index,s_index] = prod([noise_list[i][r[i],s[i]] for i in xrange(n)])

    A = np.matrix(A)
    return A, None#np.linalg.pinv(A)

def joint(K):
    return np.zeros((2,)*K)


def KL(A,B, verbose=False):
    #print 'KL A,B'
    #print A.shape
    #print B.shape
    div = 0
    A = np.array(A).reshape((A.size, 1))
    B = np.array(B).reshape((B.size,1))
    div = [A[i]*np.log(A[i]/B[i]) for i in xrange(A.size) if A[i] > 0]
    if verbose:
        print div
    div = np.sum(div)
    #assert div >= -10**(-9)
    return div

def gradKL(X,Y, A):
    B = A*X
    grad = -(Y / B).T*A
    return grad.reshape((Y.size,1))

def smoothKL(A,B):
    div = 0
    A = np.array(A).reshape((A.size, 1))
    B = np.array(B).reshape((B.size,1))
    B += 10**(-9)
    B /= B.sum()
    div = A*np.log(A/B)
    div = np.nansum(div)
    assert div >= -10**(-9)
    return div
    
def gradsmoothKL(X,Y, A):
    B = A*X
    B += 10**(-9)
    B /= B.sum()
    grad = -(Y / B).T*A
    return grad.reshape((Y.size,1))


def L2(A,B):
    #print 'L2', float(np.dot((A-B).T, (A-B)))
    return float(np.dot((A-B).T, (A-B)))

def gradL2(X,Y,A):
    #d/dX (Y-AX)**2
    grad = -2*(Y-A*X).T*A
    return grad.reshape((Y.size,1))



def reformat(input, output):
    print 'reformatting', input, 'into', output
    try:
        f = file(output)
        f.close()
        print 'already done'
        return 0
    except:
        pass

    try:
        parent_networkdir = input.split('-')[0]
        print 'try to copy from parent directory', parent_networkdir
        subprocess.call(('cp -l --verbose '+parent_networkdir+'/samples/samples.dat '+output).split())
    except:
        print 'could not reformat. this could take a long time'

        infile = file(input)
        outfile = file(output, 'w')
        infile.readline() #skip header
        for l in infile:
            l = l.split(',')
            l = [str(int(int(z) > 0)) for z in l]
            print >>outfile, " ".join(l)
        print 'done'
        infile.close()
        outfile.close()

def execute_query(L, networkdir, sample_size, nvars):
    counter = {}
    schedule_file = file(networkdir+'/true_params/schedule.txt', 'w')
    print >>schedule_file, len(L)

    for l in L:
        print >>schedule_file, ' '.join([str(x) for x in list(l)+[-1]*(4-len(l))])

    schedule_file.close()
    reformat(networkdir+'/data', networkdir+'/samples/'+str(sample_size)+'_samples.dat')

    proc =  ['sampling/read_ss']
    args =  [networkdir, networkdir+'/samples/'+str(sample_size)+'_samples.dat',  'schedule.txt', str(sample_size), str(nvars), str(0), str(0)]
    print proc + args
    sys.stdout.flush()
    subprocess.call(proc + args)
    
    id_file = file(networkdir+'/samples/id_n'+str(sample_size)+'_s0_schedule.txt')
    stats_file = file(networkdir+'/samples/sufficientStatistics_n'+str(sample_size)+'_s0_schedule.txt')

    stats_file.readline()
    for l in L:
        iden = [int(z) for z in id_file.readline().strip().split()]
        stat = [int(z) for z in stats_file.readline().strip().split()]
        #print iden
        stat = np.array(stat).reshape((2,)*4)

        sum_over = []
        for n,i in enumerate(iden):
            if i == -1:
                sum_over.append(n)
        stat = stat.sum(tuple(sum_over))
        iden = tuple([z for z in iden if not z == -1])
        counter[iden] = stat
        print counter[iden].sum()
    print 'query result returned', len(counter), 'statistics'
    return counter

def parallel_execute_query(L, networkdir, sample_size, nvars, nthreads):
    print 'reading in parallel!', 'want to read', sample_size, 'elements with', nthreads, 'threads'

    np.random.seed(os.getpid())
    rng = np.random.random_integers(0,1000000,size=nthreads)
    counter = {}
    countShelf = shelve.open(networkdir+'/samples/counterShelf')
    L = [tuple(l) for l in L]

    L = set(L)
    print len(L), 'statistics to read'
    #print countShelf.keys()
    for l in list(L):
        if str(l) in countShelf:
            try:
                counter[l] = countShelf[str(l)]
                L.remove(l)
                #print 'loaded from shelf'
            except:
                pass
        else:
            #print str(l), 'could not load'
            pass

    L = list(L)

    if len(L) > 0:
        schedule_file = file(networkdir+'/true_params/schedule.txt.'+str(rng[0]), 'w')
        print >>schedule_file, len(L)

        for l in L:
            print >>schedule_file, ' '.join([str(x) for x in list(l)+[-1]*(4-len(l))])

        schedule_file.close()
        reformat(networkdir+'/data', networkdir+'/samples/'+str(sample_size)+'_samples.dat')

        processes = []
        proc =  ['sampling/read_ss']


        for i in xrange(nthreads):
            try:
                a = file(networkdir+'/samples/'+str(sample_size)+'_samples.dat0'+str(i))
                a.close()
            except:
                print 'could not open', networkdir+'/samples/'+str(sample_size)+'_samples.dat0'+str(i)
                #subprocess.call("head -n "+str(sample_size)+" "+networkdir+'/samples/'+str(sample_size)+'_samples.dat | split -l '+str(sample_size//nthreads)+' -d - '+networkdir+'/samples/'+str(sample_size)+'_samples.dat', shell=True)
                print "cat "+networkdir+'/samples/'+str(sample_size)+'_samples.dat |grep -v ^$| head -n ' +str(sample_size)+' | split -l '+str(sample_size//nthreads)+' -d - '+networkdir+'/samples/'+str(sample_size)+'_samples.dat'

                subprocess.call("cat "+networkdir+'/samples/'+str(sample_size)+'_samples.dat |grep -v ^$| head -n ' +str(sample_size)+' | split -l '+str(sample_size//nthreads)+' -d - '+networkdir+'/samples/'+str(sample_size)+'_samples.dat', shell=True)
                print 'done'
                break

        for i in xrange(nthreads):
            offset = i*(sample_size // nthreads)
            if i < 10:
                args =  [networkdir, networkdir+'/samples/'+str(sample_size)+'_samples.dat0'+str(i),  'schedule.txt.'+str(rng[0]), str(sample_size//nthreads), str(nvars), '0', str(rng[i])]
            else:
                args =  [networkdir, networkdir+'/samples/'+str(sample_size)+'_samples.dat'+str(i),  'schedule.txt.'+str(rng[0]), str(sample_size//nthreads), str(nvars), '0', str(rng[i])]

            p1 = subprocess.Popen(proc+args) 
            print ' '.join(proc + args)
            processes.append(p1)

        for p in processes:
            p.wait()

        for thread in xrange(nthreads):
            id_file = file(networkdir+'/samples/id_n'+str(sample_size//nthreads)+'_s'+str(rng[thread])+'_schedule.txt.'+str(rng[0]))
            stats_file = file(networkdir+'/samples/sufficientStatistics_n'+str(sample_size//nthreads)+'_s'+str(rng[thread])+'_schedule.txt.'+str(rng[0]))

            stats_file.readline()
            for l in L:
                iden = [int(z) for z in id_file.readline().strip().split()]
                stat = [int(z) for z in stats_file.readline().strip().split()]
                stat = np.array(stat).reshape((2,)*4)

                sum_over = []
                for n,i in enumerate(iden):
                    if i == -1:
                        sum_over.append(n)
                stat = stat.sum(tuple(sum_over))
                iden = tuple([z for z in iden if not z == -1])
                if thread == 0:
                    counter[iden] = stat
                else:
                    counter[iden] += stat
            stats_file.close()
            id_file.close()

            #os.remove(stats_file.name)
            #os.remove(id_file.name)
        #os.remove(schedule_file.name)

        for iden, val in counter.items():
            try:
                if not str(iden) in countShelf:
                    countShelf[str(iden)] = val
            except:
                print 'could not save', iden
                pass

        try:
            countShelf.close()
        except:
            pass
    print 'query result returned', len(counter), 'statistics'
    return counter

def mean_stdev(X):
    return np.mean(X), np.std(X)

def normalize(p):
    return p / float(p.sum())   

def bernoulli(p):
    if type(p) == float:
        return int(np.random.rand() < p)
    else:
        return np.array(np.random.rand(*p.shape) < p,dtype=int)

def beta(a,b):
    return np.random.beta(a,b)

def prod(l):
    return reduce(mul, l, 1)

def approx_equal(a,b,eps):
    return np.abs(a-b) < eps

def rootedTree(vars, edges, root):
    directed_edges = {}
    new_edges = []
    initial_root = root

    for root,v in ((root,v) for v in vars if ((root, v) in edges or (v,root) in edges)):
        new_edges.append((root,v))

    while(len(new_edges)):
        root,v = new_edges.pop()

        try:
            directed_edges[(root,v)] = edges[(root,v)]
        except:
            directed_edges[(root,v)] = edges[(v,root)].transpose()

        root = v
        for root,v in ((root,v) for v in vars if ((root, v) in edges or (v,root) in edges)):
            if not (root,v) in directed_edges and not (v,root) in directed_edges.keys():
                new_edges.append((root,v))

    children_of = {}
    parents_of = {}
    for v in vars:
        children_of[v] =  set((z for z in vars if (v,z) in directed_edges.keys()))
        parents_of[v] =  set((z for z in vars if (z,v) in directed_edges.keys()))
        assert(len(parents_of[v]) == 1 or v == initial_root), str(v)+':'+str(parents_of[v])

    return initial_root, directed_edges, children_of, parents_of

def subsets(L, complement=False):
    S = []
    C = []
    for inclusion in itertools.product([0,1], repeat=len(L)):
        S.append([L[i] for i,include in enumerate(inclusion) if include==1])
        if complement:
            C.append([L[i] for i,include in enumerate(inclusion) if include==0])
    if complement:
        return zip(S, C)
    else:
        return S

def prob(model, X, condition=None):
    print "ERROR SHOULD NOT BE CALLING THIS FUNCTION"
    sys.exit()
    if not condition==None:
        p = prob(model, X+condition) / prob(model, condition)
    else:
        marginals = getMarginals(model, [x[0] for x in X])
        nvars = len(X)

        joint = marginals2joint(model, marginals, [x[0] for x in X])

        if model.noise:
            print 'model has noise!'
            #print nvars
            #print joint.shape
            #print 2**nvars
            joint = calculate_adjustment([model.noise[x[0]] for x in X])*np.matrix(joint.reshape((2**nvars, 1)))
            joint = np.array(joint)
            joint = joint.reshape((2,)*nvars)

        index = tuple([x[1] for x in X])
        p = joint[index]
    assert p <= 1+10**(-6), str(X)+"|"+str(condition) + ':'+str(p)
    return p

def getJoint(model, X):
    marginals = getMarginals(model, X)
    nvars = len(X)
    joint = marginals2joint(model, marginals, X)
    return joint

def getMarginals(model,X):
    marginals = {}
    for T in subsets(X):
        adjustment=[]
        observable_X = []
        marginals[tuple(T)] = model.counts[frozenset(T)]
    return marginals

def marginals2joint(model, marginals, X=None):
    X = sorted(marginals.keys(), key=len, reverse=True)[0]
    numVars = int(np.log2(len(marginals)))

    joint = np.zeros((2,)*numVars)
    for T in sorted(itertools.product([0,1],repeat=numVars), key=sum, reverse=True):
        ones = [i for i in xrange(numVars) if T[i]]
        zeros = [i for i in xrange(numVars) if not T[i]]
        joint[T] = marginals[tuple([X[i] for i in ones])]
        for Z in subsets(zeros):
            if len(Z)==0:
                continue

            Tprime = list(T)
            for z in Z:
                Tprime[z] = 1
            joint[T] -= joint[tuple(Tprime)]
    return joint

if __name__ == "__main__":
    print "testing subsets"
    print subsets([2*i for i in xrange(4)])

    print "testing marginals2joint"
    joint = np.random.rand(2,2,2)
    joint = joint / joint.sum()
    marginals = {}
    for T in subsets(range(3)[::-1]):
        marginals[tuple(T)] = joint.sum((tuple(set(range(3))-set(T))))[(1,)*len(T)]

    print marginals
    recovered_joint = marginals2joint(None, marginals, range(3)[::-1])
    print joint - recovered_joint

def mutual_information(P):
    P = np.clip(P,0,1)
    P = P / P.sum()
    M = np.sum([0]+[P[i,j]*np.log(P[i,j]/(P[i,:].sum()*P[:,j].sum())) for i,j in itertools.product([0,1], repeat=2) if P[i,j] > 0])
    if np.isnan(M):
        print 'mutual information is nan?', P
        sys.exit()
    return M

def no_noise():
    return np.matrix(np.identity(2))

def calculate_noise(true_tag_counter, noisy_tag_counter, true_and_noisy_tag_counter, tags, limit):
    noise = defaultdict(no_noise)
    for i,t in enumerate(tags):

        marginals = {}
        marginals[tuple()] =  1.0
        marginals[('true',)] = true_tag_counter[frozenset([t])]
        marginals[('noisy',)] = noisy_tag_counter[frozenset([t])]
        marginals[('noisy', 'true')] = true_and_noisy_tag_counter[t]

        noise[i] = marginals2joint(None, marginals, ['noisy', 'true'])
        for j in xrange(2):
            noise[i][:, j] = normalize(noise[i][:,j])

        #noise[0,1] is the failure probability P(noisy_counter=False|true_counter=True)

        #noise[i] = np.zeros((2,2))
        #if true_tag_counter[t]:
        #    noise[i][1,1] = true_and_noisy_tag_counter[t] / true_tag_counter[t]

        #noise[i][0,1] = 1-noise[i][1,1] #failure probability

        #if limit-true_tag_counter[t]:
        #    noise[i][1,0] = len(noisy_tag_counter[t] - true_tag_counter[t]) / (limit - len(true_tag_counter[t])) #leak
        
        #noise[i][0,0] = 1-noise[i][1,0] 
    return noise
