from __future__ import division
import hashlib
from gurobipy import *
from cycleInequalities import cycle_sep
from ExponetiatedGradient import expGrad
import ast
import time
import cPickle as pickle
import scipy.sparse as sparse
import numpy as np
from itertools import combinations, product
import itertools
import sys
from FrankWolf import FW
#from L1_minimization import L1_minimization
from helpers import *
from utils import extractOrder
from multiprocessing import Pool


def extract_candidates(networkdir, label_name):
    #returns a list of tuples: (score_a, score_b, candidate, conflicts)

    infile = file(networkdir+'/candidates/'+'candidates.'+label_name)
    retval = []
    while 1:
        l = infile.readline()
        if l == '':
            break
        candidate = l.strip()
        l = infile.readline()
        score_a, score_b = l.split()
        score_a = float(score_a)
        score_b = float(score_b)
        conflicts = []
        while 1:
            l = infile.readline()
            if '---' in l:
                break
            conflicts.append(l.strip())
        retval.append((score_a, score_b, candidate, conflicts))
    infile.close()
    
    return retval

def init_anchors(networkdir, labels, header, default_anchors):
    learned_anchors = {}
    for i in labels:
        candidates_i = extract_candidates(networkdir, header[i])
        candidates_i.sort(key=lambda p: p[0], reverse=True)
        learned_anchors[i,]  =  (header.index(candidates_i[0][2]),)

    for i,j in itertools.combinations(labels, 2):
        candidates_i = extract_candidates(networkdir, header[i])
        candidates_j = extract_candidates(networkdir, header[j])
        valid_pairing = False
        for a,b in sorted(itertools.product(candidates_i, candidates_j), key=lambda p: p[0][0] + p[1][1], reverse=True):
            print 'attempting to pair', a[2], b[2]
            if (not (a[2] in b[3])) and (not (b[2] in a[3])):
                learned_anchors[i,j] = (header.index(a[2]),header.index(b[2]))
                valid_pairing = True
                break
            else:
                print 'conflict!'

        if not valid_pairing:
            print 'warning! could not find a valid pairing for', header[i], header[j]
            learned_anchors[i,j] = (default_anchors[labels.index(i)], default_anchors[labels.index(j)])

    return learned_anchors


def restartCheckpoint(networkdir):
    n0 = 0
    logfile = file(networkdir+'/logging/transform.log')
    for l in logfile:
        if l[0] == 'n':
            try:
                _, z = l.split()
                n0 = int(z)
            except:
                pass

    if n0 % 100 > 0:
        n0 -= (n0 % 100)
       
    x0 = pickle.load(file(networkdir+'/pickles/transform.latest_iterate.pk'))
    return x0, n0


def project((T, A_mat, b_vec, marginals, gamma, eps)):
    K = len(T)

    x = np.zeros((2,)*K)
    for ind in product([0,1], repeat=K):
        x[ind] = prod([marginals[t] if state==1 else 1-marginals[t] for t,state in zip(T, ind)])

    m_vec = np.matrix(x.reshape((x.size,1)))
    temp = np.array(b_vec).reshape((2,)*K)

    print 'projecting', T, "A b", A_mat, b_vec
    for k in xrange(K):
        if np.prod(temp.sum(tuple(set(xrange(K)) - set([k])))) == 0: #if any complete marginalization has a 0 - ie there's an     anchor that never appears
            print 'here'
            return T, np.array(m_vec).reshape((2,)*K)

    x_init = m_vec.copy()
    x_init = x_init / x_init.sum()

    #A_mat = np.matrix(A[T])

    #b_vec = np.matrix(B[K][T])

    def f(x, withGrad=False, args=''):
        if 'L2' in prog_args:
            _dist = L2
            _grad = gradL2
        elif 'smoothkl' in prog_args:
            _dist = smoothKL
            _grad = gradsmoothKL
        else:
            _dist = KL
            _grad = gradKL
        kl =  _dist(b_vec, A_mat*x) + gamma*_dist(m_vec, x)
        if withGrad:
            grad = _grad(x,b_vec,A_mat).T + gamma*_grad(x,m_vec, sparse.identity(m_vec.size)).T
            return kl, grad
        return kl

    x,f,steps,gap = expGrad(f, x_init, eps=eps / float(len(T)), verbose=True)
    print 'f steps gap', f, steps, gap
    return T,  np.array(x).reshape((2,)*K)

def buildObjectiveFunction(A_mat, B_vec, u_vec, gamma, order, prog_args):
    def f(X_in=None, withGrad=False, args=''):

        if args == 'order':
            return order

        if args == 'N':
            return len(B[1])

        if args == 'lower':
            return 0

        if args == 'smallest':
            return np.sort(A_mat*X_in[-(2**order)*len(B[order]):,:], None)[0][-10:]

        if args == 'A':
            return A_mat

        if args == 'B':
            return B_vec

        if X_in is None:
            return sum([(2**k)*len(B[k]) for k in xrange(1,order+1)])

        if 'L2' in prog_args:
            _dist = L2
            _grad = gradL2

        elif 'smoothkl' in prog_args:
            _dist = smoothKL
            _grad = gradsmoothKL
        else:
            _dist = KL
            _grad = gradKL

        kl =  _dist(B_vec, A_mat*X_in[-(2**order)*len(B[order]):,:]) 
        kl = kl + gamma*_dist(u_vec, X_in[-(2**order)*len(B[order]):,:])

        if withGrad:
            grad = np.vstack([np.zeros((X_in.shape[0]-(2**order)*len(B[order]),1)), _grad(X_in[-(2**order)*len(B[order]):,:], B_vec, A_mat)]) 
            grad = grad + gamma*np.vstack([np.zeros((X_in.shape[0]-(2**order)*len(B[order]),1)), _grad(X_in[-(2**order)*len(B[order]):,:], u_vec, sparse.identity(u_vec.size))]) 
            print 'kl, grad', kl, np.max(grad), np.min(grad)
            return kl, grad
        
        return kl

    return f

def buildConstraints(prog_args,  order, set_to_var):
    constraints = []
    separation_oracle = None

    #first 2*n variables are marginals
    #next  4*n-choose-2 variables are second order
    #next  8*n-choose-3 variables are third order

    #nonnegative
    for i in xrange(sum([(2**k)*len(B[k]) for k in xrange(1,order+1)])):

        if 'integer' in prog_args:
            c = ([(1,i)], GRB.GREATER_EQUAL, 0)
        else:
            c = ([(1,i)], GRB.GREATER_EQUAL, 0)

        constraints.append(c)
    
    #sum to one
    for K in xrange(1,order+1):
        if K == 1:
            base = 0
        else:
            base = sum([(2**k)*len(B[k]) for k in xrange(1,K)])

        for i in xrange(len(B[K])):
            c = ([(1,base+(2**K)*i+j) for j in xrange(2**K)], GRB.EQUAL, 1)
            constraints.append(c)

    print 'simplex constraints'
    sys.stdout.flush()

    if 'local' in prog_args:
        #kth order marginals should match k-1th order
        for K in xrange(2,order+1):
            #create a dummy reference high order array
            H = np.array(xrange(2**K)).reshape((2,)*K)
            for n, T in enumerate(combinations(labels, K)):
                h_base =sum([(2**k)*len(B[k]) for k in xrange(1,K)]) + (2**K)*n
                for t in xrange(K): #sum along the t'th dimension
                    mask = tuple([slice(None) if z == t else 0 for z in xrange(K)]) #set all other values to 0
                    LHS = [(1, h_base+h) for h in H[mask]]

                    #find the lower order moment without the t'th dimension
                    S = list(T) #copy t
                    S.pop(t) #remove the t'th element
                    l_base = set_to_var[tuple(S)]
                    RHS = [(-1, l_base)] #take the 0'th element. That's the all 0's assignment

                    c = (LHS+RHS, GRB.EQUAL, 0)
                    print 'consistency constraint', c
                    constraints.append(c)    

    if 'cycle' in prog_args:
        separation_oracle = cycle_sep
    
    return constraints, separation_oracle


def elkan_noise(anchor):
   calibration = elkan_calibration(anchor)

   #emulate counter[t,s]
   #t- latent
   #s- observed
   noise = np.zeros((2,2))
   noise[0,1] = 0#1-clf.calibration
   noise[0,0] = 1#clf.calibration
   noise[1,1] = calibration
   noise[1,0] = 1-calibration
   return noise

def elkan_calibration(anchor):
   print "establishing elkan calibration for", anchor 
   clf = pickle.load(file(networkdir+'/concepts/'+header[anchor].replace('header_anchor_', 'label_')+'.elkan'))
   calibration = np.max(clf.calibration)
   if calibration == -1:
       print 'could not calibrate with such few examples', len(clf.full_calibration)
       print 'default calibration value is 1'
       calibration = 1
   print 'calibration is', calibration
   return calibration

def resolve_header(L):
    return [inv_header[l] for l in L]

def get_anchors(L, learned_anchors=None):
    if learned_anchors:
        return learned_anchors[L]
    else:
        return tuple([anchors[labels.index(l)] if l in labels else l for l in L]) #if l is not a label, it is its own anchor

if __name__ == "__main__":
    start_time =  time.time()
    networkdir = sys.argv[1]
    sample_size = int(sys.argv[2])
    prog_args = sys.argv[3:]


    #md5 = hashlib.md5()
    #md5.update(' '.join(sys.argv))
    #digest = md5.hexdigest()
    #number = int(digest, 16) % 4294967295
    np.random.seed(100)
    
    print ' '.join(sys.argv)
    print 'prog args', prog_args
    sys.stdout.flush()
    order = extractOrder(prog_args)

    if order == 3:
        eps = 5e-2
    else:
        eps = 5e-3

    if 'tighten' in networkdir:
        eps /= 10

    header = pickle.load(file(networkdir+'/pickles/header.pk'))
    labels = pickle.load(file(networkdir+'/pickles/labels.pk'))
    anchors = pickle.load(file(networkdir+'/pickles/anchor_list.pk'))
    #gamma = float(networkdir.split('-')[-3].strip('G')) / float(sample_size)
    if 'noreg' in networkdir:
        gamma = 0
    elif 'constreg' in networkdir:
        gamma = 0.01
    else:
        gamma = 0.01

    print 'gamma', gamma
    if 'third_order' in networkdir:
        gamma *= 10

    if 'learned_anchors' in networkdir:
        learned_anchors = init_anchors(networkdir, labels, header, anchors)
    else:
        learned_anchors = None


    counter = pickle.load(file(networkdir+'/counter.'+str(sample_size)+'.'+str(order)+'.pk'))
    sampled = set()

    A = {}
    A_inv = {}
    B = [{} for _ in xrange(order+2)]

    L = []

    for K in xrange(1, order+1):
        L+= list(combinations(labels,K))

    for T in L:
        S = get_anchors(T, learned_anchors)
        print 'anchors of', [header[t] for t in T], 'are', [header[s] for s in S]

        try:
            K = len(T)
            B[K][T] = counter[S].reshape((2**K,1))
        except:
            print "Unexpected error:", sys.exc_info()
            print "error!", T, S, K
            print counter[S]
            sys.exit()

        #normalize
        B[K][T] = B[K][T] / float(B[K][T].sum())

        #matricize
        B[K][T] = np.matrix(B[K][T])

        #determine noise matrix A

        if 'noisy_100' in prog_args:
            if not (t,s) in sampled:
                counter[t,s] = sample(counter[t,s],100)
                sampled.add((t,s))

        elif 'noisy_200' in prog_args:
            if not (t,s) in sampled:
                counter[t,s] = sample(counter[t,s],200)
                sampled.add((t,s))

        elif 'noisy_500' in prog_args:
            if not (t,s) in sampled:
                counter[t,s] = sample(counter[t,s],500)
                sampled.add((t,s))

        elif 'noisy_1000' in prog_args:
            if not (t,s) in sampled:
                counter[t,s] = sample(counter[t,s],1000)
                sampled.add((t,s))

        noise = [row_normalize(counter[t,s]).T if not s == t else no_noise() for s,t in zip(S,T)]

        if 'elkan_noise' in prog_args:
            noise = [elkan_noise(s).T for s in S]
        
        if 'noisy_none' in prog_args:
            noise = [row_normalize(counter[t,s]).T if not s == t else no_noise() for s,t in zip(S,T)]

        if 'naive' in prog_args:
            noise = [no_noise() for s,t in zip(S,T)]

        A[T], A_inv[T] = calculate_adjustment(noise)
        if np.min(A[T].sum(1)) < 10**(-9) or np.min(A[T].sum(0)) < 10**(-9):
            print 'low value warning!'
            print T, [header[t] for t in T], 'has an A matrix with values', A[T]

    marginals = {}
    #compute marginals
    for i in labels:
        if 'elkan_noise' in prog_args: #from elkan adjustment
            j = get_anchors((i,))
            marginals[i] = (counter[j][1] / float(counter[j].sum())) / elkan_calibration(j[0])

        elif 'noisy' in prog_args:
            Z = counter[(i,)]
            Z = sample(Z, 100)
            marginals[i] = Z[1].sum() / float(Z.sum())

        elif 'naive' in prog_args:
            j = get_anchors((i,))
            marginals[i] = (counter[j][1] / float(counter[j].sum()))

        else: #from ground truth
            marginals[i] = counter[(i,)][1].sum() / float(counter[(i,)].sum())
            
        print 'marg', i, header[i], marginals[i]
        print 'cond', i, header[i], counter[i, get_anchors([i])[0]]

    X = {}
    L = []
    for K in xrange(1, order+1):
        L+= list(combinations(labels,K))


    print prog_args
    if 'ground_truth' in prog_args or 'impute_10' in prog_args or 'impute_1' in prog_args or 'ground_graph' in prog_args:
        ground_truth = {}
        for key in L:
            ground_truth[key] = counter[key] / float(counter[key].sum())
        X = ground_truth
        report = []
        #'''
        #print 'working with ground_truth'
        #X_projected = {}
        #args = ((T, np.matrix(np.identity(2**len(T))), (counter[T] / float(counter[T].sum())).reshape((2**len(T),1)), marginals, gamma, 1e-9) for T in L)
        #pool = Pool(10)
        #for T,x in pool.imap(project, args):
        #    print T
        #    sys.stdout.flush()
        #    X_projected[T] = x
        #    
        #X = X_projected
        #report = []

    elif 'inversion' in prog_args:
        X_by_inversion = {}
        for T in L:
            K = len(T)
            x = np.mat(A_inv[T])*np.mat(B[K][T])
            x = np.clip(x, 10**(-6),float('inf'))
            x /= float(x.sum())
            X_by_inversion[T] = np.array(x).reshape((2,)*K)
        X = X_by_inversion
        report = []

    elif 'simplex' in prog_args:

        X_projected = {}

        args = ((T, A[T], B[len(T)][T], marginals, gamma, eps) for T in L)


        #for arg in args:
        #    T,x = project(arg)
        pool = Pool(10)
        for T,x in pool.imap(project, args):
            #print T
            sys.stdout.flush()
            X_projected[T] = x
            
        X = X_projected
        report = []

    elif 'opt' in prog_args:
        print 'optimizing'
        sys.stdout.flush()

        B_vec = np.vstack([B[order][T] for T in combinations(labels,order)])
        mats = [A[T] for T in combinations(labels,order)]
        A_mat = sparse.block_diag(mats)

        u_vec = np.zeros((sum([(2**k)*len(B[k]) for k in xrange(1,order+1)]),1))
        base = 0
        for k in xrange(1,order+1):
            for i,T in enumerate(combinations(labels,k)):
                x = np.zeros((2,)*k)
                for ind in product([0,1], repeat=k):
                    x[ind] = prod([0.5 for t,state in zip(T, ind)])
                u_vec[base:base+2**k] = x.reshape((2**k,1))
                base += 2**k
        
        m_vec = np.zeros((sum([(2**k)*len(B[k]) for k in xrange(1,order+1)]),1))
        base = 0
        for k in xrange(1,order+1):
            for i,T in enumerate(combinations(labels,k)):
                x = np.zeros((2,)*k)
                for ind in product([0,1], repeat=k):
                    x[ind] = prod([marginals[t] if state==1 else 1-marginals[t] for t,state in zip(T, ind)])
                m_vec[base:base+2**k] = x.reshape((2**k,1))
                base += 2**k
        
        #print 'A', A_mat.shape
        #print 'uvec', u_vec.size
        #print 'bvec', B_vec.size
        f = buildObjectiveFunction(A_mat, B_vec, m_vec[-(2**order)*len(B[order]):,:], gamma, order, prog_args)

        set_to_var = {}
        for K in xrange(1,order+1):
            for n,T in enumerate(combinations(labels, K)):
                set_to_var[T] = sum([(2**k)*len(B[k]) for k in xrange(1,K)])+(2**K)*n
        
        if 'integer' in prog_args:
            constraints, cutting_plane = None, None
        else:
            constraints, cutting_plane = buildConstraints(prog_args, order, set_to_var)


        if 'L1' in prog_args:
            print 'solving by L1 optimization', time.time() - start_time
            res = L1_minimization(A_mat,B_vec, np.sum([(2**k)*len(B[k]) for k in xrange(1,order)]), constraints)
            report = []

        elif any([z in prog_args for z in ('kl', 'smoothkl', 'L2')]):
            print 'solving by optimization', time.time() - start_time

            sys.stdout.flush()
            x0 = m_vec

            n0 = 0
            try:
                print 'attempt to restart from existing logfile'
                assert 0
                x0, n0 = restartCheckpoint(networkdir)
                print 'starting at', n0
                print 'x0', x0
            except Exception, e:
                print 'failed',e

            if 'tighten' in networkdir:
                max_steps = 5000
            else:
                max_steps = 5000

            def logger(n, x):
                if n % 100 == 0:
                    pickle.dump(x, file(networkdir+'/pickles/transform.latest_iterate.pk', 'w'))
                return 1

            res,score,gap,report = FW(f, constraints, x0=x0, tol=eps, integer='integer' in prog_args, max_steps=max_steps, n0=n0, fully_corrective = True, alternate_directions = [], logger=logger)


        else:
            print 'unknown optimization procedure!'
            print prog_args
            assert 0        

        #convert back
        for S, n in set_to_var.items():
            X[S]  = np.array(res[n:n+2**len(S)]).reshape((2,)*len(S))
            print [header[s] for s in S]
            print X[S]
            
            
    else:
        print 'unknown setting', prog_args
        assert 0


    print 'done solving', time.time() - start_time
    pickle.dump(report, file(networkdir+'/reports/frank_wolf_report.pk', 'w'))
    print X
    pickle.dump(X, file(networkdir+'/pickles/estimated_moments.'+'.'.join(prog_args)+'.simple_anchors'+str(sample_size)+'.'+str(order)+'.pk', 'w'))

    anchor_dict = dict(zip(labels, [get_anchors([l])[0] for l in labels]))
    pickle.dump(anchor_dict,file(networkdir+'/pickles/anchors.pk', 'w'))

    noise = {}
    for t,s in anchor_dict.items():
        noise[s] = row_normalize(counter[t,s]+10**(-6)).T

    pickle.dump(noise, file(networkdir+'/pickles/noise.pk', 'w'))
    print 'done creating pickles'

