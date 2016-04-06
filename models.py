from __future__ import division
import networkx as nx
import copy
import numpy as np
import sys
from helpers import *
import time
from VariableElimination import eliminate
import compute_bic_score
import os


def optimizeStructure(counts, header, labels, N=1, seed=1):

    null = file(os.devnull, 'w')
    tags = [header[t] for t in labels]

    np.random.seed(seed)
    T = str(np.random.random_integers(5000,100000))

    tempfile = file('temp.'+T+'.bic', 'w')
    order = 2
    scores = compute_bic_score.main(counts, header, labels, tempfile, order, N)
    tempfile.close()

    #learn structure
    subprocess.call(['./gobnilp', 'temp.'+T+'.bic'], stdout=null)
    edges = extract_edges('temp.'+T+'.bn_mat')

    marginals = counts
    CPDs = {}
    #cleanup
    print T
    subprocess.call('rm temp.'+T+'.*', shell=1)
    for j in labels:
        parents = [z for z in labels if (labels.index(z),labels.index(j)) in edges]
        t,s = tags[labels.index(j)], tuple([tags[labels.index(i)] for i in parents])
        key = tuple([j] + parents)
        first_index = sorted(key).index(j)
        other_indices = [sorted(key).index(p) for p in parents]
        transpose_order = tuple([first_index]+other_indices)

        m = marginals[tuple(sorted(key))].transpose(transpose_order)
        smoothing = 0
        CPDs[t,s] = create_CPD(m, smoothing=smoothing)

    tree = TreeModel(CPDs, #dictionary of CPDs
            tags, #variable ids
            format="CPDs" #latent structure already holds cpds
            )
    counter = get_counts(marginals, header)
    tree.addCounts(counter)
    return tree

def bernoulli(p):
    if type(p) == float:
        return int(np.random.rand() < p)
    else:
        return np.array(np.random.rand(*p.shape) < p,dtype=int)


def infer(self, data, conditioning=None, mode="gibbs", debug=False):
    np.random.seed(100)
    start = time.time()
    model = self
    '''
    do inference
        model -- has latent_tree, failures, noise
        data -- list of [-1,1,0], 0 means unobserved or don't know
        mode -- defaults to gibbs sampling
        conditioning -- condition on Y variables
    '''

    data = dict(zip(self.observations, data))

    failures = model.failures
    noise = model.noise
    for k,val in noise.items():
        noise[k] = np.clip(val, 0,0.999)
    N = len(model.latents)
    M=  len(model.observations)
    CPD = model.CPD
    parents = model.parents_of
    children = model.children_of

    burnin = 1000
    spacing = 1
    passes = 500

    #initialize X to all 0s
    X = dict(zip(self.latents, np.array([0]*N)))
    conditioned = set()
    for i,val in conditioning:
        X[i] = val
        conditioned.add(i)
        assert(i in self.latents)
        #print 'conditioning on', i, 'having value', val
        
    samples = np.zeros(N)
    sample_count = 0

    #locate positive and negative symptoms
    posDataIndices = set([i for i in self.observations if data[i] > 0])
    negDataIndices = set([i for i in self.observations if data[i] < 0])

    if debug:
        print 'pos data indices', posDataIndices
        print 'neg data indices', negDataIndices

    #print 'pos data indices', posDataIndices
    negFindingCache = {}
    negFindings = prod([noise[j] for j in negDataIndices])

    for i in self.latents:

        negFindingCache[i] = prod([failures[i,j] for j in negDataIndices])
        if debug:
            print 'negFindingCache', i, negFindingCache[i]
        children[i] = set(children[i]) | set([j for j in self.observations if failures[i,j] < 0.999])
        if X[i] > 0:
            negFindings *= negFindingCache[i]

    def updateNegFindings(negFindings, i,oldVal,newVal):
        if oldVal == newVal:
            if debug:
                print 'short circuit'
            return negFindings
        retVal = float(negFindings)

        if oldVal < newVal:
            retVal *= negFindingCache[i]
            if debug:
                print 'turning on, pay a penalty of', negFindingCache[i], 'for neg observations'
        if newVal < oldVal:
            retVal /= negFindingCache[i]
            if debug:
                print 'turning off, get a gain of of', negFindingCache[i], 'for neg observations'

        return retVal

    posFindingCache = {}
    for j in posDataIndices:
        posFindingCache[j] = noise[j]*prod([failures[i,j] for i in self.latents if X[i] > 0])

    if debug:
        print 'posFindingCache', posFindingCache
    posFindings = prod([(1-posFindingCache[j]) for j in posDataIndices])

    def updatePosFindings(posFindings, i,oldVal,newVal,updateCache=False):
        if debug:
            print "update posFindings -- disease", i, "changes from", oldVal, "to", newVal
        if oldVal == newVal:
            if debug:
                print "short circuit"
            return posFindings
        
        retVal = float(posFindings)
        
        if debug:
            print "posDataIndices", posDataIndices

        for j in (posDataIndices & children[i]):
            if debug:
                print "evaluating child", j
            temp = float(posFindingCache[j])
            retVal /= (1-temp)
            if oldVal < newVal:
                temp *= failures[i,j]
            if newVal < oldVal:
                temp /= failures[i,j]
            if updateCache:
                posFindingCache[j] = temp
            retVal *= (1-temp)
        
        return retVal
    
    if debug:
        print "init X", X
    #print 'initialized', time.time()-start
    rand_index = 0
    randstring = np.random.rand((burnin+passes)*len(self.latents))
    for sample in xrange(burnin + passes):

        for i in self.latents:
            if i in conditioned:
                continue
            #start = time.time()    
            if debug:
                print 'current X is', X
                print i, "X[i] is", X[i]
        
            oldX = float(X[i])
            
            P = updatePosFindings(posFindings,i,X[i],0)
            Q = updateNegFindings(negFindings,i,X[i],0)
            Ra = CPD[i][tuple([0]+[X[s] for s in parents[i]])]
            Rb = prod([CPD[s][tuple([X[s]]+[0 if j == i else X[j] for j in parents[s]])] for s in children[i] if s in self.latents])
            p0 = P*Q*Ra*Rb

            if debug:
                print 'p0', P, Q, 'Ra', [CPD[i][X[s],0] for s in parents[i]], Ra,  'Rb', [CPD[s][0,X[s]] for s in children[i] if s in self.latents], Rb
                print 'p0 final val', p0
            P = updatePosFindings(posFindings,i,X[i],1)
            Q = updateNegFindings(negFindings,i,X[i],1)
            Ra = CPD[i][tuple([1]+[X[s] for s in parents[i]])]
            Rb = prod([CPD[s][tuple([X[s]]+[1 if j == i else X[j] for j in parents[s]])] for s in children[i] if s in self.latents])

            p1 = P*Q*Ra*Rb
            if debug:
                print 'p1', P, Q, 'Ra', [CPD[i][X[s],1] for s in parents[i]], Ra,  'Rb', [CPD[s][1,X[s]] for s in children[i] if s in self.latents], Rb
                print 'p1 final val', p1
            p = p1 / (p0 + p1)
            if debug:
                print 'p', p
            #print 'compute', time.time()-start
            #print "prob of turning", i, "on", p
            X[i] = int(randstring[rand_index] < p)
            rand_index += 1
            #print 'sample', time.time()-start
            
            posFindings = updatePosFindings(posFindings,i,oldX, X[i], updateCache=1)
            negFindings = updateNegFindings(negFindings,i,oldX, X[i])
            #print 'update', time.time()-start

        if sample > burnin and (sample-burnin) % spacing == 0:                
            samples += np.array([X[i] for i in self.latents])
            sample_count += 1
            

    end = time.time()
    #print "total time", end-start, 'seconds'

    return samples.T / float(sample_count)


class TreeModel:

    def __init__(self, CPDs, latents, format='potentials'):
        #INPUT FORMAT:
        #edges: dictionary structure
        #       keys - binary, unary cliques
        #       values - potentials

        #latents: variable names
        self.edges = []
        self.root = []
        self.directed_edges = []
        self.children_of = None
        self.parents_of = None
        self.observations = []
        self.counters = {}
        self.noise = {}
        self.failures = {}
        self.structured = False
        self.latents = latents
        self.latent_lookup = dict(zip(latents, xrange(len(latents))))
        self.CPD = CPDs
        self.lCPD = {}

        if format=='potentials':
            self._initializeStructure(self.latents[0])
        
        if format=='CPDs':
            self.structured = True
            self.parents_of = {}
            self.children_of = defaultdict(list)
            for l,parent_set in self.CPD.keys():
                if len(parent_set) == 0:
                    self.root.append(l)

                self.CPD[l] = self.CPD[l,parent_set] #shorthand for CPD of l conditioned on parents
                #print self.CPD[l].sum(0), 'should be ones!'
                self.parents_of[l] = parent_set
                for p in parent_set:
                    self.children_of[p].append(l)
                    self.directed_edges.append((p,l))

                for k in itertools.product([0,1], repeat=len(parent_set)):
                    assert np.abs(self.CPD[l][tuple([slice(None)]+list(k))].sum() -1) < 1e-9, 'improper CPD :'+ str(self.CPD[l][tuple([slice(None)]+list(k))]) + ' ' + str(l) + ':' + str(parent_set) + '--' + str(self.CPD[l]) + '::::' + str(self.CPD[l][tuple([slice(None)]+list(k))].sum())
                self.lCPD[l] = np.log(self.CPD[l])

        depth = defaultdict(int)
        stack = list(self.root)
        for r in self.root:
            depth[r] = 0

        while len(stack):
            p = stack.pop()
            for c in self.children_of[p]:
                depth[c] = max(depth[c], depth[p] + 1)
                stack.append(c)
        self.depth = depth


        # print self.latents
        # print self.observations
        # print 'children', self.children_of
        # print 'parents', self.parents_of
        # print 'directed edges', self.directed_edges
        # print 'root is', self.root
        # print 'depths are', self.depth


    def descendants(self, L):
        D = []
        stack = []
        print 'listing descendants of', L
        for c in self.children_of[L]:
            stack.append(c)
        while len(stack):
            l = stack.pop()
            D.append(l)
            if l in self.children_of:
                for c in self.children_of[l]:
                    stack.append(c)
        return D

    def coparents(self, L):
        C = set()
        for c,parents in filter(lambda k: type(k) is tuple and len(k) == 2, self.CPD):
            if L in parents:
                C |= set(parents)
        C.discard(L)
        return list(C)

    def _initializeStructure(self,root):
        self.root, self.directed_edges, self.children_of, self.parents_of = rootedTree(self.latents, self.edges, root=root)
        self.CPD = {}
        self.lCPD = {}
        for i,l in enumerate(self.latents):
            self.CPD[l] = np.zeros((2,2))
            self.lCPD[l] = np.zeros((2,2))
            for parent_state in [0,1]:
                parent = set(self.parents_of[l])
                factor = 1.0
                if len(parent):#parent
                    parent = parent.pop()
                    factor *= self.directed_edges[(parent, l)][parent_state, :]
                else:
                    parent = None
                    for c in self.children_of[l]:#children
                        factor *= sum([self.directed_edges[(l,c)][:,i]*self.edges[c][i] for i in [0,1]])
                factor *= self.edges[l] #unary potential
                self.CPD[l][parent_state,:] = normalize(factor)
                self.lCPD[l][parent_state,:] = np.log(normalize(factor))

        self.structured = True

    def addAnchors(self, D, failures, noise):
        self.anchors = D
        for l in self.latents:
            self.observations.append(D[l])
            self.failures[l,D[l]] = failures[l, D[l]]
            
            for t in self.latents:
                if not t == l:
                    self.failures[t, D[l]] = 1.0
                

            self.noise[D[l]] = noise[D[l]]

    def addObservations(self, L):
        for l in L:
            self.observations.append(l)

    def addCounts(self, counts):
        for k in counts:
            assert np.max(counts[k]) <= 1.0 + 10**(-6), str(k) + ' ' + str(counts[k])
            assert np.min(counts[k]) >= 0.0 - 10**(-6), str(k) + ' ' + str(counts[k])
        self.counts = counts

    def addResiduals(self, residuals):
        self.residuals = residuals


    # def prob(self, X, condition=None):
    #     X = sorted(X)
    #     if not condition==None:
    #         a = self.prob(X+condition)
    #         b = self.prob(condition)
    #         p =  a/b 
    #         print 'prob', X, condition, a, '/', b, '=', p 
    #     else:
    #         var,val = zip(*X)
    #         joint = self.counts[tuple(var)]

    #         p = joint[tuple(val)]
    #         #print 'joint', joint
    #         print 'prob', X, condition, '=', p
    #     assert p <= 1+10**(-6), str(X)+"|"+str(condition) + ':'+str(p)
    #     return np.clip(p, 10**(-9), 1-10**(-9))
        
    def prob(self, X, condition=None):
        if condition == None:
            condition = []

        var,val = zip(*sorted((X+condition)))
        joint = self.counts[tuple(var)]


        #print 'prob', X, condition, 'joint', joint, var
        if len(condition):
            C_var, C_val = zip(*sorted(condition))
            cond_val = tuple([val[i] if var[i] in C_var else slice(None) for i in xrange(len(var))])
            joint = joint[cond_val]
            if joint.sum() < 1e-4:
                print 'warning, conditioning event has low probability:', X, condition, joint.sum()
                return -1

            joint = joint / joint.sum()
            var,val = zip(*sorted(X))

        p = joint[tuple(val)]
        #print 'prob res', X, condition, '=', p

        assert p <= 1+10**(-6), str(X)+"|"+str(condition) + ':'+str(p)

        return np.clip(p, 10**(-9), 1-10**(-9))

    def estimateCrossEdges(self, method='moments-general', min_fail=0, max_fail=1, do_checks=False, ignore_correction=False, min_noise=1.0):
        if method=='moments-general':
            top_sort = sorted(self.latents, key=lambda l: self.depth[l], reverse=True)
            for L in top_sort:
                print '-'*20
                print "latent variable ", L
                P = list(self.parents_of[L]) #ASSUME THERE IS ONLY 1 parent
                for X in self.observations:
                    if X in self.anchors.values():
                        continue

                    print 'looking at child', X
                    
                    if len(P):
                        parent_condition = max(itertools.product([0,1], repeat=len(P)), key = lambda cond: self.prob(zip(P, cond)))
                        parent_condition = zip(P, parent_condition)
                        print 'conditioning on parents = ', parent_condition
                    else:
                        print 'no parents'
                        parent_condition = []
                        
                    f_num = self.prob([(X,0)], condition=[(L,1)]+parent_condition) 
                    f_denom = self.prob([(X,0)], condition=[(L,0)]+ parent_condition)
                    
                    if np.isnan(f_num) or np.isnan(f_denom):
                        self.failures[(L,X)] = 1.0
                        continue

                    f = f_num / f_denom


                    print 'ratio', f
                    D = self.descendants(L) 
                    print 'subtracting off prior influence of descendants and coparents', D
                    num_correction = self.general_correction(L,1,D,X, explicit_check=do_checks)
                    denom_correction = self.general_correction(L,0,D,X, explicit_check=do_checks)

        
                    print 'correction', num_correction , '/', denom_correction, '=', num_correction / denom_correction

                    f /= num_correction 
                    f *= denom_correction

                    print 'res', f

                    if f < max_fail:
                        self.failures[(L,X)] = f
                    else:
                        self.failures[(L,X)] = 1.0


            L = self.root
            for X in self.observations:
                if X in self.anchors.values():
                    continue
                #self.noise[X] =  self.prob([(X,0)]) / self.evaluateProb((X,0))
                a = self.prob([(X,0)])
                b = self.general_correction(self.root, slice(None), self.descendants(self.root), X).dot(self.CPD[self.root])
                print X
                print 'a', a
                print 'b', b
                self.noise[X] =  a/b

        if method=='moments-tree':
            for L in self.latents:
                print 'latent variable', L
                for X in self.observations:
                    print '\tobservation', L, X
                    if X in self.anchors.values():
                        print '\t\t', X, 'is an anchor'
                        continue

                    num = self.prob([(X,0)], condition=[(L,1)])
                    denom = self.prob([(X,0)], condition=[(L,0)])
                    if num < 0 or denom < 0:
                      f = None
                    else:
                      f = num / denom
                    print '\tuncorrected failure', num, '/', denom, '=', f
                    print '\tcounts', self.counts[tuple(sorted([X,L]))]
                    print '\tsums', self.counts[tuple(sorted([X,L]))].sum(0), self.counts[tuple(sorted([X,L]))].sum(1)

                    if f is None:
                      f = 1
                    else:
                      for B in list(self.children_of[L]) + list(self.parents_of[L]):
                          print '\t\tcorrecting for', B
                          corr = self.correction(B,L,X,alternate=False)
                          print '\t\tcorrection is', corr
                          alternate_correction = self.correction(B,L,X,alternate=True)
                          print '\t\t(alternate correction is:', alternate_correction, ')'
                          #if corr > 1:
                          #    print 'ignoring positive correction'
                              
                          if ignore_correction:
                              print 'ignoring correction'
                              pass
                          elif corr < 0:
                              print 'ignoring correction < 0'
                              pass
                          else:
                              f/=corr
                          print '\t\tnew failure', f
                      print '\t\tfinal failure', f

                    if f < max_fail:
                        self.failures[(L,X)] = f
                    else:
                        self.failures[(L,X)] = 1.0

            for X in self.observations:
                if X in self.anchors.values():
                    continue

                a = self.prob([(X,0)])
                #b = self.general_correction(self.root, slice(None), self.descendants(self.root), X).dot(self.CPD[self.root])#self.evaluateProb((X,0))
                b = self.evaluateProb((X,0))
                print X
                print 'a', a
                print 'b', b
                #assert a <= b
                self.noise[X] = min(a/b, min_noise)
                
    def evaluateProb(self, X):
        #uses belief propagation -- assumes tree structure
        message = {}
        X,index = X
        top_sort = sorted(self.latents, key=lambda l: self.depth[l], reverse=True)
        final_messages = []
        for L in top_sort:
            #print L
            for p in self.parents_of[L]:
                #print [message[c,L] for c in self.children_of[L]]
                m = prod([message[c,L] for c in self.children_of[L]])
                #print m
                m *= np.array([[1.0],[self.failures[L,X]]]) 
                #print m
                #print self.CPD[L].T
                m = np.matrix(self.CPD[L].T) * np.matrix(m)
                #print m
                message[L, p]  = np.array(m)
                #print L,p
                #print message[L,p]

                assert all([z <= 1.0+1e-6 for z in message[L,p]])

            if self.depth[L] == 0:
                m = prod([message[c,L] for c in self.children_of[L]])
                m *= np.array([[1.0],[self.failures[L,X]]]) 
                m = np.matrix(self.CPD[L].T) * np.matrix(m)
                #print L, m
                final_messages.append(m)

        #print final_messages
        m = prod(final_messages)
        return m[index]

    def correction(self, B,A,X, alternate=False):

        if not alternate:
            denom = self.prob([(X,0)], condition=[(A,0)])
            a =  self.prob([(B,0)], condition=[(A,1)])
            b =  self.prob([(X,0)], condition=[(A,0), (B,0)]) 
            c =  self.prob([(B,1)], condition=[(A,1)])
            d =  self.prob([(X,0)], condition=[(A,0), (B,1)])
            numerator = a*b + c*d
            print 'num', a,'*',b, '+', c, '*', d
            print 'denom', denom

        else:
            numerator = self.prob([(X,0)], condition=[(A,1)])
            a =  self.prob([(B,0)], condition=[(A,0)])
            b =  self.prob([(X,0)], condition=[(A,1), (B,0)]) 
            c =  self.prob([(B,1)], condition=[(A,0)])
            d =  self.prob([(X,0)], condition=[(A,1), (B,1)])
            denom = a*b+c*d

        if any([a < 0, b<0, c<0, d<0]):
            return -1

        return numerator/denom

    def tree_correction(self,L0,L0_val,D,X):
        if not len(D): #L0 is a leaf node
            return 1.0

        message = {}
        top_sort = sorted(D, key=lambda l: self.depth[l], reverse=True)

        for L in top_sort:
            for p in self.parents_of[L]:
                m = np.array([[1],[self.failures[L,X]]]) 
                m *= prod([message[c,L] for c in self.children_of[L]])
                m = np.matrix(self.CPD[L].T) * np.matrix(m)
                message[L, p]  = np.array(m)

        corr = prod([message[L,L0] for L in self.children_of[L0]])
        return corr[L0_val]


    def general_correction(self,L0,L0_val,D,X, explicit_check=False):

        #computes \sum_{Y in D} P(Y|L0=L0_val) P \prod_{Yk in D} f_k^{y_k}
        #corrosponds to prob(X=0) after accounting for (Y in D) and knowing that L0=L0_val
                
        if not len(D): #L0 is a leaf node
            return np.array([1.0, 1.0])[L0_val]
        
        if explicit_check:
            print 'explicit check, correcting for', D
            check_val = 0
            for D_val in itertools.product([0,1], repeat=len(D)):
                temp = self.prob(X=zip(D, D_val), condition=[(L0, L0_val)])
                print '\t', temp,
                for d in xrange(len(D)):
                    if D_val[d] == 1:
                        temp  *= self.failures[D[d],X]
                print temp
                check_val += temp
            print check_val

        factors = []
        latents = set()
        for d in D:
            latents.add(d)
            latents |= set(self.parents_of[d])

            factors.append([[d]+list(self.parents_of[d]), self.CPD[d]])
            factors.append([[d], np.array([1, self.failures[d,X]])])

        latents.discard(L0)
        #print 'eliminate', factors

        top_sort = sorted(latents, key=lambda l: self.depth[l], reverse=True)
        for L in top_sort:
            if L == L0:
                continue

            print 'eliminating', L
            factors = eliminate(L, factors)

        #print 'the remaining factors are', factors
        #print 'they should only have', L0
        ret = factors[0][1]

        for f in factors[1:]:
            ret *= f[1]

        for r in ret:
            assert r < 1 + 1e-6, 'how can a correction be > 1? :'+str(r)
        #check = self.tree_correction(L0,L0_val,D,X)
        #assert np.abs(ret[L0_val] - check) < 10**(-6), str(ret[L0_val]) + ' not equal ' + str(check)
        if type(L0) is int:
            assert ret[L0_val].size == 1 
        


        if explicit_check:
            try:
                assert ret[L0_val] == check_val, "failed check "+str(ret[L0_val])+"!="+str(check_val)
            except:
                print 'failed check', ret[L0_val], '!=', check_val
                return check_val

        return ret[L0_val]

    def createAdjustment(self, X):
        n = len(X)
        A = np.matrix(np.zeros(2**n, 2**n))
        for r_index, r in enumerate(itertools.product([0,1], repeat=n)):
            for s_index, s in enumerate(itertools.product([0,1], repeat=n)):
                A[r_index,s_index] = prod([self.noise[X[i]][r[i],s[i]] for i in xrange(n)])

        A = np.matrix(A)
        A_inv =  np.linalg.pinv(A)
        return A_inv

    def write_graph(self, filename, problem_edges=[], max_edge=0.99):
        G = nx.DiGraph()
        print 'writing graph!'
        print 'internal edges', self.directed_edges

        for o in self.observations:
            if o in self.anchors.values():
                G.add_node(o, style='filled', color='red')
            else:
                G.add_node(o, style='filled', color='gray')

        for e in self.directed_edges:
            i = e[0]
            j = e[1]
            print ''
            if e in problem_edges:
                G.add_edge(i, j, color='red')
            else:
                G.add_edge(i, j, color='blue')
        for e in self.failures:
            i = e[0]
            j = e[1]
            if self.failures[e] < max_edge:
                G.add_edge(i, j, color='green', weight=(1-self.failures[e])*10)


        nx.write_dot(G, filename)


    def describe(self, threshold=1):
        for o in sorted(self.failures.items(), key=lambda f:f[1]):
            if o[1] < threshold:
                print o
        for o in self.noise.items():
            if o[1] < threshold:
                print o

    def eval_likelihood(self, Y, debug=False, verbose=False, accept_check=False, blacklist=[], do_check=True):

        lprob = 0
        for i,l in enumerate(self.latents):
            if l in blacklist:
                continue

            parents = list(copy.copy(self.parents_of[l]))
            parent_states = []
            for p in parents:
                if p in blacklist:
                    print "warning: blacklisted parent", p, "is a parent of", l, "what should we do??"
                    sys.exit()

                parent_index = self.latent_lookup[p]
                parent_states.append(Y[parent_index])
            
            key = tuple([Y[i]]+parent_states)

            val = self.CPD[l][key]
            if verbose == True and self.lCPD[l][key] < -2:
                print l, '|', parents, key, self.lCPD[l][key], self.CPD[l]
                print '\n'
            if do_check:
                check =  self.prob([(l,Y[i])], condition=zip(parents, parent_states)) 
            else:
                check = val

            if accept_check:
                val = check
                lprob += np.log(val)

            elif not np.abs(val - check) < 10**(-6):
                print 'val', val
                print 'check', check

                print 'l', l
                print 'parents', parents
                print 'key', key
                print 'i', i
                print 'Y[i]', Y[i]
                print 'this is a big problem?'

                print '\n\n'
                print 'cpd', self.CPD[l]

                sys.exit()
            #print (l, Y[i]), zip(parents, parent_states)
            #print 'compare to', self.prob([(l,Y[i])], condition=zip(parents, parent_states))
            #print 'should be the same...'

            else:
                lprob += self.lCPD[l][key]

        if debug==True:
            if not approx_equal(self.debug_likelihood(Y), np.exp(lprob), 10**-6):
                print "error!", self.debug_likelihood(Y), np.exp(lprob), Y
            else:
                print "correct", self.debug_likelihood(Y), np.exp(lprob), Y 
        return lprob

    def debug_likelihood(self,Y):
        D = np.zeros((2,)*len(Y), dtype=float)
        for t in itertools.product([0,1], repeat=len(Y)):
            D[tuple(t)] = self.unnormalized_likelihood(t)
        D /= D.sum()
        return D[tuple(Y)]

    def unnormalized_likelihood(self, Y):
        prob = 1
        for e,val in self.edges.items():
            if type(e) == tuple:
                i = self.latents.index(e[0])
                j = self.latents.index(e[1])
                prob *= val[Y[i], Y[j]]
            else:
                i = self.latents.index(e)
                prob *= val[Y[i]]

        return prob

    

        
      
        
if __name__ == "__main__":

    
    CPDs = {}
    tags = ['var'+str(i) for i in range(4)]
    CPDs['var0', tuple()] = np.array([0.75, 0.25])
    CPDs['var1', tuple()] = np.array([0.75, 0.25])

    m = np.zeros((2,2,2))

    m[0, 1,1] = 0.9
    m[1, 1,1] = 0.1

    m[0, 1,0] = 0.5
    m[1, 1,0] = 0.5

    m[0, 0,1] = 0.5
    m[1, 0,1] = 0.5

    m[0, 0,0] = 0.1
    m[1, 0,0] = 0.9

    CPDs['var2', ('var0','var1')] = copy.deepcopy(m)
    CPDs['var3', ('var0','var1')] = copy.deepcopy(m)

    T = TreeModel(CPDs, tags, format="CPDs")
    T.addObservations(['var4'])
    T.anchors = {}

    failure = 0.1
    noise = 1

    counts = np.zeros((2,)*5)
    for Y in itertools.product([0,1], repeat=4):
        py = np.exp(T.eval_likelihood(Y, do_check=False))
        print 'likelihood', Y, py
        counts[tuple(list(Y)+[0])] = py * failure**(sum(Y))
        counts[tuple(list(Y)+[1])] = py *(1- failure**(sum(Y)))

    counter = {}
    for K in xrange(1,6):
        for S in itertools.combinations(['var'+str(i) for i in xrange(5)], K):
            Sprime = sorted(set(['var'+str(i) for i in xrange(5)]) - set(S))
            V = counts.sum(tuple([int(s.replace('var', '')) for s in Sprime]))
            counter[tuple(sorted(S))] = V

    for k in sorted(counter.items()):
        print k

    T.addCounts(counter)
    print 'P(X=0|Y_0=0)', T.prob([('var4', 0)], [('var0', 0)])
    temp = 0
    for Y in itertools.product([0,1], repeat=3):
        temp += T.prob(zip(['var'+str(i) for i in [1,2,3]], Y), [('var0',0)]) * failure**sum(Y)
    print 'sum_y123 P(y123 | y0=0)\\prod f^yi)', temp 
    
    temp = 0
    for Y1 in [0,1]:
        temp2 = 0
        for (Y2,Y3) in itertools.product([0,1], repeat=2):
            temp2 += T.prob([('var2',Y2), ('var3', Y3)] , [('var0',0), ('var1',Y1)]) * failure**(Y2+Y3)
        temp += temp2 * T.prob([('var1', Y1)]) * failure**Y1

    print '(sum_y1 P(y1)f^y1)(sum_y23 P(y23|y0=0, y1) prod f^yi)', temp


    

    #T.estimateCrossEdges('moments-general', 0, 1, do_checks = True)
    T.describe()
