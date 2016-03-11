import time
from helpers import logsumexp
import cPickle as pickle
from collections import defaultdict
import sys 
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free


def marginal_likelihood(self, list data, int passes, int seed):
    '''
    do inference
        model -- has the following fields:
                observations
                latents

                Related to the structure of the latent variables:
                  CPD (if latents are independent, each CPD is simply [P(Y=0), P(Y=1)])
                  parents_of (if latents are independent, this is a list of empty sets)
                  children_of (if latents are independent, this is a list of empty sets)

                Related to the edges from latent to observed:
                  failures
                  noise

        data -- list of list of ints in [-1,1,0], 0 means unobserved or don't know
        mode -- defaults to gibbs sampling
        conditioning -- condition on Y variables
    '''

    debug = False
    np.random.seed(seed)

    start = time.time()
    model = self
    CPD = model.CPD
    parents_of = model.parents_of
    children_of = model.children_of
    sample_list = []

    observed_children = []
    relevant_parents = set()
        

    #start = time.time()
    cdef np.ndarray failures = model.failures
    #cdef np.ndarray lfailures = model.lfailures
    cdef np.ndarray noise = model.noise
    cdef int L = len(model.latents)
    cdef int O=  len(model.observations)

    cdef int burnin = 100
    cdef int spacing = 5
    #cdef int passes = 1000
    
    #counters
    cdef int sample_count = 0
    cdef int sample = 0
    cdef s = 0
    cdef int ii,i,j

    cdef double negFindings = 0
    cdef double posFindings = 0
    cdef double latentlikelihood = 0

    #value holders
    cdef int oldY
    cdef double p0
    cdef double p1
    cdef double total = 0

    
    #initialize Y and caches
    Y = [0]*L
    cdef double* negFindingCache = <double *>malloc(L*sizeof(double))
    cdef double* Ycounter = <double *>malloc(L*sizeof(double))
    cdef double* posFindingCache = <double *>malloc(O*sizeof(double))
    cdef int* parent_state = <int *>malloc(L*sizeof(int))

    cdef int rand_index = 0
    
    for i from 0 <= i < L:
        Y[i] = 0
        Ycounter[i] = 1.0
        negFindingCache[i] = -99

    #locate positive and negative symptoms
    posDataIndices = set([i for i in self.observations if data[i] > 0])


    for j in posDataIndices:
        for i from 0 <= i < L:
            if failures[i,j] < 0.95:
                if not i in relevant_parents:
                    #print model.latent_names[i], 'is relevant because of observation', model.observation_names[j]
                    relevant_parents.add(i)

    #print 'relevant_parents', len(relevant_parents)
    negDataIndices = set([i for i in self.observations if data[i] < 0])

    #print 'pos data indices', posDataIndices
    #print 'neg data indices', negDataIndices


    #initialize negative findings val
    for j from 0 <= j < O:
        if data[j] < 0:
            negFindings += log(noise[j])

    #initialize neg findings cache
    for i from 0 <= i < L:
        observed_children.append(set())
        negFindingCache[i] = 0
        for j from 0 <= j < O:
            if failures[i,j] < 1.0:
                observed_children[i].add(j)
            if data[j] < 0:
                negFindingCache[i] += log(failures[i,j])
                if debug:
                    print 'negFindingCache', i, negFindingCache[i]
        

    #initialize pos findings cache
    posFindings = 0
    for j from 0 <= j < O:
        posFindingCache[j] = log(noise[j])
        if data[j] > 0:
            posFindings += log(1-exp(posFindingCache[j]))

    for i from 0 <= i < L:
        parent_state[i] = 0
        assert CPD[i].size == 2**(len(parents_of[i])+1)
        latentlikelihood += log(CPD[i][0])
        assert np.abs(CPD[i][parent_state[i] + 0] + CPD[i][parent_state[i] + 1] - 1) < 1e-9, CPD[i][parent_state[i] + 0] + CPD[i][parent_state[i] + 1]

    def updateLatentLikelihood(double likelihood, int i, int oldVal, int newVal, int updateCache=False):
        if oldVal == newVal:
            return likelihood

        cdef double retVal = likelihood
        cdef int cVal = -1
        cdef int p_index = -1
        cdef temp = 0
        cdef adjustment = 0
        
        #parent factor
        p_index = parent_state[i]    
        retVal -= log(CPD[i][p_index + oldVal])
        retVal += log(CPD[i][p_index + newVal])
        
        #children factors
        for c in children_of[i]:
            p_index = parent_state[c]
            cval = Y[c]
            retVal -= log(CPD[c][p_index + cval])

            temp = parent_state[c]
            adjustment = 0
            for p in parents_of[c]:
                if p == i:
                    adjustment += 1
                adjustment *= 2

            if newVal > oldVal:
                temp += adjustment
            else:
                temp -= adjustment

            if updateCache:
                parent_state[c] = temp 
            
            p_index = temp
            retVal += log(CPD[c][p_index + cval])

        return retVal

    def updateNegFindings(double negFindings, int i,int oldVal,int newVal):
        cdef double retVal

        if oldVal == newVal:
            if debug:
                print 'short circuit'
            return negFindings

        retVal = float(negFindings)

        if oldVal < newVal:
            retVal += negFindingCache[i]
            if debug:
                print 'turning on, pay a penalty of', negFindingCache[i], 'for neg observations'
        
        if newVal < oldVal:
            retVal -= negFindingCache[i]
            if debug:
                print 'turning off, get a gain of of', negFindingCache[i], 'for neg observations'
        
        return retVal
    
    def updatePosFindings(double posFindings, int i, int oldVal,int newVal, int updateCache=False):
        if debug:
            print "update posFindings -- disease", i, "changes from", oldVal, "to", newVal
        if oldVal == newVal:
            if debug:
                print "short circuit"
            return posFindings
        

        cdef double retVal = posFindings
        cdef double temp
        cdef int j
        
        if debug:
            print "posDataIndices", posDataIndices

        for j in posDataIndices & observed_children[i]:
            if debug:
                print "evaluating child", j

            temp = posFindingCache[j]

            retVal -= log(1-exp(temp))

            if oldVal < newVal:
                temp += log(failures[i,j])

            if newVal < oldVal:
                temp -= log(failures[i,j])

            if updateCache:
                posFindingCache[j] = temp

            retVal += log(1-exp(temp))

        return retVal
    
    randstring = np.random.rand((burnin+passes)*L*2)
    #print randstring[:10]

    for sample from 0 <= sample < burnin + passes:
        round_start = time.time()
        order = range(L)
        np.random.shuffle(order)

        for ii from 0 <= ii < L:
            if debug:
                print 'current Y', Y
            i = order[ii]


            oldY = Y[i]
            
            t = time.time()
            P = updatePosFindings(posFindings,i,Y[i],0)
            Q = updateNegFindings(negFindings,i,Y[i],0)
            R = updateLatentLikelihood(latentlikelihood, i,Y[i],0)

            p0 = P+Q+R
            p0P = P
            p0Q = Q
            p0R = R
            if debug:
                print 'p0 PQR', p0, P, Q, R
                print 'p0 final val', p0

            P = updatePosFindings(posFindings,i,Y[i],1)
            Q = updateNegFindings(negFindings,i,Y[i],1)
            R = updateLatentLikelihood(latentlikelihood, i,Y[i],1)

            p1 = P+Q+R
            if debug:
                print 'p1 PQR', p1, P, Q, R
                print 'p1 final val', p1

            larger = p1
            if p0 > p1:
                larger = p1

            p = exp(p1 - (larger + log(exp(p0-larger) + exp(p1-larger))))

            Y[i] = int(randstring[rand_index] < p)
            rand_index += 1
            
            posFindings = updatePosFindings(posFindings,i,oldY, Y[i], updateCache=1)
            negFindings = updateNegFindings(negFindings,i,oldY, Y[i])
            latentlikelihood = updateLatentLikelihood(latentlikelihood, i,oldY, Y[i], updateCache=1)

        #print 'round took', time.time()-round_start
        
        if sample >= burnin and (sample-burnin) % spacing == 0:

            for i from 0 <= i < L:
                Ycounter[i] += Y[i] 

    marginals = []
    for i from 0 <= i < L:
        marginals.append(Ycounter[i] / (float(passes / spacing) + 1.0))

    #print len(set(likelihood))
    free(negFindingCache)
    free(posFindingCache)
    free(parent_state)
    free(Ycounter)

    return marginals
