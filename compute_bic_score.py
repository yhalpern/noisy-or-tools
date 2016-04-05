from __future__ import division
import time
import sys
from collections import defaultdict
import itertools
import cPickle as pickle
import numpy as np

def quote(w):
    return '"'+w+'"'
def subsets(L):
    ret = [tuple()]
    for k in xrange(len(L)+1):
        ret += list(itertools.combinations(L, k+1))
    return ret



def main(moments, header, labels, outfile, order, N, attractive=False):
   

    print 'moment keys', len(moments.keys())
    print 'labels', labels
    sys.stdout.flush()

    def information(X, parents):
        start = time.time()
        #print 'check00', time.time() - start
        try:
            key = tuple(sorted(list(parents)+[X])) #filter(lambda k: set([X]+list(parents)) == set(k), moments)[0]
            assert key in moments
        except:
            return None
        #print 'check0', time.time() - start

        #print 'information', X, parents

        D = moments[key]
        i = key.index(X)
        #print 'D', D 
        #print 'i', i
        #print 'parents', parents
        #print 'transpose', tuple([i] + [z for z in xrange(len(parents)+1) if not z == i])

        D = D.transpose(tuple([i] + [z for z in xrange(len(parents)+1) if not z == i])) #put x in the first index
        

        #print 'check1', time.time() - start

        inf = 0 
        for x in [0,1]:
            for y in itertools.product([0,1], repeat=len(parents)):
                pxy =  D[tuple([x] + list(y))] 
                px = D.sum(tuple(xrange(1,len(parents)+1)))[x]
                #print 'D', D
                #print 'x', x
                #print 'y', y
                #print 'D.sum(0)', D.sum(0)
                #print type(D)
                py = D.sum(0)[tuple(y)]
                if pxy > 0:
                    inf += pxy * np.log(pxy / (px*py))
        #        print 'check2', time.time() - start

        for i in xrange(len(parents)):
            px = D.sum(tuple(xrange(1,len(parents)+1)))[1]
            py = D.sum(tuple([z for z in xrange(0,len(parents)+1) if not z==i+1]))[1]
            pxy = D.sum(tuple([z for z in xrange(1,len(parents)+1) if not z==i+1]))[1,1]

            if pxy < px*py and attractive:
                return -1000
            
        return inf
        
    def score(x, parents, method='bic'):
        if method == 'bic':
            i = information(x, parents)
            if i is None:
                return None
            else:
                return N*i - (np.log(N) / 2) * (2**len(parents))

    vars = set()
    for k in moments.keys():
        if set(k).issubset(labels):
            vars |= set(k)

    candidate_parents = defaultdict(set)
    for K in moments.keys():
        print K, labels
        if set(K).issubset(labels):
            for k in K:
                for l in subsets(set(K)-set([k])):
                    if len(l) < order:
                        candidate_parents[k] |= set([l])

    print 'candidate parents, len', len(candidate_parents)
    print >>outfile, len(vars)

    for v in sorted(vars):
        print header[v], len(candidate_parents[v])
        sys.stdout.flush()
        print >>outfile, quote(header[v]), len(candidate_parents[v])
        for c in candidate_parents[v]:
            #print 'considering parents', [header[z] for z in c], c
            start = time.time()
            s = score(v, c)
            end = time.time()
            #print 'scoring takes', end-start
            if s is None:
                continue
            else:
                print >>outfile, s, len(c), " ".join([quote(header[z]) for z in c])
            end = time.time()
            #print 'printing takes', end-start
    outfile.close()

if __name__ == "__main__":
    networkdir = sys.argv[1]
    N = sys.argv[2]
    source = sys.argv[3]
    try:
        outfile = file(sys.argv[4], 'w')
    except:
        outfile = sys.stdout

    if 'third_order' in source:
        order = 3
    elif 'fourth_order' in source:
        order = 4
    elif 'tree' in source:
        order = 2
    elif 'indep' in source:
        order = 1
    else:
        order = 2

    print 'computing bic scores of order', order
    print 'loading moments from ', networkdir+'/pickles/estimated_moments.'+source+''+N+'.'+str(order)+'.pk'

    moments = pickle.load(file(networkdir+'/pickles/estimated_moments.'+source+''+N+'.'+str(order)+'.pk'))
    #moments = pickle.load(file(networkdir+'/true_moments.'+N+'.2.pk'))
    labels = pickle.load(file(networkdir+'/pickles/labels.pk'))
    print 'labels', labels
    N = int(N)
    header = pickle.load(file(networkdir+'/pickles/header.pk'))
    attractive = 'attractive' in networkdir
    main(moments, header, labels, outfile, order, N, attractive)
