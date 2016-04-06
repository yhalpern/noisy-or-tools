import numpy as np
from multiprocessing import Pool
import networkx as nx
import itertools
from helpers import parallel_execute_query, create_CPD, index_sort
import models
import cPickle as pickle
import sys

def eval_likelihood(self, Y, debug=False):

    return models.TreeModel.eval_likelihood(self, Y, do_check=False)
    if hasattr(self, 'NONPARAM'):
        Y = tuple([i for i in xrange(len(Y)) if Y[i] == 1])
        return np.log(self.Y_counts[Y])

    lprob = 0
    for i,l in enumerate(self.latents):
        parent_states = []
        parents = self.parents_of[l]
        for p in parents:
            parent_index = self.latent_lookup[p]
            parent_states.append(Y[parent_index])
        key = tuple([Y[i]]+parent_states)
        #if debug:
        #    print l, self.parents_of[l], key, self.lCPD[l][key]
        lprob += self.lCPD[l][key]
    assert lprob == models.TreeModel.eval_likelihood(self, Y, do_check=False)

    return lprob

networkdir = sys.argv[1].strip('/')
print 'networkdir', networkdir
N = int(networkdir.split('-')[-2].strip('N'))

args = networkdir.split('-')[1:]
if 'local_anchors' in networkdir:
    anchor_source = 'local'
else:
    anchor_source = 'simple'

if 'indep' in networkdir:
    complexity = 1
elif 'third_order' in networkdir:
    complexity = 3
else:
    complexity = 2

header = pickle.load(file(networkdir+'/pickles/header.pk'))
inv_header = dict(zip(header, xrange(len(header))))
latents = pickle.load(file(networkdir+'/pickles/labels.pk'))
labels = latents
latents = [header[l] for l in latents]
L = len(labels)

adj = np.loadtxt(networkdir+'/'+'.'.join(args)+'.'+anchor_source+'_anchors.'+str(N)+'.bn_mat')
print adj.sum(0)
best_score = 0
root = [i for i,v in enumerate(adj.sum(0)) if v==0]
print 'root is', root

parents_of = {}
children_of = {}
queries = []

for j in xrange(L):
    parents_of[latents[j]] = [latents[i] for i in np.nonzero(adj[:,j])[0]]
    children_of[latents[j]] = [latents[i] for i in np.nonzero(adj[j,:])[0]]
    queries.append(tuple([inv_header[latents[j]]] + [inv_header[i] for i in parents_of[latents[j]]]))


moments = pickle.load(file(networkdir+'/pickles/estimated_moments.'+'.'.join(args)+'.'+anchor_source+'_anchors'+str(N)+'.'+str(complexity)+'.pk'))
counter = {}

for q in queries:
    counter[q] = moments[tuple(sorted(q))].transpose(index_sort(q))

CPDs = {}
lCPDs = {}
for k,val in counter.items():
    CPDs[header[k[0]], tuple([header[z] for z in k[1:]])] = create_CPD(np.array(val + 1e-6, dtype=float))

model = models.TreeModel(CPDs, latents, format='CPDs')

for k in sorted(model.lCPD):
    print k, model.lCPD[k]

model_ll = 0
data = file(networkdir+'/samples/'+str(N)+'_samples.dat').readlines()

def eval(dat):
    if len(dat) == 0:
      return 0
    dat = dat.split()[2:]
    Y = [int(str(z) in dat) for z in labels]
    e =  eval_likelihood(model, Y, debug=True)
    return e
  

pool = Pool(48)
for i,e in enumerate(pool.imap(eval, data[:N])):
    if i % 10000 == 0:
      print i
    model_ll += e

#for d,dat in enumerate(data[:N]):
#    if d % 10000 == 0:
#      print d
#    if len(dat) == 0:
#      continue
#    try:
#      assert dat.split()[1] == 'compact'
#    except:
#      print 'could not read', dat
#      continue
#    dat = dat.split()[2:]
#    Y = [int(str(z) in dat) for z in labels]
#    e =  eval_likelihood(model, Y, debug=True)

print 'train', model_ll / N

model_ll = 0
for dat in data[-5000:]:
    assert dat.split()[1] == 'compact'
    dat = set([int(z) for z in dat.strip().split()[2:]])
    Y = [int(z in dat) for z in labels]
    e =  eval_likelihood(model, Y, debug=True)
    model_ll += e

print 'heldout', model_ll / 5000.0



print 'saving adj matrix'
np.savetxt(networkdir+'/results/adjacency.mat', adj)
pickle.dump(model, file(networkdir+'/results/skeleton.pk', 'w'))


moments = parallel_execute_query([tuple(sorted(q)) for q in queries], networkdir, N, len(header), 10)

for k,val in moments.items():
    moments[k] = val / float(val.sum())

counter = {}
for q in queries:
    counter[q] = moments[tuple(sorted(q))].transpose(index_sort(q))

CPDs = {}
lCPDs = {}
for k,val in counter.items():
    CPDs[header[k[0]], tuple([header[z] for z in k[1:]])] = create_CPD(np.array(val + 1e-6, dtype=float))

model = models.TreeModel(CPDs, latents, format='CPDs')

model_ll = 0
for dat in data[-5000:]:
    assert dat.split()[1] == 'compact'
    dat = set([int(z) for z in dat.strip().split()[2:]])
    Y = [int(z in dat) for z in labels]
    e =  eval_likelihood(model, Y, debug=True)
    model_ll += e

print 'optimistic heldout', model_ll / 5000.0
