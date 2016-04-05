from __future__ import division
from ExponetiatedGradient import expGrad
import random
from models import TreeModel
import sys
import numpy as np
import cPickle as pickle
import itertools
from copy import deepcopy
from multiprocessing import Pool
from FrankWolf import FW
from helpers import *
import scipy.optimize as opt 
from helpers import extract_edges, create_CPD, get_counts


def bootstrap_sample(seed, r):
  if seed == 0:
    return r

  np.random.seed(seed)
  key,val = r
  val = np.array(val, dtype=float)
  print 'in', key, val
  shape = val.shape
  N = val.sum()
  val /= float(N)
  res = np.random.multinomial(N, val.reshape(val.size))
  #res /= float(N)
  res = res.reshape(shape)
  print 'out', key, res
  return key, res



def row_normalize(M):
    M = deepcopy(M)
    for i in xrange(M.shape[0]):
        M[i,:] /= M[i,:].sum()
    return M

def nonoise():
    M = np.zeros((2,2))
    M[0,0] = 1
    M[1,1] = 1
    return M

print 'args', ' '.join(sys.argv)

bootstrap = 1
for b in xrange(bootstrap):
  networkdir = sys.argv[1].strip('/')
  print 'networkdir', networkdir
  structure_file = networkdir+'/results/adjacency.mat'
  method = sys.argv[2]
  marginal_source = '.'.join(networkdir.split('-')[1:])
  complexity = 2
  if 'third_order' in networkdir:
      complexity = 3
  if 'indep' in networkdir:
      complexity = 1

  N = networkdir.split('-')[-2].strip('N')
  Nsamples = int(N)

  edges = extract_edges(structure_file)

  header = pickle.load(file(networkdir+'/pickles/header.pk'))
  inv_header = dict(zip(header, xrange(len(header))))
  labels = pickle.load(file(networkdir+'/pickles/labels.pk'))
  tags = [header[t] for t in labels]
  anchors = {}
  noise = {}
  anchor_src = 'simple'
  if 'ground_truth' in sys.argv or 'ground_truth' in networkdir:
      for a in labels:
          anchors[a] = a
          noise[a] = nonoise()

  else:
      anchors = pickle.load(file(networkdir+'/pickles/anchors.pk'))
      noise = pickle.load(file(networkdir+'/pickles/noise.pk'))

  moment_src = networkdir+'/pickles/estimated_moments.'+marginal_source+'.'+anchor_src+'_anchors'+str(Nsamples)+'.'+str(complexity)+'.pk'
  print 'reading marginals from', moment_src
  marginals = pickle.load(file(moment_src))

  def get_anchors(L):
      if 'ground_graph' in networkdir:
        if all([l in anchors for l in L]): #special case, all latent variables, get the true correlation from the grahp
          return list(L)
        else:
          return [anchors[l] if l in anchors else l for l in L]
      else:
        return [anchors[l] if l in anchors else l for l in L]


  if 'skeleton' in sys.argv:
      words = []
  else:
      words = [w for w in header if not w in tags and not inv_header[w] in anchors.values()]

#words = ['header_trigger\colon', 'header_pain']

  def transform_to_latent((key, intent, noise_list, D)):
      if 'ground_truth' in sys.argv:
          print 'short circuit'
          return tuple(key), D / float(D.sum()), 0

      #if 'ground_graph' in sys.argv and set(key)==set(intent):
      #    return tuple(key), D / float(D.sum()), 0

      new_k = []
      original_shape = D.shape
      latents = []
      observed = []

      #print 'transforming', intent, 'to', key


      for var,obs in zip(intent,key):
          if var == obs:
              new_k.append(obs)
              observed.append(key.index(obs))
          else:
              new_k.append(var)
              latents.append(var)

      if len(latents) == 0 or 'ground_truth' in sys.argv:
          return tuple(new_k), D / float(D.sum()), 0

      A, A_inv = calculate_adjustment(noise_list)

      order = len(key)

      counter = marginals[tuple(sorted(latents))].transpose(index_sort(latents))
      counter = counter / counter.sum()

      x_indep = np.zeros(original_shape)
      obs_index = new_k.index(key[observed[0]])
      assert obs_index == len(new_k)-1, "observed index is always last?"

      if len(original_shape) == 2:
          x_indep[:,0] = counter / 2.0
          x_indep[:,1] = counter / 2.0
      elif len(original_shape) == 3:
          x_indep[:,:,0] = counter / 2.0
          x_indep[:,:,1] = counter / 2.0
      else:
          print 'moment larger than expected!'
          assert 0
      x_indep = x_indep.reshape((x_indep.size, 1))


      D = np.matrix(D.reshape((D.size, 1)))
      D = D / D.sum()
      A = np.matrix(A)
      gamma= 0.1

      def _f(X, withGrad=False):
          val = KL(D, A*X) + gamma*KL(x_indep, X)
          if withGrad:
              g = gradKL(X, D, A).T + gamma*gradKL(X, x_indep, np.matrix(np.identity(x_indep.size))).T
              return val, g
          return val

      #print [header[i] for i in intent]
      #print 'A', A
      #print 'D', np.array(D).reshape((2,)*len(intent))

      res,val,steps,gap = expGrad(_f, x_indep.copy(), 1e-5, verbose=False, lower=0)
      #print 'X', np.array(res).reshape(original_shape)

      return tuple(new_k), np.array(res).reshape(original_shape), val

  CPDs = {}
  for j in labels:
      parents = [z for z in labels if (labels.index(z),labels.index(j)) in edges]
      t,s = tags[labels.index(j)], tuple([tags[labels.index(i)] for i in parents])
      key = tuple([j] + parents)
      first_index = sorted(key).index(j)
      other_indices = [sorted(key).index(p) for p in parents]
      transpose_order = tuple([first_index]+other_indices)
      m = marginals[tuple(sorted(key))].transpose(transpose_order)
      smoothing = 0#10**(-6)
      CPDs[t,s] = create_CPD(m, smoothing=smoothing, verbose=False)


  tree = TreeModel(CPDs, #dictionary of CPDs
                  tags, #variable ids
                  format="CPDs" #latent structure already holds cpds
                  )

  anchor_failures = {}
  anchor_noise = {}
  anchor_dict = {}

  for l,a in anchors.items():
      anchor_dict[header[l]] = header[a]
      print 'a is', a
      #noise[a] holds P(anchor | latent)
      anchor_noise[header[a]] = 1-noise[a][1,0]
      print 'anchor',header[a], 'has noise', anchor_noise[header[a]]
      anchor_failures[header[l],header[a]] = noise[a][0,1] / anchor_noise[header[a]]
      print 'anchor',header[a], 'has failure', anchor_failures[header[l], header[a]]

  tree.addAnchors(anchor_dict, anchor_failures, anchor_noise)
  tree.addObservations(words #variable ids
                       )

#print 'tree observations are', tree.observations
  print 'latents', tree.latents
  print 'labels', [header[l] for l in labels]

  if 'maximum_likelihood' in networkdir:
      for O in tree.observations:
          i = inv_header[O]
          if 'no_sparsity' in networkdir:
              failures = file(networkdir+'/maximum_likelihood/'+str(i)+'.fail.nosparse').readline().split()
          else:
              failures = file(networkdir+'/maximum_likelihood/'+str(i)+'.fail').readline().split()

          for i,f in enumerate(failures):
            print f
            try:
              l,val = f.split(':')
              val = float(val)
              if l == 'noise':
                print 'noise', O
                tree.noise[O] = val
              else:
                tree.failures[l,O] = val
            except:
              f = float(f)
              if i == len(failures)-1:
                tree.noise[O] = f
              else:
                L = header[labels[i]]
                tree.failures[L,O] = f

          #for j,L in enumerate(tree.latents):
          #    if float(failures[j]) < 0.99:
          #        print L, O, failures[j]
          #    tree.failures[L,O] = float(failures[j])
          #tree.noise[O] = float(failures[-1])
          


  elif 'moments-tree' in sys.argv:
      queries = []
      intent = []
      for w in words:
          queries.append((inv_header[w],))
          intent.append((inv_header[w],))

      for l in labels:
          print 'listed', len(queries), 'queries'
          for w in words:

              intent.append([l,inv_header[w]])

              queries.append(get_anchors([l,inv_header[w]]))
              q = [l]

              for p in tree.parents_of[header[l]]:
                  p = inv_header[p]
                  q.append(p)

              q.append(inv_header[w])
              print 'query', [header[z] for z in queries[-1]]
              assert len(q) <= 4

              intent.append(q)
              queries.append(get_anchors(q))
              print 'query', [header[z] for z in queries[-1]]
      if len(queries): 
          res = parallel_execute_query(queries, networkdir, Nsamples, len(header), 10)
      else:
          res = {}

      res = dict([bootstrap_sample(b, r) for r in res.items()])

      noise_queries = []
      for Q,I in zip(queries, intent):
          for q,i in zip(Q,I):
              noise_queries.append((i,q))

      noise_res = parallel_execute_query(noise_queries, networkdir, Nsamples, len(header), 10)

      noise_list = []

      for idx in xrange(len(queries)):
          temp = []
          for q,i in zip(queries[idx], intent[idx]):
              if i == q:
                  temp.append(nonoise())
              else:
                  temp.append(row_normalize(noise_res[i,q]+10**(-6)).T)

          noise_list.append(temp)
              
      t = 0

      pool = Pool(16)

      residuals = {}
      #for arg in ((queries[i], intent[i], noise_list[i], res[tuple(queries[i])]) for i in xrange(len(queries))):
      #    k,val,residual = transform_to_latent(arg)
      for k,val,residual in pool.imap(transform_to_latent, ((queries[i], intent[i], noise_list[i], res[tuple(queries[i])]) for i in xrange(len(queries)))):
          t += 1
          if t % 100 == 0:
              print 'transformed', t, '/', len(res), 'items'
          sys.stdout.flush()
          print k, residual
          try:
              #if residual > 0.1:
              #  print 'warning: transforming marginal to uniform', k
              #  val = np.ones((2,)*len(k), dtype=float)
              #  val = val/float(val.sum())

              marginals[k] = val
              residuals[k] = residual
          except:
              print 'error!', k
              assert 0

      counter = get_counts(marginals, header)
      tree.addCounts(counter)
      tree.addResiduals(residuals)

      minf = 0
      maxf = 1.0
      min_noise = 1.0
      tree.estimateCrossEdges('moments-tree', minf, maxf, min_noise=min_noise, ignore_correction='nocorrection' in networkdir)


#re-assert anchor property
  anchors = pickle.load(file(networkdir+'/pickles/anchors.pk'))
  for l in tree.latents:
      a = header[anchors[header.index(l)]]
      for k in tree.latents:
          if not l == k:
              tree.failures[k,a] = 1.0

  tree.describe(0.85)
  fname = networkdir+'/results/'+marginal_source
  fname += '.tree_model'
  if "upper-bound" in sys.argv:
      fname += '.upperbound'
  if 'optkl' in sys.argv:
      fname += '.optkl'
  if 'optl2' in sys.argv:
      fname += '.optl2'
  if 'inv' in sys.argv:
      fname += '.inv'
  if not 'skeleton' in sys.argv:
      fname += '.full'
  if "moments-tree" in sys.argv:
      fname += '.moments_tree'
  if "ground_truth" in sys.argv:
      fname += '.ground_truth'
  if 'clip' in sys.argv:
      fname += '.clip'

  fname += '.'+str(b)
  print 'dumping to', fname+'.pk'
  pickle.dump(tree, file(fname+'.pk', 'w'))


  outfile = file(fname+'.txt', 'w')

  for l in tree.latents:
      for o in sorted(tree.observations, key=lambda o: tree.failures[l,o]):
          if tree.failures[l,o] >= 1:
              break
          else:
              print >>outfile,  l, o, tree.failures[l,o]

  for o in sorted(tree.observations, key=lambda o: tree.noise[o]):
      print >>outfile, 'noise', o, tree.noise[o]

  outfile.close()


