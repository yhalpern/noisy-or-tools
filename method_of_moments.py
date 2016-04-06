import numpy as np
import glob
import subprocess
from utils import initDirectory, execute
import cPickle as pickle
import os
import tempfile


def make_file(filename):
  if not os.path.exists(os.path.dirname(filename)):
      try:
          os.makedirs(os.path.dirname(filename))
      except OSError as exc: # Guard against race condition
          if exc.errno != errno.EEXIST:
              raise

def write_samples(dest, X, Y):
  make_file(dest)
  outfile = file(dest, 'w')
  data = np.hstack([Y, X])
  N,M = data.shape
  for i in xrange(N):
    print >>outfile, i, 'compact', ' '.join([str(j) for j in xrange(M) if data[i,j] > 0])
  outfile.close()

def write_pickle(dest, obj):
  make_file(dest)
  pickle.dump(obj, file(dest, 'w'))

def write_text(dest, txt):
  outfile = file(dest, 'w')
  outfile.write(txt)
  outfile.close()

def create_source(X, X_names, Y, Y_names, noise):
  networkdir = tempfile.mkdtemp(dir='tmp/')
  write_samples(networkdir+'/samples/samples.dat', X,Y)
  write_pickle(networkdir+'/pickles/header.pk', Y_names+X_names)
  write_pickle(networkdir+'/pickles/labels.pk', range(len(Y_names)))
  write_pickle(networkdir+'/pickles/anchor_dict.pk', dict([(i,i) for i,_ in enumerate(Y_names)]))
  write_pickle(networkdir+'/pickles/noise.pk', noise)
  os.makedirs(networkdir+'/maximum_likelihood')
  os.makedirs(networkdir+'/candidates')
  os.makedirs(networkdir+'/calibrations')
  write_text(networkdir+'/data', ' '.join(Y_names+X_names))
  return networkdir

class MOMLearner():
  def __init__(self, X, X_names, Y, Y_names, noise, args):
    self.O = len(X_names)
    self.L = len(Y_names)
    self.X_names = X_names
    self.Y_names = Y_names

    self.source_dir = create_source(X,X_names, Y, Y_names, noise)
    self.N = X.shape[0]
    self.args = args

    
  def read_failures(self, networkdir):
    result_file = glob.glob(networkdir+'/results/*.txt')[0]
    results = file(result_file).readlines()
    failures = np.ones((self.L+1, self.O))
    for l in results:
      src, dest, weight = l.strip().split()
      weight = float(weight)
      if src == 'noise':
        i = self.L
      elif src in self.Y_names:
        i = self.Y_names.index(src)
      else:
        continue

      if dest in self.X_names:
        j = self.X_names.index(dest)
      else:
        continue

      failures[i,j] = weight
    self.failures = failures

  def learn_failures(self):
    args = self.args
    N = self.N
    networkdir,anchor_source = initDirectory(self.source_dir, self.args, dry=False)
    execute(networkdir, 'reading', ['python', 'read_data.py', networkdir, str(N)]+args)
    execute(networkdir, 'transform', ['python', 'transform.py', networkdir, str(N)]+args)
    execute(networkdir, 'bic', ['python', 'compute_bic_score.py', networkdir, str(N), '.'.join(args)+'.'+anchor_source+'_anchors', networkdir+'/'+'.'.join(args)+'.'+anchor_source+'_anchors.'+str(N)+'.bic'])
    execute(networkdir, 'gobnilp', ['./gobnilp', networkdir+'.'.join(args)+'.'+anchor_source+'_anchors.'+str(N)+'.bic'])
    subprocess.call(' '.join(['dot', '-Tpdf', '.'.join(args)+'.'+anchor_source+'_anchors.'+str(N)+'.dot', '>', '.'.join(args)+'.'+anchor_source+'_anchors.'+str(N)+'.pdf']), shell=1)
    subprocess.call(' '.join(['mv', '.'.join(args)+'.'+anchor_source+'_anchors.'+str(N)+'.*', networkdir]), shell=1)
    execute(networkdir, 'coloring', ['python', 'colored_graph.py', networkdir])
    execute(networkdir, 'tree', ['python', 'create_model.py', networkdir])
    execute(networkdir, 'structure', ['python', 'cross_edges.py', networkdir, 'optkl', 'moments-tree', 'clip'])
    self.read_failures(networkdir)
    return self.failures.T
