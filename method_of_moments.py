import numpy as np
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

def create_source(X, X_names, Y, Y_names, noise):
  networkdir = tempfile.mkdtemp(prefix='yoni-')
  write_samples(networkdir+'/samples/samples.dat', X,Y)
  write_pickle(networkdir+'/pickles/header.pk', Y_names+X_names)
  write_pickle(networkdir+'/pickles/labels.pk', range(len(Y)))
  write_pickle(networkdir+'/pickles/noise.pk', noise)
  return networkdir

class MOMLearner():
  def __init__(self, X, X_names, Y, Y_names, noise, args):
    self.source_dir = create_source(X,X_names, Y, Y_names, noise)
    self.N = X.shape[0]
    self.args = args

  def learn_failures(self):
    args = self.args
    N = self.N
    networkdir,anchor_source = initDirectory(self.source_dir, self.args, dry=False)
    execute(networkdir, 'reading', ['python', 'read_data.py', networkdir, str(N)]+args)
    execute(networkdir, 'transform', ['python', 'transform.py', networkdir, str(N)]+args)
    execute(networkdir, 'bic', ['python', 'compute_bic_score.py', networkdir, str(N), '.'.join(args)+'.'+anchor_source+'_anchors', networkdir+'/'+'.'.join(args)+'.'+anchor_source+'_anchors.'+str(N)+'.bic'])
    execute(networkdir, 'gobnilp', ['./gobnilp', networkdir+'.'.join(args)+'.'+anchor_source+'_anchors.'+str(N)+'.bic'])
    execute(networkdir, 'tree', ['python', 'create_model.py', networkdir])
    execute(networkdir, 'structure', ['python', 'cross_edges.py', networkdir, 'optkl', 'moments-tree', 'clip'])
