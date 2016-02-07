import numpy as np
import h5py
from maximum_likelihood import learn_failures

def generate_x(y, failures):
  p = 1-np.exp(y * failures.T)
  r = np.random.rand(*p.shape)
  return np.matrix(r < p, dtype=int)

def recovery_test():
  n_parents = 3
  n_children = 10
  n_data = 5000
  failures = np.ones((n_children, n_parents+1))
  for i in xrange(n_children):
    failures[i, i%n_parents] = 0.3
    failures[i, n_parents] = 0.5

  failures = np.matrix(np.log(failures))

  print 'generating failures\n', np.exp(failures)
  priors = np.array([0.5]*(n_parents+1))
  priors[-1] = 1
  Y_matrix = np.matrix(np.random.rand(n_data , n_parents+1) < np.tile(priors, (n_data,1)), dtype=int)
  X_matrix = np.array(generate_x(Y_matrix, failures))
  Y_matrix = np.array(Y_matrix)
  X_matrix = np.array(X_matrix)

  Y = [np.nonzero(y)[0].tolist() for y in Y_matrix[:,:-1]]
  targets = ((X_matrix[:, j].tolist(), Y, n_parents) for j in xrange(n_children))
  learned_failures = np.array(map(learn_failures, targets))

  print 'learned failures\n', learned_failures

  print 'difference\n', np.abs(np.exp(failures) - learned_failures)

  if np.all(np.abs(np.exp(failures) - learned_failures) < 0.1):
    print 'PASS'
  else:
    print "FAIL"

  #outfile = h5py.File('data.h5', 'w')
  #outfile.create_dataset('Y', data=Y_matrix)
  #outfile.create_dataset('X', data=X_matrix)
  #outfile.close()

recovery_test()
