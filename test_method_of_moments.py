import numpy as np
from method_of_moments import MOMLearner

def generate_x(y, failures):
  p = 1-np.exp(y * failures.T)
  r = np.random.rand(*p.shape)
  return np.matrix(r < p, dtype=int)

def recovery_no_noise_test():
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
  X_names = ['child_'+str(i) for i in xrange(n_children)]
  Y_names = ['parent_'+str(i) for i in xrange(n_parents)]
  noise = {}
  args = [str(n_parents)]+' opt kl simplex indep constreg simple'.split() +  ['G1000', 'N'+str(n_parents), 'S0']
  learner = MOMLearner(X_matrix, X_names, Y_matrix, Y_names, noise, args)
  learner.learn_failures()

recovery_no_noise_test()
