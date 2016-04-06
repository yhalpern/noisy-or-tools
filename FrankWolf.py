from __future__ import division
from ExponetiatedGradient import expGrad
import string
import itertools
import sys
import numpy as np
import scipy.optimize as sciopt
from gurobipy import *
import time
import os
from Toulbar import MAPsolver

def randstring(n):
    return ''.join([np.random.choice(list(string.letters)) for _ in xrange(n)])
    #return 'test'

def TOULBAR(_f, obj, max_time=60):
    N = _f(args = 'N')
    order = _f(args = 'order')

    marg_vec = [-float(z[0]) for z in obj.tolist()]
    M = max(marg_vec)
    marg_vec = [z-M for z in marg_vec]

    idx = 0
    potentials = []
    for k in xrange(1,order+1):
        for vars in itertools.combinations(xrange(N), k):
            pot = {}
            size = 2**k
            shape = (2,)*k
            pot['vars'] = vars
            pot['pot'] = np.array(marg_vec[idx:idx+size]).reshape(shape)
            idx += size
            potentials.append(pot)
            

    res,status = MAPsolver.runMAP(potentials, uai_fname='temp.'+randstring(16)+'.uai', timer=max_time)


    return np.array(res).reshape((res.size,1)),status
    
def initModel(nVars, constraints):
    #set up the model and constraints
    model = Model()
    var = []
    for i in xrange(nVars):
        var.append(model.addVar())

    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.Method, 1) #dual simplex
    model.update()
    for c in constraints:
        model.addConstr(quicksum([coeff*var[i] for coeff,i in c[0]]), c[1], c[2])
    model.update()

    return model, var

def fullyCorrectiveStep(f, vertices, alpha0, a, max_steps=200):
    #minimize f(Ax)
    #x in the simplex
    alpha0_orig = alpha0.copy()
    if alpha0.shape[0] < vertices.shape[0]:
        alpha0 = np.vstack([(1-a)*alpha0, a*np.ones((1,1))])
    print 'b'
    #print 'vertices', vertices
    #print 'alpha0', alpha0

    def _f(a, withGrad=False):

        if withGrad:
            val, grad = f(np.dot(vertices.T, a), withGrad=withGrad)
            return val, np.dot(vertices, grad).T
        else:
            return f(np.dot(vertices.T, a), withGrad=withGrad)
        
    if max_steps > 1:
        print 'c'
        alpha,f,steps,gap = expGrad(_f, alpha0, 1e-7, verbose=True, max_steps=max_steps)
    else:
        alpha = alpha0
        f = _f(alpha)
        steps = -1
        gap=-1


    print 'exp grad:', 
    print 'gap', gap,
    print 'steps', steps
    print 'alpha0', alpha0.T
    print 'alpha', alpha.T

    x = np.dot(vertices.T, alpha)
    #scale = np.linalg.norm(x, 2)

    return 1.0, f, x, alpha, gap < 1e-6


    
    

def LP(obj, var, model, max_time=float('inf'), rerun=False):
    scale = 1
    #print [float(z[0]) for z in obj.tolist()]
    s = time.time()
    o = LinExpr([float(z[0]) for z in obj.tolist()], var)
    if not rerun:
        model.reset()

    model.setObjective(o, GRB.MINIMIZE)
    print 'setting objective takes', time.time() - s, 'seconds'
    while 1:
        model.setParam(GRB.Param.TimeLimit, max_time)
        
        model.optimize()
        try:
            x = np.array([[z.getAttr("x")] for z in var])
            break
        except:
            if model.status == 9:
                max_time *= 2
            else:
                print 'model returned status', model.status
                return None, model.status

    print "status", model.status
    try:
        os.unlink('gurobi.log')
    except:
        pass
    return x, model.status

def getStep(_f, s, x):
    print 'x', x.shape, 's-x', (s-x).shape
    z = sciopt.fminbound(lambda a: _f(x+a*(s-x)), 0, 0.999, xtol=1e-5)

    while _f(x) < _f(x+z*(s-x)) and z > 0:
        z /= 2

    print 'z', z
    print 'improvement!', _f(x), _f(x+z*(s-x)) 
    return z, _f(x+z*(s-x))

def FW(_f, constraints, x0=None, tol=10**(-6), integer=False, max_steps=500, infeasible_start=False, max_time=float('inf'), n0=0, verbose=False, alternate_directions = set(), fully_corrective=False, logger=None):

    
    #initalize x
    x = x0
    a = np.ones((1,1))
    if not integer:
        model,var = initModel(_f(), constraints)

    #initialize lower bound
    l = _f(args='lower')
    report = []
    functional_iterates = []

    vertices = x0.T
    vertex_set = set([tuple(x0.T.tolist()[0])])

    f,g = _f(x, withGrad=True)
    alpha = 0
    problem_setup = {'A':_f(args='A'), 'b':_f(args='B'), 'x0':x}
    report.append({'x':np.array(x), 
                  'gap':f-l, 
                  'f':f, 
                  'l':l, 
                  'stepsize':alpha, 
                  #'smallest':small,
                  'vertex':x0.T,
                  'problem':problem_setup})

    time_limit = 2
    for n in xrange(n0, max_steps):
        print 'n',n
        
        problem_setup = None

        start = time.time()
        #get gradient
        f,g = _f(x, withGrad=True)
        print 'gradient time', time.time()-start
        sys.stdout.flush()

        rerun = False
        #minimize s'g s.t. constraints
        while 1:
            if integer:
                s, status = TOULBAR(_f, obj=g, max_time=time_limit*60)
            else:
                s, status = LP(obj=g, var=var, model=model, max_time=time_limit, rerun=rerun)

            #check for better directions
            for v in alternate_directions:
                if np.dot(g.T,v) < np.dot(g.T,s):
                    s = v

            new_vertex=False
            if not tuple(s.T.tolist()[0]) in vertex_set:
                new_vertex=True
                vertex_set.add(tuple(s.T.tolist()[0]))
                vertices = np.vstack([vertices, s.T])

            rerun = False
            print 'LP time', time.time()-start
            #search over stepsizes
            old_f = f

            vertex = s
            alpha,f = getStep(_f, s, x)
            print 'taking step', alpha

            if fully_corrective:
                print 'fully corrective'
                if new_vertex and not n % 100 == 10:
                    max_steps = 1 
                else:
                    print 'fully corrective step'
                    if not new_vertex:
                        print 'ashape', a.shape
                        indicator = np.array([1 if np.linalg.norm(s.T-vertices[i, :])<1e-9 else 0 for i in xrange(vertices.shape[0])])
                        indicator = indicator.reshape(a.shape)
                        assert indicator.sum() == 1
                        a = (1-alpha)*a + alpha*indicator
                        print 'ashape', a.shape
                    max_steps = 500
                print 'here'
                alpha,f,s,a,certificate = fullyCorrectiveStep(_f, vertices, a, alpha, max_steps=max_steps)
                sys.stdout.flush()

            print 'stepsize time', time.time()-start
            if old_f <= f:
                print 'non-decreasing step!', old_f, '<', f
                print 'throwing out vertex set and restarting'
                vertices = x0.T
                vertex_set = set([tuple(x0.T.tolist()[0])])
                a = np.ones((1,1))
                continue

            else:
                print 'made progress?', old_f, f

            if f < old_f or status == 2:
                break

            elif status == 9:
                time_limit *= 2
                rerun = True
            else:
                print "not sure what's going on!"
                return x, f, f-l, report
                
        #update duality gap
        if status == 2:
            l = max(l, f+np.dot((vertex-x).T, g))
            print 'lower', l

        #do update
        x = x + alpha*(s-x)
        print 'f', f
        print 'min x', x.min()
        #print 'smallest', _f(x, args='smallest')

        #if n % 10 == 0:
        #    print 'save x', x.tolist()

        if logger:
            logger(n,x)

        functional_iterates.append(f)
        print 'gap', f-l,'tol', tol 
        report.append({'x':np.array(x), 
                      'gap':f-l, 
                      'f':f, 
                      'l':l, 
                      'stepsize':alpha, 
                      #'smallest':small,
                      'vertex':vertex,
                      'problem':problem_setup})

        assert f - l >= 0

        if f-l < tol:
            break

        if alpha == 0:
            print 'warning: breaking due to 0 stepsize!'
            break
    
    return x, f, f-l, report

if __name__ == "__main__":

    #minimize f(x) = x^T x
    #subject to each element greater than 1
    def f(x=None):
        if x is None:
            return 10*np.ones((4,1))
        else:
            print float(np.dot(x.T,x))
            return float(np.dot(x.T,x)), 2*x

    const = []

    for i in xrange(4):
        const.append((np.array([-1*int(j==i) for j in xrange(4)]), -1.0))
    print const
    print FW(f, const)
