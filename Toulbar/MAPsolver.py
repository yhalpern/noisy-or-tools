#Source code for MAP solver using toulbar2
import numpy as np
import os,sys,subprocess,util

#Run MAP inference given a marginal vector
#IMPORTANT: Assumes potentials in marg_vec are in *LOG* format

def runMAP(potentials,uai_fname = './temp.uai',timer=-1):
    TOULBAR_BIN = '/home/halpern/binaries/toulbar2/bin/toulbar2-mod'
    if not os.path.exists(TOULBAR_BIN):
        assert False,"toulbar2 binary not found at :"+TOULBAR_BIN

    #write evidence file 
    f = open(uai_fname+'.evid','w')
    f.write('0')
    f.close()
    
    #write file 
    #Convert marg_vec to potentials
    util.writeToUAI2(uai_fname,potentials)

    soln_fname = uai_fname.split('.uai')[0]+'.sol'

    #Run MAP inference
    cmd = TOULBAR_BIN + ' '+uai_fname+' -w='+soln_fname +' > /dev/null'
    print cmd
    if timer>0:
        cmd += ' -timer='+str(timer)
    
    run_dir = uai_fname.rsplit('/',1)[0]
    run_dir += '/'
    result = subprocess.check_output(cmd,stderr=subprocess.STDOUT, shell=True)
    if "Time limit expired..." in result:
        status = 9
    else:
        status = 2

    print 'result', result
    #Get solution 
    MAPsoln = np.loadtxt(soln_fname)
    #assert MAPsoln.size == len([p for p in potentials if len(p['variables']) == 1])
    
    os.unlink(soln_fname)
    if os.path.exists(soln_fname):
        assert False,"sol file not deleted"
    os.unlink(uai_fname+'.evid')
    os.unlink(uai_fname+'.MPE')
    os.unlink(uai_fname)

    vertex = []
    
    for pot in potentials:
        shape = pot['pot'].shape
        temp = np.zeros(shape)
        vars = pot['vars']
        index = []
        for v in vars:
            index.append(MAPsoln[v])
        temp[tuple(index)] = 1
        vertex.append(temp.reshape((temp.size,1)))

    return np.vstack(vertex), status


