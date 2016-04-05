import os
import sys
import subprocess

class Bean:
    def __init__(self):
        pass

def extractOrder(prog_args):
    if 'third_order' in prog_args:
        order = 3
    elif 'fourth_order' in prog_args:
        order = 4
    elif 'indep' in prog_args:
        order = 1
    else:
        order = 2
    return order

def makedirs(name):
    try:
        os.makedirs(name)
    except:
        pass


def execute(networkdir, name, callargs, strict=True):
    logfile = file(networkdir+'/logging/'+name+'.log', 'w')
    errfile = file(networkdir+'/logging/'+name+'.err', 'w')
    print '-'*20
    print 'executing', name
    print 'args:', ' '.join(callargs)
    print 'log:', networkdir+'/logging/'+name+'.log'
    print 'err:', networkdir+'/logging/'+name+'.err'
    retval = subprocess.call(callargs, stdout=logfile, stderr=errfile)
    logfile.close()
    errfile.close()
    if retval:
        print networkdir, name, ' '.join(callargs), 'exited with exit code', retval
        print 'log:', networkdir+'/logging/'+name+'.log'
        print 'err:', networkdir+'/logging/'+name+'.err'
        print file(networkdir+'/logging/'+name+'.err').read()
        assert not strict

    else:
        print '+'*20
        return 1

def initDirectory(networkdir, args, dry=False):
    N = int(args[-2].strip('N'))
    sample_size = N
    S = int(args[-1].strip('S'))

    order = 2
    if 'local_anchors' in args:
        anchor_source = 'local'
    else:
        anchor_source = 'simple'

    print 'arguments', args
    networkdir = networkdir.strip('/')

    max_tags = args[0]

    if not dry:
        subprocess.call('rm -rf '+ networkdir+'-'+'-'.join(args), shell=1)
        print 'copying over data'
        makedirs(networkdir+'-'+'-'.join(args)+'/true_params')
        makedirs(networkdir+'-'+'-'.join(args)+'/samples')
        makedirs(networkdir+'-'+'-'.join(args)+'/logging')
        makedirs(networkdir+'-'+'-'.join(args)+'/results')
        makedirs(networkdir+'-'+'-'.join(args)+'/reports')
        makedirs(networkdir+'-'+'-'.join(args)+'/pickles')
        makedirs(networkdir+'-'+'-'.join(args)+'/timing')
        makedirs(networkdir+'-'+'-'.join(args)+'/maximum_likelihood')

        subprocess.call(['cp', '-r', networkdir+'/pickles', networkdir+'-'+'-'.join(args)])
        subprocess.call(['cp', '-r', networkdir+'/maximum_likelihood', networkdir+'-'+'-'.join(args)])
        subprocess.call(['cp', '-l', '--verbose', networkdir+'/data', networkdir+'-'+'-'.join(args)+'/data'])
        subprocess.call(['cp', '-l', '-r',  networkdir+'/candidates', networkdir+'-'+'-'.join(args)+'/candidates'])
        subprocess.call(['cp', '-r', networkdir+'/concepts', networkdir+'-'+'-'.join(args)+'/concepts'])
        subprocess.call(['cp', '-r', networkdir+'/calibrations', networkdir+'-'+'-'.join(args)+'/calibrations'])
        if 'impute_10' in args:
            print 'imputed data'
            subprocess.call(['cp', '-l', '--force', '--verbose', networkdir+'/samples/samples.impute.10.dat', networkdir+'-'+'-'.join(args)+'/samples/'+str(N)+'_samples.dat'])

        elif 'impute_1' in args:
            print 'imputed data'
            subprocess.call(['cp', '-l', '--force', '--verbose', networkdir+'/samples/samples.impute.1.dat', networkdir+'-'+'-'.join(args)+'/samples/'+str(N)+'_samples.dat'])
            subprocess.call(['cp', '-l', '--force', '--verbose', networkdir+'/samples/samples.dat', networkdir+'-'+'-'.join(args)+'/samples/'+str(N)+'_samples.test'])

        elif S > 0:
            print 'using regular data'
            subprocess.call(['cp', '-l', '--force', '--verbose', networkdir+'/samples/samples.dat.'+str(S), networkdir+'-'+'-'.join(args)+'/samples/'+str(N)+'_samples.dat'])
            subprocess.call(['cp', '-l', '--force', '--verbose', networkdir+'/samples/samples.dat.'+str(S), networkdir+'-'+'-'.join(args)+'/samples/'+str(N)+'_samples.test'])
        else:
            subprocess.call(['cp', '-l', '--force', '--verbose', networkdir+'/samples/samples.dat', networkdir+'-'+'-'.join(args)+'/samples/'+str(N)+'_samples.dat'])
            subprocess.call(['cp', '-l', '--force', '--verbose', networkdir+'/samples/samples.dat', networkdir+'-'+'-'.join(args)+'/samples/'+str(N)+'_samples.test'])


    networkdir = networkdir + '-'+'-'.join(args)+'/'
    return networkdir, anchor_source



