import sys
import cPickle as pickle
import os
import subprocess


def positive_correlation(m):
    return m[0,0]*m[1,1] > m[0,1]*m[1,0]

networkdir = sys.argv[1]
dotfile = [file(networkdir+'/'+f) for f in os.listdir(networkdir) if '0.dot' in f][0]
outfile = dotfile.name.replace('.dot', '.colored.dot')
header = pickle.load(file(networkdir+'/pickles/header.pk'))
momentfile = [file(networkdir+'/pickles/'+f) for f in os.listdir(networkdir+'/pickles') if 'estimated_moments' in f][0]

moments = pickle.load(momentfile)
out = file(outfile, 'w')

for l in dotfile:
    if '->' in l:
        try:
            a, _, b = l.strip().split()
        except:
            print "odd", l
            print >>out, l.strip()
        a = header.index(a.strip('"'))
        b = header.index(b.strip(';"'))

        try:
            m = moments[a,b]
        except:
            m = moments[b,a]

        if positive_correlation(m):
            color='darkgreen'
        else:
            color='red'

        print >>out, l.strip().replace(';', ' [color='+color+'];')

    else:
        print >>out, l.strip()

out.close()
subprocess.call('dot -Tpdf '+out.name+' > '+out.name.replace('.dot', '.pdf'), shell=1)
