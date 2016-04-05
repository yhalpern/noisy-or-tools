import sys
from utils import extractOrder
from itertools import combinations
from helpers import *
import cPickle as pickle
def extract_candidates(networkdir, label_name):
    #returns a list of tuples: (score_a, score_b, candidate, conflicts)

    try:
        network_src = networkdir.split('-')[0]
        infile = file(network_src+'/candidates/'+'candidates.'+label_name)
    except:
        return []

    retval = []
    while 1:
        l = infile.readline()
        if l == '':
            break
        candidate = l.strip()
        l = infile.readline()
        score_a, score_b = l.split()
        score_a = float(score_a)
        score_b = float(score_b)
        conflicts = []
        while 1:
            l = infile.readline()
            if '---' in l:
                break
            conflicts.append(l.strip())
        retval.append((score_a, score_b, candidate, conflicts))
    infile.close()
    
    return retval

def init_anchors(networkdir, labels, header, default_anchors):
    learned_anchors = {}
    for i in labels:
        print 'initializing', header[i]
        candidates_i = extract_candidates(networkdir, header[i])
        if len(candidates_i):
            candidates_i.sort(key=lambda p: p[0], reverse=True)
            learned_anchors[i,]  =  (header.index(candidates_i[0][2]),)
        else:
            learned_anchors[i,] = (default_anchors[labels.index(i)],)

    for i,j in itertools.combinations(labels, 2):
        candidates_i = extract_candidates(networkdir, header[i])
        candidates_j = extract_candidates(networkdir, header[j])
        valid_pairing = False
        for a,b in sorted(itertools.product(candidates_i, candidates_j), key=lambda p: p[0][0] + p[1][1], reverse=True):
            print 'attempting to pair', a[2], b[2]
            if (not (a[2] in b[3])) and (not (b[2] in a[3])):
                learned_anchors[i,j] = (header.index(a[2]),header.index(b[2]))
                valid_pairing = True
                break
            else:
                print 'conflict!'

        if not valid_pairing:
            print 'warning! could not find a valid pairing for', header[i], header[j]
            learned_anchors[i,j] = (default_anchors[labels.index(i)], default_anchors[labels.index(j)])

    return learned_anchors


if __name__ == "__main__":
    networkdir = sys.argv[1]
    sample_size = int(sys.argv[2])
    prog_args = sys.argv[3:]
    print ' '.join(sys.argv)
    print 'prog args', prog_args
    sys.stdout.flush()
    order = extractOrder(prog_args)


    infile = file(networkdir+'/'+'data')
    header = infile.readline().strip().split(' ')
    inv_header = dict(zip(header, xrange(len(header))))
    try:
        labels= pickle.load(file(networkdir+'/pickles/labels.pk'))
    except:
        labels = [i for i in xrange(1,len(header)) if not 'body_' in header[i] and not 'header_' in header[i]]

    print 'labels', [header[i] for i in labels]

    counter = parallel_execute_query([(l,) for l in labels], networkdir,sample_size, len(header), 10)

    labels.sort(key=lambda l: counter[(l,)][1], reverse=True)
    print 'labels', [header[i] for i in labels]
    #labels = filter(lambda l: counter[(l,)][1] > 50, labels)
    MAX_LABELS = int(networkdir.split('-')[1])
    labels = labels[:MAX_LABELS]
    labels.sort()

    try:
        anchors =[header.index('header_'+header[t]) for t in labels]
    except:
        anchors =[header.index('header_'+header[t].replace('label_', 'anchor_')) for t in labels]

    if 'learned_anchors' in networkdir:
        learned_anchors = init_anchors(networkdir, labels, header, anchors)
    else:
        learned_anchors = {}

    counter = parallel_execute_query([(a,) for a in anchors]+learned_anchors.values(), networkdir, sample_size, len(header), 10)


    labels = filter(lambda t: counter[header.index('header_'+header[t].replace('label_','anchor_')),][1], labels)

    anchors =[header.index('header_'+header[t].replace('label_','anchor_')) for t in labels]

    if 'anchor_baseline' in prog_args:
        labels = anchors


    print 'labels', labels, [header[l] for l in labels], len(labels)
    print 'anchors', anchors, [header[a] for a in anchors], len(anchors)
    print 'len header', len(header)


    L = []
    for K in xrange(1,order+1):
        L += list(combinations(labels, K)) + list(combinations(anchors, K))


    L += learned_anchors.values()
    L += list(zip(labels, anchors))
    for l,a in learned_anchors.items():
        L += list(zip(l,a))

    counter = parallel_execute_query(L, networkdir, sample_size, len(header), 10)
    for c,val in counter.items():
        counter[c] = val + 1.0

    pickle.dump(dict(counter), file(networkdir+'/counter.'+str(sample_size)+'.'+str(order)+'.pk', 'w'))

    pickle.dump(header, file(networkdir+'/pickles/header.pk', 'w'))
    pickle.dump(labels, file(networkdir+'/pickles/labels.pk', 'w'))
    pickle.dump(anchors, file(networkdir+'/pickles/anchor_list.pk', 'w'))
