#Utilities used by all files
from __future__ import division
import os
import numpy as np 
from scipy.io import *
import copy
import cPickle as pickle 
#Create directories if absent
#Input: List of directories
#Output: None
def createIfAbsent(dirList):
    for d in dirList:
        if os.path.exists(d)==False:
            os.mkdir(d)

#Create a marginal vector
def createMarginalVector(List,Fields,Cardinality):
    marginal_vec = []
    for i in range(Fields.shape[0]):
        marginal_vec = marginal_vec + Fields[i,:Cardinality[i]].tolist()
    for ei in range(List.shape[0]):
        vi = List[ei,0].astype(int)
        vj = List[ei,1].astype(int)
        marginal_vec = marginal_vec + List[ei,2:2+(Cardinality[vi]*Cardinality[vj])].tolist()
    return np.array(marginal_vec)
#Process final results and return
#Input: results_mat (hashtable to be stored in matlab format)
#        exact_mat_out (matlab filename of file containing exact results)
#Output: modified results_mat
def processResults(results_mat,exact_mat_out):
    #Check if exact_out folder exists
    inputfile = open(exact_mat_out,'rb')
    data = loadmat(inputfile, mdict=None)
    inputfile.close()
    results_mat['exact_node_marginals'] = data['exact_node_marginals']
    results_mat['exact_log_partition']  = float(data['log_partition'][0])
    return results_mat

#Exception handling
class FWExcept(RuntimeError):
    def __init__(self, arg):
        self.args = arg

#Write a marginal vector to UAI format
#Assumes potentials in marginal_vec are strictly positive (i.e *NOT* in log format)
#Input : outputfilename (File name to write UAI file to)
#         marginal_vec (marginal vector to write UAI file with )
#         N (number of variables)
#         List (edges in graph and their corresponding edge potentials)
#Output: None, writes to file 
def writeToUAI(outputfilename,marginal_vec,N,Cardinality,List):
    f = open(outputfilename,'w',0)
    f.write('MARKOV\n')
    f.write(str(N)+'\n')
    vert_card = np.squeeze(Cardinality).astype(int).tolist()
    f.write(" ".join([str(k) for k in vert_card]))
    graph_str = ""
    EdgeList = List[:,:2].astype(int)

    num_lines = 0

    #Write Stage 1 of UAI File 
    for i in xrange(N):
        graph_str+= "1 "+str(i)+"\n"
        num_lines+=1
    
    for ei in xrange(EdgeList.shape[0]):
        graph_str+= "2 "+str(EdgeList[ei,0])+" "+str(EdgeList[ei,1])+"\n"
        num_lines+=1

    f.write("\n"+str(num_lines)+"\n")
    f.write(graph_str+"\n")
    
    #Write Stage 2 of UAI File 
    obj_str = ""
    idx = 0
    for i in xrange(N):
        m_vec = [str(m) for m in marginal_vec[idx:idx+vert_card[i]].tolist()]
        obj_str+= str(vert_card[i])+"\n"+" ".join(m_vec)+"\n\n"
        idx+=vert_card[i]
    for ei in xrange(EdgeList.shape[0]):
        edge_card = vert_card[EdgeList[ei,0]]*vert_card[EdgeList[ei,1]]
        m_vec = [str(m) for m in marginal_vec[idx:idx+edge_card].tolist()]
        obj_str+= str(edge_card)+"\n"+" ".join(m_vec)+"\n\n"
        idx+=edge_card

    f.write(obj_str)
    f.close()
    assert idx==len(marginal_vec)


#Write a marginal vector to UAI format 2
#Assumes potentials in marginal_vec are strictly positive (i.e *NOT* in log format)
#Input : outputfilename (File name to write UAI file to)
#        potentials
#Output: None, writes to file 
def writeToUAI2(outputfilename, potentials):

    f = open(outputfilename,'w',0)
    f.write('MARKOV\n')

    unary_potentials = filter(lambda pot: len(pot['pot'].shape) == 1, potentials)

    #count unary potentials (ie count variables)
    N = len(unary_potentials)
    f.write(str(N)+'\n')

    vert_card = [pot['pot'].shape[0] for pot in unary_potentials]
    f.write(" ".join([str(k) for k in vert_card]))

    graph_str = ""

    num_lines = len(potentials)
    f.write("\n"+str(num_lines)+"\n")

    for pot in potentials:
        f.write(str(len(pot['vars'])) +' ' +' '.join([str(v) for v in pot['vars']]) +"\n")
    
    #Write Stage 2 of UAI File 
    for pot in potentials:
        f.write('\n')
        size = pot['pot'].size
        f.write(str(size))
        f.write('\n')
        f.write(' '.join([str(np.exp(v)) for v in pot['pot'].reshape(size)]))

    f.close()

#load UAI data from matlab file 
#Input : filename
#Output: Data loaded from matfile 
def loadUAIfromMAT(inputfilename):
    inputfile = open(inputfilename,'rb')
    data = loadmat(inputfile, mdict=None)
    inputfile.close()
    card = np.squeeze(data['Cardinality']).astype(int)
    # N,Nodes,Edges,Cardinality,mode = loadUAIfromMAT(inputfilename)
    return np.squeeze(data['N'][0][0]),np.squeeze(data['Fields']),np.squeeze(data['List']),card,'UAI'

#Graph class : container for all graph based structures and devices
#IMPORTANT : All potentials stored in G are in *LOG* format
class Graph:
    def __init__(self, *args, **kwargs):
        if len(kwargs.keys())==0:
            assert False,"Specify all arguments as arg1=val1, arg2=val2"
        else:
            try:
                mode = copy.deepcopy(kwargs['mode'])
                weighted = copy.deepcopy(kwargs['weighted'])
            except KeyError:
                assert False,"Mode/weighted not defined"
            assert mode=='UAI' or mode=='minimal' or mode=='copy',("Unrecognized mode: "+mode)
            if mode=='UAI' or mode=='copy':
                assert 'N' in kwargs and 'Edges' in kwargs and 'Nodes' in kwargs and 'Cardinality' in kwargs,"Insufficient inputs for UAI mode"
                N   = copy.deepcopy(kwargs['N'])
                Edges=np.copy(kwargs['Edges'])
                Nodes=np.copy(kwargs['Nodes'])
                Cardinality=np.copy(kwargs['Cardinality'])
            if mode=='minimal': 
                assert 'N' in kwargs and 'Obj' in kwargs,"Insufficient inputs for minimal mode"
                Obj = np.copy(kwargs['Obj'])
                N   = copy.deepcopy(kwargs['N'])
        #By default use the UAI format 
        self.N = N
        #print N," vertices", Cardinality.shape, " Cardinality"
        #Handle each mode of graph creation separately
        if mode=='UAI':
            #UAI mode uses data read from UAI files
            self.Edges = Edges
            #Adjust for matlab indices
            self.Edges[:,0]=self.Edges[:,0]-1
            self.Edges[:,1]=self.Edges[:,1]-1
            assert np.all(self.Edges[:,:2].astype('int')>=0),'Negative edges detected. Investigate'
            self.Nodes = Nodes
            self.Cardinality=np.reshape(Cardinality,(N,)).astype(int)
            self.weighted = weighted
        elif mode=='copy': #Use if trying to create a copy of a graph
            #UAI mode uses data read from UAI files
            self.Edges = Edges
            self.Nodes = Nodes
            self.Cardinality=np.reshape(Cardinality,(N,)).astype(int)
            self.weighted = weighted
        elif mode=='minimal':
            #Minimal mode implicitly used for pairwise binary models
            self.Nodes = np.zeros((N,2))
            self.Nodes[:,1]=np.reshape(Obj[:N],(N,1))
            self.Cardinality = 2*np.ones((N,1))
            self.Cardinality =self.Cardinality.astype(int)
            self.Cardinality=np.reshape(Cardinality,(N,)).astype(int)
            self.Edges = np.zeros((N*(N-1),6))
            self.weighted = weighted
            pos=N;ctr = 0
            for i in xrange(N):
                for j in xrange(i+1,N):
                    self.Edges[ctr,0]=i
                    self.Edges[ctr,1]=j
                    self.Edges[ctr,-1]=Obj[pos]
                    pos += 1
                    ctr += 1
            self.Edges=self.Edges[:ctr,:]
        else:
            assert False,"Investigate. mode changed when it should not have"

        #Create structures necessary for graph
        #1. Adjacency
        #2. Weight Matrix
        #3. rhos, rhos_node
        #4. nVertices, nEdges
        #5. var/edge begin/end
        #6. centre of marginal polytope for current graph
        self.createGraphStruct()

    #append additional features to graph
    def createGraphStruct(self):

        self.nVertices = self.N
        self.nEdges    = self.Edges.shape[0]
        self.Adjacency = np.zeros((self.N,self.N))
        self.Weighted  = np.zeros((self.N,self.N))
        EdgeList = self.Edges[:,:2].astype(int)
        for i in xrange(self.nEdges):
            v1 = EdgeList[i,0]
            v2 = EdgeList[i,1]
            card_edge = (self.Cardinality[v1]*self.Cardinality[v2])
            assert type(v1) is np.int64 and type(v2) is np.int64 and type(card_edge) is np.int64,"Indices not integer. Investigate"
            if int(self.Cardinality[v1])==2 and int(self.Cardinality[v2])==2:
                self.Weighted[v1,v2] = self.Edges[i,2]+self.Edges[i,5]-self.Edges[i,3]-self.Edges[i,4]
            else:
                self.Weighted[v1,v2] = np.sum(self.Edges[i,2:2+card_edge])
            self.Adjacency[v1,v2] = 1
        self.Weighted = self.Weighted + np.transpose(self.Weighted)
        self.Adjacency = self.Adjacency + np.transpose(self.Adjacency)
        self.computeEdgeRhos()
        self.computeNodeRhos()

        #Computing the marginal vector
        pot = []
        init_vec = []
        self.nodeIdx = np.zeros((self.nVertices,2)).astype(int)
        self.edgeIdx = np.zeros((self.nEdges,2)).astype(int)
        idx = int(0)
        total_card = 0
        for vi in xrange(self.nVertices):
            #Extract only upto the cardinality of the vertex
            v_card = int(self.Cardinality[vi])
            pot += self.Nodes[vi,:v_card].tolist()
            self.nodeIdx[vi,0]=int(idx)
            self.nodeIdx[vi,1]=int(idx+v_card)
            init_vec += v_card*[float(1)/v_card]
            assert np.array_equal(np.array(pot[self.nodeIdx[vi,0]:self.nodeIdx[vi,1]]),self.Nodes[vi,:v_card]),"Node potential vector not set"
            idx = idx + v_card
            total_card += v_card

        for ei in xrange(self.nEdges):
            v1 = EdgeList[ei,0]
            v2 = EdgeList[ei,1]
            card_edge = (self.Cardinality[v1]*self.Cardinality[v2])
            assert type(v1) is np.int64 and type(v2) is np.int64 and type(card_edge) is np.int64,"Indices not integer. Investigate"
            pot += self.Edges[ei,2:2+card_edge].tolist()
            self.edgeIdx[ei,0]=idx
            self.edgeIdx[ei,1]=idx+card_edge
            init_vec += card_edge*[float(1)/card_edge]
            assert np.array_equal(np.array(pot[self.edgeIdx[ei,0]:self.edgeIdx[ei,1]]),self.Edges[ei,2:2+card_edge]),"Edge potential vector not set correctly"
            idx = idx + card_edge
            total_card += card_edge
        
        #assert that size of the marginal vector is as expected
        assert len(pot)==total_card, "Length of marginal vector smaller than expected"
        self.pot = np.array(pot)
        self.init_vec = np.array(init_vec)
        #return G
        

    #compute edge appearance probabilities using matrix tree theorem
    def computeEdgeRhos(self):
        rhos_edge = np.zeros((self.nEdges,1))
        #check if weighted
        if self.weighted:
            mat = np.abs(self.Weighted)
        else:
            mat = self.Adjacency

        L = np.diag(mat.sum(axis=1))-mat
        L1 = L-1
        #Invert Matrix
        try:
            Linv = np.linalg.inv(L1)
        except LinAlgError:
            assert False, "Matrix inverse not defined for graph laplacian"
        for ei in xrange(self.nEdges):
            v1 = self.Edges[ei,0].astype('int')
            v2 = self.Edges[ei,1].astype('int')
            rhos_edge[ei] = mat[v1,v2]*(Linv[v1,v1]+Linv[v2,v2]-2*Linv[v1,v2])
        self.rhos_edge= rhos_edge


    #update node probabilities
    def computeNodeRhos(self):
        rhos_node = np.zeros((self.nVertices,1))
        for ei in xrange(self.nEdges):
            rhos_node[self.Edges[ei,0].astype('int')] += self.rhos_edge[ei]
            rhos_node[self.Edges[ei,1].astype('int')] += self.rhos_edge[ei]
        self.rhos_node= rhos_node


