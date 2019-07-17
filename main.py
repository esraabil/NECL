# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:27:31 2019

@author: eakbas
"""

import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from graph import *
import node2vec
from compression import *
from classify import Classifier, read_node_label
from multiprocessing import Pool
import time

def parse_args():
    
#default file for cora dataset
    inpt='data/cora/cora_edgelist.txt'
    outpt='data/cora/cora_Repws4.npy'
    label='data/cora/cora_labels.txt'

    wk=40
#    weighted=True
    weighted=False
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', default=inpt
                        ,help='Input graph file')
    parser.add_argument('--output', default=outpt,
                        help='Output representation file')
    parser.add_argument('--number-walks', default=40, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')
    parser.add_argument('--walk-length', default=10, type=int,
                        help='Length of the random walk started at each node')
    parser.add_argument('--workers', default=wk, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--representation-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--window-size', default=5, type=int,
                        help='Window size of skipgram model.')
    parser.add_argument('--p', default=1, type=float)
    parser.add_argument('--q', default=1, type=float)
    parser.add_argument('--label-file', default=label,
                        help='The file of node label')    
    parser.add_argument('--graph-format', default='edgelist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--weighted', default=weighted,
                        help='Treat graph as weighted')
    parser.add_argument('--clf-ratio', default=0.1, type=float,
                        help='The ratio of training data in the classification')


    
    args = parser.parse_args()
    return args

def main(args):
    print("number-walks "+str(args.number_walks))
    print("representation-size "+str(args.representation_size))
    print("walk-length "+str(args.walk_length))
    print("inout_fle "+str(args.input))
    print("******")
    g = Graph()
    deepw=False
    #similarity thresholds for compration
    trsl=[0.45,0.495, 0.5, 0.55, 0.6,0.7,0.8,1]
#    trsl=[ 0.5 ]
    learn=True
    X, Y = read_node_label(args.label_file)
    seed=0
    clsratio=[0.01, 0.05,0.07, 0.1,0.25,0.5,0.7,0.8]
#    clsratio=[ 0.1,0.2,0.4, 0.6,0.7,0.8,0.9]#,0.7,0.8]# use for blogcatalog

    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(len(X)))
    f=open(args.input+"shu.txt","w")
    f.writelines( str(item)+"\n" for item in shuffle_indices )
    f.close()    
    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input,directed=args.directed)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input, weighted=args.weighted,
                        directed=args.directed)
    G=g.G 
    print("before spar, n: " +str(len(G.nodes()))+" m: "+str(len(G.edges())))
   #compute similarity score for compression
    t1 = time.time()
    p=pC(G,0.45)
    scoreNode=p.ScoreCompute()
    t3 = time.time()
    f=open(args.input+"score.txt","w")
    f.writelines( str(n[0])+" "+str(n[1])+" "+str(scoreNode[n])+"\n" for n in scoreNode )
    f.close()
    print("total scorecom time: "+str(t3-t1))
#   read similarity scores from file
#    f=open(args.input+"score.txt","r")
#    scoreNode=dict()
#    for x in f:
#        l=x.split()
#        scoreNode[((l[0]),(l[1]))] = float(l[2])
   
    for kk in range(0,len(trsl)):
        if learn:   # do embeding
            ths=trsl[kk]#args.trs
            print("threshold is ...",ths)
            if args.graph_format == 'adjlist':
                g.read_adjlist(filename=args.input,directed=args.directed)
            elif args.graph_format == 'edgelist':
                g.read_edgelist(filename=args.input, weighted=args.weighted,
                                directed=args.directed)
            if ths!=1:#compression
                t1 = time.time()
                G=g.G
                G,nl2=makeCompression(G,scoreNode,ths)
                f=open(args.input+"af_spar.txt","w")
                f.writelines( str(n)+" "+str(nl2[n])+"\n" for n in nl2 )
                f.close()
                writeg(G,args)
                t2 = time.time()
                print("total_sparc_time: "+str(t2-t1))
            #embedding
            t1 = time.time()
            print("After_compresing,n,m " +str(len(g.G.nodes()))+" "+str(len(g.G.edges())))
            model = node2vec.Node2vec(graph=g, path_length=args.walk_length,
                                          num_paths=args.number_walks, dim=args.representation_size,
                                          workers=args.workers, p=args.p, q=args.q, window=args.window_size,  dw=deepw) 
            t2 = time.time()
            print("total_embeding_time "+str(t2-t1))
            vectors = model.vectors
            if ths!=1:#add embedding of removed nodes in compression
                    addBack(nl2,vectors)
            np.save(args.output+"_"+str(ths)+".npy",vectors)
        else:#load embeddings 
            vectors=np.load(args.output+"_"+str(ths)+".npy")
            vectors=vectors.item(0)
            print("file_loaded")
    
    #print("Training classifier") 
    #split_train_evaluate2 for single label (cora and wiki)
    #split_train_evaluate for multi lable (dblp and blogcatalog)
        for r in clsratio:
            clfa = Classifier(vectors, clf=LogisticRegression(solver='liblinear'))
            res=clfa.split_train_evaluate2(X, Y,r,shuffle_indices) # args.clf_ratio)
            print(str(r)+" "+str(res["macro"])+" "+str(res["micro"]))
def writeg(G,args):
     f=open(args.input+"comrg.txt","w")
     for n in G.nodes():
         for ne in G[n]:
             f.write( str(n)+" "+str(ne)+" "+str(G[n][ne]['weight'])+"\n")
     f.close()            
def addBack(nl,vectors):
    for n in nl:
        if n!=nl[n]:
            s=n;
            while 1:
                sid=nl[s]
                if s==sid:
                    break
                else:
                    s=sid
            if n in vectors:
                print("error")
            vectors[n]=vectors[s]
if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main(parse_args())