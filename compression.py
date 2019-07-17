# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 08:46:13 2019

@author: eakbas
"""
from multiprocessing import Pool
import time
# compute similariy scores for possible pairs that we can merge, 
#thr2 is min similarity score if similarity of pair is lees than this, no need to compute, check based on number of neighbours
class pC(object):

    def __init__(self, G,trs):
        self.G=G
        self.thr=trs
        self.thr2=trs/2
        self.score2=set()
    def ScoreCompute(self):
    #look 2 nodes which are not neighbor but share commen friend
        G=self.G
        nl=list(G.nodes)
        score=dict()
        t2 = time.time()
        for n in nl:
            self.createList(n)
        print("number of pairs "+str(len(self.score2)))
        t1 = time.time()   
        print(t1-t2)
          #compute similarity score for pair n 
        for n in self.score2:
            fs=self.getscore(n)
            score[n]=fs;
        t2 = time.time()
        print(t2-t1)
        return score
        #compute similarity score for pair n 
    def ScoreComputeparalel(self):
    #look 2 nodes which are not neighbor but share commen friend
        G=self.G
        nl=list(G.nodes)
        score=dict()
        t2 = time.time()
        with Pool(processes=4) as pool:         # start 4 worker processes
            pool.map(self.createList, nl  )
        print("number of pairs "+str(len(self.score2)))
        t1 = time.time()   
        print(t1-t2)
           #compute similarity score for pair n 
        with Pool(processes=3) as pool:         # start 4 worker processes
            scoreNode = pool.map(self.getscore, self.score2  )
        pool.close()
        pool.join()
        t2 = time.time()
        print(t2-t1)
        score = dict(zip(self.score2, scoreNode))    
        return score
    def getscore(self,n):
        (n1,n2)=n
        GG=self.G
        sco=0
        for w in GG[n1]:
            if w in GG[n2]:
                sco=sco+1
        if n1 in GG[n2]:
            fs=(2*(sco+1))/(len(GG[n1])+len(GG[n2]))#if there is an edge between nodes of pair, count it as similar neighbor
        else:
            fs=(2*sco)/(len(GG[n1])+len(GG[n2]))
        return fs;
    #create possible pairs to combine which are neighbors' neighbors for node n
    def createList(self,n):
        thr2=self.thr2
        G=self.G
        nel=G[n]
        for ne in nel:
            for nne in G[ne]:
                if n!=nne:    
                    if (n,nne) not in self.score2 and (nne,n) not in self.score2:
                                mn=len(nel)
                                mx=len(G[nne])
                                if mn>mx:
                                    mm=mn
                                    mn=mx
                                    mx=mm
                                if mn/(mn+mx)>=thr2:
                                    self.score2.add((n,nne))
                                        
def makeCompression(G,score,ths):
    maping=dict()# keep a reference from old node id to new node id
    for n in G.nodes():
        maping[n]=n     
    for e in score:
        s=e[0]
        t=e[1]
        if score[e]> ths:
            #find node id if it was merged before woth another node
            while 1:
                sid=maping[s]
                if s==sid:
                    break
                else:
                    s=sid
            while 1:
                tid=maping[t]
                if t==tid:
                    break
                else:
                    t=tid
            
            if sid!=tid:
                #delete node with less edges to make deletion faster
                if G.degree(sid)>G.degree(tid):
                    rid=tid
                    nrid=sid
                else:
                    rid=sid
                    nrid=tid
                #remode node rid and combine it wiht nrid so keep it in the mapping
                maping[rid]=nrid
                ngr=G[rid]
                G.remove_node(rid)
                #add all edges of rid to nrid
                # give weights
                for n in ngr:
                    if n!=nrid:
                        if G.has_edge(n,nrid):
                            G[n][nrid]['weight']=G[n][nrid]['weight']+ngr[n]['weight']
                            G[nrid][n]['weight']=G[n][nrid]['weight']+ngr[n]['weight']
                        else:
                            G.add_edge(n,nrid)
                            G.add_edge(nrid,n)
                            G[n][nrid]['weight'] = ngr[n]['weight']
                            G[nrid][n]['weight'] = ngr[n]['weight']
#    print("After_compresing,n,m " +str(len(G.nodes()))+" "+str(len(G.edges())))
    return G,maping