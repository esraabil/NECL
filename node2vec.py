from __future__ import print_function
import time
from gensim.models import Word2Vec
import walker
class Node2vec(object):

    def __init__(self, graph, path_length, num_paths, dim, p=1.0, q=1.0, dw=False,  **kwargs):
        kwargs["workers"] = kwargs.get("workers", 5)
        if dw:
            kwargs["hs"] = 0
            p = 1.0
            q = 1.0
        t1 = time.time()
        self.graph = graph
        if dw:
            self.walker = walker.BasicWalker(graph, workers=kwargs["workers"],walk_length=path_length)
            t2 = time.time()
            self.walker.preprocess_transition_probs()
            t3 = time.time()
            print("prob time: "+str(t3-t2))
            sentences = self.walker.simulate_walks(num_walks=num_paths, walk_length=path_length)
        else:
            self.walker = walker.Walker(
                graph, p=p, q=q, workers=kwargs["workers"])
            t2 = time.time()
            self.walker.preprocess_transition_probs()
            t3 = time.time()
            print("prob time: "+str(t3-t2))
            sentences = self.walker.simulate_walks(num_walks=num_paths, walk_length=path_length)
        t2 = time.time()
        print("walking time: "+str(t2-t1))
        kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = kwargs.get("size", dim)
        kwargs["sg"] = 1

        self.size = kwargs["size"]
#        print("Learning representation...")
        word2vec = Word2Vec(**kwargs)
        self.vectors = {}
        for word in graph.G.nodes():
            self.vectors[word] = word2vec.wv[word]
        del word2vec

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
