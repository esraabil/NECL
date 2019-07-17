from __future__ import print_function
import numpy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score,accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from time import time

from sklearn import svm
class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)
#multilabel
    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)
#single label
    def train2(self,X,Y):
        self.clf = svm.LinearSVC()#gamma=0.001,decision_function_shape='ovo')
        X_train = [self.embeddings[x] for x in X]
        self.clf.fit(X_train,Y)
    #multi label
    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        results = {}
        results["macro"] = f1_score(Y, Y_, average="macro")
        results["micro"] = f1_score(Y, Y_, average="micro")
        return results
        # print('-------------------')
    #single label
    def evaluate2(self, X, Y):
        Y2 = self.predict2(X)
        averages = ["micro", "macro"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y2, average=average)
        return results
      #multi label
    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y
    #single label
    def predict2(self, X):
        X2 = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X2)
        return Y
      #multi label
    def split_train_evaluate(self, X, Y, train_precent,shuffle_indices):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)
  #single label
    def split_train_evaluate2(self, X, Y, train_precent,shuffle_indices):
        state = numpy.random.get_state()
        training_size = int(train_precent * len(X))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]
        self.train2(X_train, Y_train)
        numpy.random.set_state(state)
        return self.evaluate2(X_test, Y_test)


def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size+1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num
    return vectors


def read_node_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y

def read_node_labelsl(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(int(vec[1]))
    fin.close()
    return X, Y