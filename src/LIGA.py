'''
Algorithm:

Tromp, E., and M. Pechenizkiy.
"Graph-Based N-gram Language Identification on Short Texts."
Proceedings of the 20th Machine Learning conference of
    Belgium and The Netherlands (2011): 27-34.

Code adaptation:
Python implementation of Java code found at: https://github.com/ErikTromp/LIGA/
'''
from __future__ import division
import numpy as np


class LIGA(object):

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.initialized = False

    def _get_ngrams(self, X, n, analyzer, tokenizer):
        ngrams = []
        for t in X:
            if analyzer == 'word':
                t = tokenizer(t) if tokenizer else t.split()
            ngrams.append([t[i:i+n] for i in xrange(len(t)-n+1)])
        return ngrams

    def _add_vertices(self, N):
        vertices = {}
        for idx, ngrams in enumerate(N):
            for ngram in ngrams:
                l = self.y_train[idx]
                if ngram not in vertices:
                    vertices[ngram] = {}
                if l not in vertices[ngram]:
                    vertices[ngram][l] = 0
                vertices[ngram][l] += 1
                self.counter[l]['vertices'] += 1
        return vertices

    def _add_edges(self, N):
        edges = {k: {} for k in self.vertices}
        for idx, ngrams in enumerate(N):
            l = self.y_train[idx]
            for i in xrange(len(ngrams) - 1):
                src = ngrams[i]
                tar = ngrams[i + 1]
                if tar not in edges[src]:
                    edges[src][tar] = {}
                if l not in edges[src][tar]:
                    edges[src][tar][l] = 0
                edges[src][tar][l] += 1
                self.counter[l]['edges'] += 1
        return edges

    def _recursive_path_matching(self, path, scores, curr_depth, max_depth):
        if curr_depth > max_depth or len(path) == 0:
            return scores
        if len(path) == 1:
            ngram = path[0]
            if ngram in self.vertices:
                for l, weight in self.vertices[ngram].iteritems():
                    if l not in scores:
                        scores[l] = 0.0
                    # scores[l] += (weight / self.counter[l]['edges'])
                    scores[l] += (weight / self.counter[l]['vertices'])
            return scores
        src = path[0]
        tar = path[1]
        if src in self.vertices:
            for l, weight in self.vertices[src].iteritems():
                if l not in scores:
                    scores[l] = 0.0
                # scores[l] += (weight / self.counter[l]['edges'])
                scores[l] += (weight / self.counter[l]['vertices'])
            if src in self.edges:
                if tar in self.edges[src]:
                    for l, weight in self.edges[src][tar].iteritems():
                        if l not in scores:
                            scores[l] = 0.0
                        # scores[l] += (weight / self.counter[l]['vertices'])
                        scores[l] += (weight / self.counter[l]['edges'])
        path = path[1:]
        return self._recursive_path_matching(path,
                                             scores,
                                             curr_depth + 1,
                                             max_depth)

    def _path_matching(self, ngrams):
        scores = {}
        depth = 0
        max_depth = 1000
        return self._recursive_path_matching(ngrams, scores, depth, max_depth)

    def _prediction(self, scores, top_pred):
        if top_pred > len(scores):
            top_pred = len(scores)
        langs = []
        vals = []
        for k, v in scores.iteritems():
            langs.append(k)
            vals.append(v)
        vals = np.array(vals)
        indices = np.argsort(vals)[::-1]
        # top_scores = {}
        top_scores = []
        for i in xrange(top_pred):
            idx = indices[i]
            # top_scores[langs[idx]] = vals[idx]
            top_scores.append((langs[idx], vals[idx]))
        return top_scores

    def initialize(self):
        try:
            decode_unicode(self.X_train, self.y_train)
        except UnicodeEncodeError:
            # already decoded
            pass
        self.counter = {}
        for l in self.y_train:
            if l not in self.counter:
                self.counter[l] = {'vertices': 0, 'edges': 0}
        self.initialized = True

    def learn_model(self, n=2, analyzer='char', tokenizer=None):
        if not self.initialized:
            self.initialize()
        self.n = n
        self.analyzer = analyzer
        self.tokenizer = tokenizer
        N = self._get_ngrams(self.X_train, n, analyzer, tokenizer)
        self.vertices = self._add_vertices(N)
        self.edges = self._add_edges(N)

    def classify(self, X_test, top_preds=5):
        N = self._get_ngrams(X_test, self.n, self.analyzer, self.tokenizer)
        pred = []
        for ngrams in N:
            scores = self._path_matching(ngrams)
            if scores:
                top_scores = self._prediction(scores, top_preds)
            else:
                top_scores = [('UNKNOWN', -1.0)]
            pred.append(top_scores)
        return pred
