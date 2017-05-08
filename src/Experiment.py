import numpy as np
import scipy.sparse as ss
import sklearn.linear_model
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer


class Experiment(object):
    '''
    Organize the process of getting data, building a classifier,
    and exploring new representations. Experiment class pattern credit to:
    Matthew Stone, CS 533, Spring 2017 - Classifier Patterns
    '''

    def __init__(self,
                 data_manager,
                 classifier,
                 cleaning_operations,
                 filtering_operations,
                 vectorizer,
                 shuffle=False,
                 seed=None):
        '''set up the problem of learning a classifier from a data manager'''
        self.data_manager = data_manager
        self.classifier = classifier
        self.cleaning_operations = cleaning_operations
        self.filtering_operations = filtering_operations
        self.vectorizer = vectorizer
        self.shuffle = shuffle
        self.seed = seed
        self.initialized = False
        self.validated = False

    def initialize(self):
        '''materialize the training data, dev data and test data as matrices'''
        if self.initialized:
            return
        self.X_train_text, self.y_train_text = \
            self.data_manager.training_data()
        self.X_dev_text, self.y_dev_text = \
            self.data_manager.dev_data()
        self.X_test_text, self.y_test_text = \
            self.data_manager.test_data()
        self._filter()
        self._clean()
        self._extract_features()
        if self.shuffle:
            self._shuffle()
        self.initialized = True

    def _filter(self):
        '''
        The order of the list of operations matters!
        Filtering will remove data!
        '''
        for op in self.filtering_operations:
            if isinstance(op, tuple):
                f = op[0]
                args = op[1]
            else:
                f = op
                args = {}
            self.X_train_text, self.y_train_text = \
                f(self.X_train_text, self.y_train_text, **args)
            self.X_dev_text, self.y_dev_text = \
                f(self.X_dev_text, self.y_dev_text, **args)
            self.X_test_text, self.y_test_text = \
                f(self.X_test_text, self.y_test_text, **args)

    def _clean(self):
        '''
        The order of the list of operations matters! It is recommended to have
            'replace_newline_char' be the first in cleaning_operations.
            Cleaning will not remove data!
        '''
        for op in self.cleaning_operations:
            if isinstance(op, tuple):
                f = op[0]
                args = op[1]
            else:
                f = op
                args = {}
            f(self.X_train_text, self.y_train_text, **args)
            f(self.X_dev_text, self.y_dev_text, **args)
            f(self.X_test_text, self.y_test_text, **args)

    def _extract_features(self):
        '''
        Vectorizer should return NxM matrix where N is number of samples and
            M is number of features (dimension)
        '''
        try:
            self.X_train = self.vectorizer.transform(self.X_train_text)
            print('Using vectorizer fitted elsewhere')
        except NotFittedError:
            self.X_train = self.vectorizer.fit_transform(self.X_train_text)
        self.X_dev = self.vectorizer.transform(self.X_dev_text)
        self.X_test = self.vectorizer.transform(self.X_test_text)

        # TODO: Ensure no zero vectors or any funny business
        S = list(set(self.y_train_text + self.y_dev_text + self.y_test_text))
        self.lang_to_num = dict(zip(S, range(len(S))))
        self.num_to_lang = {v: k for k, v in self.lang_to_num.iteritems()}

        conv = lambda x: self.lang_to_num[x]
        self.y_train = map(conv, self.y_train_text)
        self.y_dev = map(conv, self.y_dev_text)
        self.y_test = map(conv, self.y_test_text)

    def _shuffle(self):
        if self.seed:
            np.random.seed(self.seed)
        p = np.random.permutation(len(self.y_train))
        self.X_train = self.X_train[p]
        self.X_train_text = [self.X_train_text[i] for i in p]
        self.y_train = [self.y_train[i] for i in p]
        self.y_train_text = [self.y_train_text[i] for i in p]

    def stack_features(self, vectorizers):
        if not self.initialized:
            self.initialize()
        for vec in vectorizers:
            s = 'Number of features {} stacking: {}'
            print(s.format('before', self.X_train.shape[1]))
            try:
                self.X_train = \
                    ss.hstack([self.X_train,
                               vec.transform(self.X_train_text)])
                print('Using vectorizer fitted elsewhere')
            except NotFittedError:
                self.X_train = \
                    ss.hstack([self.X_train,
                               vec.fit_transform(self.X_train_text)])
            self.X_dev = \
                ss.hstack([self.X_dev, vec.transform(self.X_dev_text)])
            self.X_test = \
                ss.hstack([self.X_test, vec.transform(self.X_test_text)])
            print(s.format('after', self.X_train.shape[1]))

    def set_classifier(self, classifier):
        if self.validated:
            self.validated = False
        self.classifier = classifier

    def fit_and_validate(self):
        '''train the classifier and assess predictions on dev data'''
        if not self.initialized:
            self.initialize()
        if (not hasattr(self.classifier, 'coef_') or
                self.classifier.coef_ is None):
            self.classifier.fit(self.X_train, self.y_train)
        else:
            print('Using classifier fitted elsewhere')

        self.dev_predictions = self.classifier.predict(self.X_dev)
        self.dev_accuracy = \
            sklearn.metrics.accuracy_score(self.y_dev, self.dev_predictions)
        self.validated = True

    def test_results(self):
        ''' Get results from testing data '''
        if not self.initialized:
            self.initialize()
            self.fit_and_validate()
        if not self.validated:
            self.fit_and_validate()
        self.test_predictions = self.classifier.predict(self.X_test)
        self.test_accuracy = \
            sklearn.metrics.accuracy_score(self.y_test, self.test_predictions)
