import numpy as np


class Experiment(object):
    '''
    Organize the process of getting data, building a classifier,
    and exploring new representations. Modified from and credit to:
    Matthew Stone, CS 533, Spring 2017 - Classifier Patterns
    '''

    def __init__(self,
                 data_manager,
                 classifier,
                 cleaning_operations,
                 vectorizer,
                 shuffle=False,
                 seed=None):
        # TODO: Change name/defaults of some of these parameters
        '''set up the problem of learning a classifier from a data manager'''
        self.data_manager = data_manager
        self.classifier = classifier
        self.cleaning_operations = cleaning_operations
        self.vectorizer = vectorizer
        self.shuffle = shuffle
        self.seed = seed
        self.initialized = False

    def initialize(self):
        'materialize the training data, dev data and test data as matrices'
        if self.initialized:
            return
        self.X_train_text, self.y_train_text = \
            self.data_manager.training_data()
        self.X_dev_text, self.y_dev_text = \
            self.data_manager.dev_data()
        self.X_test_text, self.y_test_text = \
            self.data_manager.test_data()
        self._clean()
        self._extract_features()
        if self.shuffle:
            self._shuffle()
        self.initialized = True

    def _clean(self):
        for op in self.cleaning_operations:
            self.X_train_text, self.y_train_text = \
                op(self.X_train_text, self.y_train_text)
            self.X_dev_text, self.y_dev_text = \
                op(self.X_dev_text, self.y_dev_text)
            self.X_test_text, self.y_test_text = \
                op(self.X_test_text, self.y_test_text)

    def _extract_features(self):
        '''
        Vectorizer should return NxM matrix where N is number of samples and
            M is number of features (dimension)
        '''
        self.X_train = vectorizer(self.X_train_text)
        self.X_dev = vectorizer(self.X_dev_text)
        self.X_test = vectorizer(self.X_test_text)

        # TODO: Ensure no zero vectors or any funny business
        # TODO: convert labels to integers and maintain a mapping
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
        self.X_train = self.X_train[p, :]
        self.X_train_text = [self.X_train_text[i] for i in p]
        self.y_train = self.y_train[p]
        self.y_train_text = [self.y_train_text[i] for i in p]

    '''
    def fit_and_validate(self):
        'train the classifier and assess predictions on dev data'
        if not self.initialized:
            self.initialize()
        self.classifier.fit(self.X_train, self.y_train)
        self.dev_predictions = self.classifier.predict(self.X_dev)
        self.accuracy = \
            sklearn.metrics.accuracy_score(self.y_dev, self.dev_predictions)

    @classmethod
    def transform(cls, expt, operation, classifier):
        'use operations to transform the data and set up new expt'
        if not expt.initialized:
            expt.initialize()
        result = cls(expt.data, classifier)
        result.X_train, result.y_train = operation(expt.X_train, expt.y_train)
        result.X_dev, result.y_dev = operation(expt.X_dev, expt.y_dev)
        result.X_test, result.y_test = operation(expt.X_test, expt.y_test)
        result.initialized = True
        return result
    '''
