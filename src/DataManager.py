from collections import Counter
import numpy as np
import Constants


class DataError(Exception):
    def __init__(self, message):
        self.message = message


class DataManager(object):

    def __init__(self,
                 annotated_tsv=Constants.RECALL_TSV,
                 retrieved_data=Constants.RECALL_DATA,
                 lang_codes_file=Constants.LANG_CODES):
        self.initialized = False
        self.annotated_tsv = annotated_tsv
        self.retrieved_data = retrieved_data
        self.lang_codes_file = lang_codes_file

    def initialize(self, splits=(0.60, 0.20, 0.20), shuffle=True):
        if self.initialized:
            return
        self.lang_codes = self._load_lang_codes()
        self.tweet_labels, self.canonical_inv_idx = \
            self._load_tweet_labels()
        self.tweet_text, self.hydrated_inv_idx = self._load_tweet_text()
        self._split_dataset(splits, shuffle)
        self.initialized = True

    def _split_dataset(self, splits, shuffle):
        if type(splits) not in [list, tuple] or len(splits) != 3:
            err_msg = 'splits should be collection (list or tuple) of length 3'
            raise DataError(err_msg)
        if sum(splits) != 1.0:
            raise DataError('split percentages should sum to 1.0')

        # percentage_split() code source: http://bit.ly/2buzsDm
        def percentage_split(seq, percentages):
            if shuffle:
                np.random.shuffle(seq)
            cdf = np.cumsum(percentages)
            stops = map(int, cdf * len(seq))
            return [seq[a:b] for a, b in zip([0] + stops, stops)]

        self.X_train, self.y_train = [], []
        self.X_dev, self.y_dev = [], []
        self.X_test, self.y_test = [], []
        for lang in self.hydrated_inv_idx:
            if lang not in self.lang_codes:
                continue
            tweet_ids = list(self.hydrated_inv_idx[lang])
            train, dev, test = percentage_split(tweet_ids, splits)
            self.X_train += [self.tweet_text[t] for t in train]
            self.y_train += [lang] * len(train)
            self.X_dev += [self.tweet_text[t] for t in dev]
            self.y_dev += [lang] * len(dev)
            self.X_test += [self.tweet_text[t] for t in test]
            self.y_test += [lang] * len(test)

    def _load_lang_codes(self):
        lang_codes = set()
        with open(self.lang_codes_file) as fp:
            for line in fp:
                lang_codes.add(line.strip())
        return lang_codes

    def _update_inverted_index(self, inv_idx, lang, tweet_id):
        if lang in inv_idx:
            inv_idx[lang].add(tweet_id)
        else:
            inv_idx[lang] = set([tweet_id])
        return inv_idx

    def _load_tweet_labels(self):
        D = {}
        inv_idx = {}
        with open(self.annotated_tsv) as tsv:
            for line in tsv:
                t_lang, t_id = line.strip().split('\t')
                D[t_id] = t_lang
                self._update_inverted_index(inv_idx, t_lang, t_id)
        return D, inv_idx

    def _load_tweet_text(self):
        D = {}
        inv_idx = {}
        with open(self.retrieved_data) as fp:
            for line in fp:
                stripped = line.strip()[2:-2]  # remove quotes and brackets
                t_id, t_text = stripped.split('\",\"')
                D[t_id] = t_text
                self._update_inverted_index(inv_idx,
                                            self.tweet_labels[t_id],
                                            t_id)
        return D, inv_idx

    def training_data(self):
        if not self.initialized:
            raise DataError('Must call initialize() first')
        return self.X_train[:], self.y_train[:]

    def dev_data(self):
        if not self.initialized:
            raise DataError('Must call initialize() first')
        return self.X_dev[:], self.y_dev[:]

    def test_data(self):
        if not self.initialized:
            raise DataError('Must call initialize() first')
        return self.X_test[:], self.y_test[:]

    def hydrated_diff(self):
        if not self.initialized:
            raise DataError('Must call initialize() first')
        for lang in self.lang_codes:
            try:
                diff = (len(self.canonical_inv_idx[lang]) -
                        len(self.hydrated_inv_idx[lang]))
            except KeyError:
                diff = None
            print('{}: {}'.format(lang, diff))
