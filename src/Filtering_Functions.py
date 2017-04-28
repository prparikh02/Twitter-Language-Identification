def filt_by_lang(X, y, langs, exclude=False):
    def predicate(lang):
        if exclude:
            return lang not in langs
        return lang in langs

    indices = []
    for idx, lang in enumerate(y):
        if predicate(lang):
            indices.append(idx)
    return [X[i] for i in indices], [y[i] for i in indices]


def filt_long_length(X, y, max_length=140):
    indices = []
    for idx, tweet in enumerate(X):
        if len(tweet) <= max_length:
            indices.append(idx)
    return [X[i] for i in indices], [y[i] for i in indices]


def filt_short_length(X, y, min_length=5):
    indices = []
    for idx, tweet in enumerate(X):
        if len(tweet) >= min_length:
            indices.append(idx)
    return [X[i] for i in indices], [y[i] for i in indices]


def relabel(X, y):
    for idx, lang in enumerate(y):
        if lang in ['hr', 'sr', 'bs']:
            y[idx] = 'hr'
    return [X[i] for i in indices], [y[i] for i in indices]


def count_thres(X, y, n, thres_type='at_least'):
    '''types: at_least, at_most, exactly'''

    if thres_type not in ['at_least', 'at_most', 'exactly']:
        thres_type = 'at_least'

    def predicate(v):
        if thres_type in ['at_least', 'exactly']:
            return v >= n
        return v <= n

    indices = []
    lang_counts = dict(collections.Counter(y))
    lang_counts = {k: v for k, v in lang_counts.iteritems() if predicate(v)}
    if thres_type == 'exactly':
        lang_counts = {k: n for k in lang_counts}
    for idx, lang in enumerate(y):
        if lang in lang_counts:
            if lang_counts[lang] == 0:
                continue
            indices.append(idx)
            lang_counts[lang] -= 1
    return [X[i] for i in indices], [y[i] for i in indices]
