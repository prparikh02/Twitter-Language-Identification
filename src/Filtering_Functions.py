def filt_by_lang(X, y, labels_to_keep):
    indices = []
    for idx, label in enumerate(y):
        if label in labels_to_keep:
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


def change_lang(X, y):
    for idx, lang in enumerate(y):
        if lang in ['hr', 'sr', 'bs']:
            y[idx] = 'hr'
    return [X[i] for i in indices], [y[i] for i in indices]
