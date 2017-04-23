import re


def decode_unicode(X, y):
    for idx, tweet in enumerate(X):
        X[idx] = tweet.decode('utf-8')
    return X, y


def remove_newline_char(X, y):
    for idx, tweet in enumerate(X):
        X[idx] = tweet.replace(r'\n', ' ')
    return X, y


def remove_RT(X, y):
    for idx, tweet in enumerate(X):
        X[idx] = tweet.replace('RT', '')
    return X, y


def remove_urls(X, y):
    pattern = 'http\S+'
    p = re.compile(pattern)
    for idx, tweet in enumerate(X):
        if p.search(tweet):
            X[idx] = p.sub('', tweet)
    return X, y


def remove_handles(X, y):
    pattern = '@[a-z,A-Z]*'
    p = re.compile(pattern)
    for idx, tweet in enumerate(X):
        if p.search(tweet):
            X[idx] = p.sub('', tweet)
    return X, y


def remove_hashtags(X, y):
    pattern = '#[a-z,A-Z]*'
    p = re.compile(pattern)
    for idx, tweet in enumerate(X):
        if p.search(tweet):
            X[idx] = p.sub('', tweet)
    return X, y


def remove_emojis(X, y, emojis):
    for idx, x in enumerate(X):
        for emoji in emojis:
            x = x.replace(emoji, '')
        X[idx] = x
    return X, y
