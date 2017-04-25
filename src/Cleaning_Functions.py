import re
import string


def decode_unicode(X, y):
    for idx, text in enumerate(X):
        X[idx] = text.decode('utf-8')
    return X, y


def remove_newline_char(X, y):
    for idx, text in enumerate(X):
        X[idx] = text.replace(r'\n', ' ')
    return X, y


def remove_RT(X, y):
    for idx, text in enumerate(X):
        X[idx] = text.replace('RT', '')
    return X, y


def remove_urls(X, y):
    pattern = 'http\S+'
    p = re.compile(pattern)
    for idx, text in enumerate(X):
        if p.search(text):
            X[idx] = p.sub('', text)
    return X, y


def remove_handles(X, y):
    pattern = '@[a-z,A-Z]*'
    p = re.compile(pattern)
    for idx, text in enumerate(X):
        if p.search(text):
            X[idx] = p.sub('', text)
    return X, y


def remove_hashtags(X, y):
    pattern = '#[a-z,A-Z]*'
    p = re.compile(pattern)
    for idx, text in enumerate(X):
        if p.search(text):
            X[idx] = p.sub('', text)
    return X, y


def remove_punctutation(X, y):
    for idx, text in enumerate(X):
        X[idx] = text.translate(None, string.punctuation)
    return X, y


def remove_emojis(X, y, emojis):
    for idx, x in enumerate(X):
        for emoji in emojis:
            x = x.replace(emoji, '')
        X[idx] = x
    return X, y
