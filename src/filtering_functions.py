def filt_by_lang(X, y, labels_to_keep):
    indices = []
    for idx, label in enumerate(y):
        if label in labels_to_keep:
            indices.append(idx)
    return [X[i] for i in indices], [y[i] for i in indices]
