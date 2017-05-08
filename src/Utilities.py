import cPickle as pickle
import itertools
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm,
                          classes,
                          normalize=True,
                          title='Confusion matrix',
                          tile_text=False,
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if classes:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    if tile_text:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def load_emojis(pickle_file='../data/emoji_unicode.pkl'):

    with open(pickle_file, 'rb') as f:
        emojis = pickle.load(f)
    return emojis

def show_under_performed(expt, cm, thres=0.50):
    diags = cm.diagonal()
    under_peformed = \
        {expt.num_to_lang[i]: diags[i] for i in np.where(diags < thres)[0]}
    s = '{} / {} underperformed (below {} accuracy)'
    print(s.format(len(under_peformed), len(diags), thres))
    return under_peformed