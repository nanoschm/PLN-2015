"""Train an n-gram model.

Usage:
  train.py -n <n> -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from nltk.corpus import PlaintextCorpusReader

from languagemodeling.ngram import NGram



if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    corpus = PlaintextCorpusReader('.', 'raw.txt')

    sents = corpus.sents()

    # train the model
    n = int(opts['-n'])
    model = NGram(n, sents)
    generate = NGramGenerator(model)
    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
