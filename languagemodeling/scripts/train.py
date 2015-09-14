"""
Train an n-gram model.

Usage:
  train.py -n <n> [-m <model>] -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from nltk.corpus import PlaintextCorpusReader

from languagemodeling.ngram import NGram, AddOneNGram



if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    corpus = PlaintextCorpusReader('.', 'raw.txt')

    sents = corpus.sents()
    n = int(opts['-n'])
    m = int(opts['m'])
    if m:
      #train the addone model
      model = AddOneNGram(n, sents)
    else:
      # train the simple model
      model = NGram(n, sents)
     # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()

