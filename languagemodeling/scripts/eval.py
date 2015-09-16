# -*- coding: utf-8 -*-

"""
Evaulate a language model using the test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.

  """


from docopt import docopt
import pickle

from nltk.corpus import PlaintextCorpusReader

from languagemodeling.ngram import NGramGenerator

def log_prob(model, t_sents):
    number_of_words = model.V()
    prob = 0.0
    for sent in t_sents:
        s_prob = model.sent_log_prob(sent)
        prob += s_prob

    return float(prob)/number_of_words
def log_probability(model):

def cross_entropy(model):




if __name__ == '__main__':
    opts = docopt(__doc__)
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)

    corpus_10 = PlaintextCorpusReader('.', 'Corpus_10.0')
    test_sents = corpus.sents()

    log_prob(model, test_sents)


    #perplexiti







