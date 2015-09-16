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
    prob = 0.0
    count_tokens = 0
    for sent in t_sents:
        s_prob = model.sent_log_prob(sent)
        prob += s_prob
        count_tokens += len(sent)
    return prob, count_tokens

def cross_entropy(log_prob, m):

    return -1*log_prob/float(m)

def perplexity(cross_entropy):

    return pow(2, cross_entropy)

if __name__ == '__main__':
    opts = docopt(__doc__)
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)

    corpus_10 = PlaintextCorpusReader('.', 'Corpus_10.0')
    test_sents = corpus.sents()

    log_prob, m = log_prob(model, test_sents)
    cross_entropy = cross_entropy(log_prob, m)
    perplexity = perplexity(cross_entropy)

    print ("the perplexity is %s", str(perplexity))






