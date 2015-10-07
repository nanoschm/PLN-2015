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
                  interpolated: Suavizado por Interpolación
                  backoff: Suavizado por Back-Off con Discounting
  -a            ONLY IN INTERPOLATED or BACKOFF MODEL - addone to unigram model
  -g            ONLY IN INTERPOLATED MODEL - Gamma to Interpolated Model
  -b            ONLY IN BACKOFF MODEL - Beta to Back-Off Discounting Model
  -o <file>     Output model file.
  -h --help     Show this screen.
"""

from docopt import docopt
import pickle

from nltk.corpus import PlaintextCorpusReader

from languagemodeling.ngram import NGram, AddOneNGram, InterpolatedNGram, BackOffNGram

if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    print ("Leyendo Corpus...")
    corpus = PlaintextCorpusReader('.', 'languagemodeling/scripts/Corpus90.0')

    sents = corpus.sents()
    n = int(opts['-n'])
    m = str(opts['-m'])
    #train the addone model
    print ("Entrenando modelo...")
    if m == "backoff":
      print ("Model Back-Off")
      model = BackOffNGram(n, sents)
    elif m == "interpolated":
      print ("Model interpolated")
      model = InterpolatedNGram(n, sents)
    elif m == "addone":
      print ("Model AddOne")
      model = AddOneNGram(n, sents)
    elif m == "ngram" :
      print ("Model simple ngram")
      model = NGram(n, sents)
    else:
      print("Error, try with \'-h\' option")

     # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    print ("Dumpeando el modelo a un archivo")
    pickle.dump(model, f)
    f.close()

