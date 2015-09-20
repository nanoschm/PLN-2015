"""Generate natural language sentences using a language model.

Usage:
  generate.py -i <file> -n <n>
  generate.py -h | --help

Options:
  -i <file>     Language model file.
  -n <n>        Number of sentences to generate.
  -h --help     Show this screen.

"""
from docopt import docopt
import pickle

from nltk.corpus import PlaintextCorpusReader

from languagemodeling.ngram import NGramGenerator



if __name__ == '__main__':
    opts = docopt(__doc__)
    n = int(opts['-n'])
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    sent_generator = NGramGenerator(model)

    for i in range(n):
      print ("Generando oración numero : \n" , i )
      sent = sent_generator.generate_sent()
      print (sent)



