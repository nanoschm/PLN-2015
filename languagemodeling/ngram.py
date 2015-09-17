# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from numpy import log2
from random import random
from math import fsum, log
import numpy as np
from functools import partial

class NGram(object):

    def __init__(self, n, sents):
        """ n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)
        self.probs = probs = defaultdict(partial(defaultdict, int))
        self.sents = sents
        for sent in sents:
            for i in range(n-1):
                sent = ["<s>"] + sent
            sent = sent + ["</s>"]
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1
                # Armamos lo que va a ser el futuro diccionario de probabilidades en NGramGenerator
                probs[ngram[:-1]][ngram[-1]] += 1.0

    def __getstate__(self):
        """ This is called before pickling. """
        state = self.__dict__.copy()
        del state['sents']
        return state


    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.
 
        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return (self.counts[tokens])
 

    def cond_prob(self, token, prev_tokens=None):
        n = self.n
        if not prev_tokens:     
            prev_tokens = []

        for i in range(n - len(prev_tokens) - 1):
            prev_tokens = ["<s>"] + prev_tokens
        assert len(prev_tokens) == n - 1
        tokens = (list(prev_tokens)) + [(token)]

        try:
            print ("NUM" + str(self.counts[tuple(tokens)]))
            print ("DEN" + str(self.counts[tuple(prev_tokens)]))
            prob = float(self.counts[tuple(tokens)]) / self.counts[tuple(prev_tokens)]
            print ("a")
        except ZeroDivisionError:
            print("b")
            prob = float(0.0)
        return prob

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.
 
        sent -- the sentence as a list of tokens.
        """
        n = self.n
        prob = 1.0

        n_sent = (["<s>"] * (n - 1) + list(sent) + ["</s>"]) 
        for i in range(n-1,len(n_sent)):
            token = (n_sent[i])
            r = max(0,i-n+1)
            prev_tokens = (n_sent[r:i])

            prob = float(prob) * float(self.cond_prob(token=token, prev_tokens=prev_tokens))
            
        return prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.
 
        sent -- the sentence as a list of tokens.
        """
        n = self.n
        prob = 0.0
        n_sent = ["<s>"] * (n - 1) + list(sent) + ["</s>"]
        print (len(n_sent))
        for i in range (len(n_sent)-n+1):
            token = ((n_sent[i+n-1]))
            r = max(0,i-n+1)
            prev_tokens = (n_sent[i:i+n-1])
            print (token) 
            print (prev_tokens)
            try:
                prob = prob + log((self.cond_prob(token=token, prev_tokens=prev_tokens)),2)
            except ValueError:
                prob = float('-inf')

        return prob

class NGramGenerator(object):
 
    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self.ngram = model
        self.probs = model.probs
        for elem in self.ngram.probs.items():
            count = 0
            for o_elem in elem[1].items():
                count += o_elem[1]
            for o_elem in elem[1].items():
                elem[1][o_elem[0]] = float(o_elem[1])/float(count)

    def generate_sent(self):
        """Randomly generate a sentence."""
        n = self.ngram.n
        sent = tuple([])
        for i in range(n-1):
            sent = sent + tuple(["<s>"])
        while(1):
            if n >= 2:
                sent = sent + tuple([self.generate_token(sent[-1*n+1:])])
            else:
                sent = sent + tuple([self.generate_token()])
            if sent[-1]=="</s>":
                break
        
        return sent[n-1:-1]

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.
 
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """

        #generamos un numero aleatorio entre 0 y 1.

        p_random = random()
        if not prev_tokens:
            prev_tokens = tuple([])
        #Buscamos el Dicciionario de Probabilidad de la palabra siguiente a prev_tokens
        p_diccionario = self.ngram.probs[prev_tokens]
        xk = [i for i in range(len(p_diccionario.items()))]
        
        xk, yk = list(p_diccionario.keys()), list(p_diccionario.values())
        # Creamos una lista con indice del elemento en p_diccionario, y el valor acumulado.
        word_prob = [(xk[i], fsum(yk[0:i])) for i in range(len(xk))]
        length_wp = len(word_prob)
        for i in range(length_wp):
            if (word_prob[length_wp-i-1])[1] < p_random and i < length_wp:
                index = length_wp-i-1
                break
            elif word_prob[length_wp-i-1][1] > p_random and i == length_wp - 1:
                index = 0
                break
        word = xk[index]

        return word

class AddOneNGram(NGram):
    
    def __getstate__(self):
        """ This is called before pickling. """
        state = self.__dict__.copy()
        del state['sents']
        return state

    def V(self):
        """Size of the vocabulary.
        """ 
        set_of_words = set(["</s>"])
        for sent in self.sents:
            set_of_words = set_of_words.union(set(sent))

        return len(set_of_words)

    def cond_prob(self, token, prev_tokens=None):
        n = self.n
        if not prev_tokens:     
            prev_tokens = []

        for i in range(n - len(prev_tokens) - 1):
            prev_tokens = ["<s>"] + prev_tokens
        assert len(prev_tokens) == n - 1
        tokens = tuple(prev_tokens) + tuple([token])
        try:    
            prob = (float(self.counts[tuple(tokens)]) + 1) / (self.counts[tuple(prev_tokens)] + self.V())
        except ZeroDivisionError:
            prob = 0.0
        return prob

class InterpolatedNGram(NGram):

    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        super(InterpolatedNGram, self).__init__(n, sents)
        if gamma:
            self.gamma = gamma
        else:
            self.gamma = 1
        
        self.list_of_counts = list()
        for i in range(1,n+1):
            ngram = NGram(i, sents).counts
            self.list_of_counts.append(ngram)
        self.counts = self.list_of_counts[n-1]
    def get_lambdas(self, tokens):
        n = self.n
        lambda_list = list()
        for i in range(1,n+1):
            esc = 1 - sum(lambda_list[1:i-1])
            dict_of_counts = self.list_of_counts[len(tokens[i:n-i])]
            count = dict_of_counts[tokens[i:n-i]]
            actual_lambda = esc * ( count / (count + self.gamma))

        return lambda_list


    def cond_prob(self, token, prev_tokens=None):
        n = self.n
        if not prev_tokens:     
            prev_tokens = []

        for i in range(n - len(prev_tokens) - 1):
            prev_tokens = ["<s>"] + prev_tokens
        assert len(prev_tokens) == n - 1
        tokens = tuple(prev_tokens) + tuple([token])

        lambda_list = self.get_lambdas(tokens)

        prob = 1.0
        for i in range(len(lambda_list)):

            dict_of_counts = self.list_of_counts[len(tokens[:n-i])]
            num_qml = dict_of_counts[tokens[:n-i]]
            dict_of_counts2 = self.list_of_counts[len(tokens[:n-i])]
            den_qml = dict_of_counts2[tokens[:n-(i+1)]]
            prob = prob * lambda_list[i] * float(num_qml)/den_qml

        return prob




    
