# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from numpy import log2
from random import random
from math import fsum
import numpy as np

import scipy.stats as st
class NGram(object):

    def __init__(self, n, sents):
        """ n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)
        self.probs = probs = defaultdict(lambda : defaultdict(int))
        for sent in sents:
            for i in range(n-1):
                sent = ["<s>"] + sent
            sent = sent + ["</s>"]
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1
                probs[ngram[:-1]][ngram[-1]] += 1.0


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
        tokens = tuple(prev_tokens) + tuple([token])
        try:
            prob = float(self.counts[tuple(tokens)]) / self.counts[tuple(prev_tokens)]
        except ZeroDivisionError:
            prob = 0.0
        return prob

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.
 
        sent -- the sentence as a list of tokens.
        """
        n = self.n
        prob = 1.0
        n_sent = tuple(list(sent) + ["</s>"])
        for i in range(len(n_sent)):
            token = n_sent[i]
            r = max(0,i-n+1)
            prev_tokens = tuple(n_sent[r:i])
            prob = prob * self.cond_prob(token=token, prev_tokens=prev_tokens)

        return prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.
 
        sent -- the sentence as a list of tokens.
        """
        prob = self.sent_prob(sent)
        log_prob = log2(prob)


        return log_prob

class NGramGenerator:
 
    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self.ngram = model

        for elem in self.ngram.probs.items():
            count = 0
            for o_elem in elem[1].items():
                count += o_elem[1]
            for o_elem in elem[1].items():
                elem[1][o_elem[0]] = float(o_elem[1])/float(count)

        self.probs = model.probs



    def generate_sent(self):
        """Randomly generate a sentence."""
        n = self.ngram.n
        sent = tuple([])
        for i in range(n-1):
            sent = tuple(["<s>"])      
        while(1):
            if n >= 2:
                print sent
                print sent[-1*n:]
                sent = sent + tuple([self.generate_token(sent[-1*n+1:])])
            else:
                sent = sent + tuple([self.generate_token()])
            if sent[-1]=="</s>":
                break
        
        print sent
        return sent

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
        
        xk, yk = p_diccionario.keys(), p_diccionario.values()
        word_prob = [(xk[i], fsum(yk[0:i])) for i in range(len(xk))]
        length_wp = len(word_prob)
        print word_prob
        for i in range(length_wp):
            if (word_prob[length_wp-i-1])[1] < p_random and i < length_wp:
                index = length_wp-i-1
                print "a"
                break
            elif word_prob[length_wp-i-1][1] > p_random and i == length_wp - 1:
                index = 0
                print "b"
                break
            else:
                pass
                print "c"
        
        word = p_diccionario.keys()[index]

        return word

    def create_distribution(self, xk, pk):
        rv = st.rv_discrete(xk[0], xk[-1], values=(xk, pk)) 
        return rv 

