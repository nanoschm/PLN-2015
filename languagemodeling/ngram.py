# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from numpy import log2
from random import random
from math import fsum, log, floor
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
        self.next = next = defaultdict(list)
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
                next[ngram[:-1]].append(ngram[-1])


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
            prob = float(self.counts[tuple(tokens)]) / self.counts[tuple(prev_tokens)]
        except ZeroDivisionError:
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
        for i in range (len(n_sent)-n+1):
            token = ((n_sent[i+n-1]))
            r = max(0,i-n+1)
            prev_tokens = (n_sent[i:i+n-1])
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
        self.n = n
        a = len(sents)
        num_held_out = floor((len(sents)) * 0.1)
        if num_held_out == 0:
            num_held_out = 1
            
        if not gamma:
            self.held_out = sents[len(sents)-num_held_out:]
            self.sents = sents[:len(sents)-num_held_out]
            lista_gammas_perplexity = list()
            lista_parametros = [i*100 for i in range(1,3)]
            print (lista_parametros)
            for i in range(len(lista_parametros)):
                interp_model = InterpolatedNGram(n, self.sents, gamma=lista_parametros[i], addone=addone)
                
                #Pasar esto a una clase.
                log_prob, m = interp_model.log_prob(self.held_out)
                cross_entropy = interp_model.cross_entropy(log_prob, m)
                perplexity = interp_model.perplexity(cross_entropy)
                # ---

                lista_gammas_perplexity.append(perplexity)
            index = lista_gammas_perplexity.index(min(lista_gammas_perplexity))
            self.gamma = lista_parametros[index]
            print ("Gamma = ", self.gamma )
        else:
            self.sents = sents
            self.gamma = gamma

      
        if not addone:
            for i in range(1,n+1):
                ngram = NGram(i, self.sents).counts
                self.counts.update(ngram)

        else:
            for i in range(2,n+1):
                ngram = NGram(i, self.sents).counts
                self.counts.update(ngram)
            unigram = AddOneNGram(1, self.sents).counts
            self.counts.update(unigram)


    def get_lambdas(self, tokens):
        n = self.n
        lambda_list = list()
        for i in range(1,n):
            esc = 1 - sum(lambda_list[0:i-1])
            count = self.counts[tokens[i:n-i]]
            actual_lambda = esc * ( count / (count + self.gamma))
            lambda_list.append(actual_lambda)

        lambda_list.append(1-(sum(lambda_list)))

        return lambda_list


    def cond_prob(self, token, prev_tokens=None):
        n = self.n
        if not prev_tokens:     
            prev_tokens = []

        for i in range(n - len(prev_tokens) - 1):
            prev_tokens = ["<s>"] + prev_tokens
        assert len(prev_tokens) == n - 1
        tokens = tuple(prev_tokens) + tuple([token])
        print ("tokens...")
        print (tokens)
        lambda_list = self.get_lambdas(tokens)
        print ("lambda list...")
        print (lambda_list) 

        prob = 1.0
        for i in range(len(lambda_list)):

            num_qml = self.counts[tokens[:n-i]]
            den_qml = self.counts[tokens[:n-(i+1)]]
            try:
                prob = prob * lambda_list[i] * float(num_qml)/den_qml
            except ZeroDivisionError:
                prob = 0.0

        return prob




class BackOffNGram(NGram):
 
    def __init__(self, n, sents, beta=None, addone=True):
        """
        Back-off NGram model with discounting as described by Michael Collins.
 
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        beta -- discounting hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        self.n = n
        self.sents = sents

        #cambiar para usar barrido, ¿Que debería calcular? FIX

        num_held_out = floor((len(sents)) * 0.1)
        if num_held_out == 0:
            num_held_out = 1

        if not beta:
            self.held_out = sents[len(sents)-num_held_out:]
            self.sents = sents[:len(sents)-num_held_out]
            lista_betas_perplexity = list()
            lista_parametros = [i*0.1 for i in range(1,11)]
            print (lista_parametros)
            for i in range(len(lista_parametros)):
                backoff_model = BackOffNGram(n, self.sents, beta=lista_parametros[i], addone=addone)
                
                #Pasar esto a una clase.
                log_prob, m = backoff_model.log_prob(self.held_out)
                cross_entropy = backoff_model.cross_entropy(log_prob, m)
                perplexity = backoff_model.perplexity(cross_entropy)
                # ---

                lista_gammas_perplexity.append(perplexity)
            index = lista_betas_perplexity.index(min(lista_betas_perplexity))
            self.beta = lista_parametros[index]
            print ("Beta = ", self.beta )
        else:
            self.beta = beta

        self.counts = defaultdict(int)
        self.next = defaultdict(list)   
        if not addone:
            for i in range(1,n+1):
                ngram = NGram(i, self.sents)
                ngram_c = ngram.counts
                ngram_n = ngram.next
                self.counts.update(ngram_c)
                self.next.update(ngram_n)

        else:
            for i in range(2,n+1):
                ngram = NGram(i, self.sents)               
                ngram_c = NGram(i, self.sents).counts
                ngram_n = ngram.next
                self.counts.update(ngram_c)
                self.next.update(ngram_n)
            unigram = AddOneNGram(1, self.sents)
            unigram_c = unigram.couts
            unigram_n = unigram.next
            self.counts.update(unigram_c)
            self.next.update(unigram_n)

        self.next = dict(self.next)
        self.counts = dict(self.counts)

    def prob_aux_1(self, t_token, t_prev_tokens):

        t_tokens = t_token + t_prev_tokens


    def cond_prob(self, token, prev_tokens):
        n = self.n
        if not prev_tokens:     
            prev_tokens = []

        for i in range(n - len(prev_tokens) - 1):
            prev_tokens = ["<s>"] + prev_tokens
        assert len(prev_tokens) == n - 1
        t_token = tuple([token])
        t_prev_tokens = tuple(prev_tokens)
        t_tokens = t_prev_tokens + t_token 
        
        try:
            c_estrella = self.counts[t_tokens] - self.beta
            float(c_estrella) / float(self.counts[prev_tokens])        
        except KeyError:
            prob = self.alpha(t_tokens) * self.cond_prob(t_prev_tokens[1:], t_token ) / self.denom(t_prev_tokens)



    def A(self, tokens):
        """Set of words with counts > 0 for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
        return set(self.next[tokens])
 
    def alpha(self, tokens):
        """Missing probability mass for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """

        list_of_next_words = self.A(tokens)

        alpha = 1.0
        for w in list_of_next_words:
            alpha = alpha - self.cond_prob(tuple(w), tokens)

    def denom(self, tokens):
        """Normalization factor for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """

        list_of_next_words = self.A(tokens)
        denom = 1.0
        for w in list_of_next_words:
            denom = denom - self.cond_prob(tuple(w), tokens[1:])

    def log_prob(self, t_sents):
        prob = 0.0
        count_tokens = 0
        for sent in self.sents:
            s_prob = self.sent_log_prob(sent)
            prob += s_prob
            count_tokens += len(sent)
        return prob, count_tokens

    def cross_entropy(self, log_prob, m):

        return -1*log_prob/float(m)

    def perplexity(self, cross_entropy):

        return pow(2, cross_entropy)