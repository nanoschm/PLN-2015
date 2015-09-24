# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from numpy import log2
from random import random
from math import fsum, log, floor
import numpy as np
from functools import partial
import operator


class NGram(object):

    def __init__(self, n, sents):
        """ n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)
        #self.probs = probs = defaultdict(partial(defaultdict, int))
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
                #probs[ngram[:-1]][ngram[-1]] += 1.0
                next[ngram[:-1]].append(ngram[-1])
        self.counts = dict(self.counts)
        #self.probs = dict(self.probs)
        self.next = dict(self.next)



    def __getstate__(self):
        """ This is called before pickling. """
        state = self.__dict__.copy()
        del state['sents']
        del state['next']
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
                prob = prob + log((self.cond_prob(token=token, prev_tokens=prev_tokens)),2.0)
            except ValueError:
                prob = float('-inf')

        return prob

class NGramGenerator(object):
 
    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self.ngram = model
        n = model.n

        self.probs = probs = defaultdict(partial(defaultdict, int))

        for ngram in model.counts.keys():
            if (len(ngram)) == n:
                prev_tokens = tuple(ngram[:-1])
                token = ngram[-1]
                probs[prev_tokens][token] = model.cond_prob(token, prev_tokens)

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
        p_diccionario = self.probs[prev_tokens]
        d_it = list(p_diccionario.items())

        x = 0
        acumular = 0.0
        for i in range(len(d_it)):
            acumular += d_it[i][1]

            if p_random > acumular:
                x += 1
            else:
                break
        word = d_it[x][0]
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
        ngrama = NGram(self.n, self.sents)
        dict_uni = [i[0] for i in ngrama.counts.items() if len(i) == 1]

        return len(dict_uni)

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
        num_held_out = floor((len(sents)) * 0.1)
        if num_held_out == 0:
            num_held_out = 1
            
        if not gamma:
            self.held_out = sents[len(sents)-num_held_out:]
            self.sents = sents[:len(sents)-num_held_out]
            lista_gammas_perplexity = list()
            lista_parametros = [i*100 for i in range(1,3)]
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
        else:
            self.sents = sents
            self.gamma = gamma

        self.counts = defaultdict(int)
      
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
        largo = len(tokens)
        lambda_list = list()
        for i in range(1,largo+1):
            esc = 1 - sum(lambda_list[0:i-1])
            count = self.counts[tokens[i-1:]]
            actual_lambda = esc * ( count / (count + self.gamma))
            lambda_list.append(actual_lambda)
        lambda_list.append(1-(fsum(lambda_list)))

        return lambda_list


    def cond_prob(self, token, prev_tokens=None):

        if not prev_tokens:     
            prev_tokens = []
        
        n = self.n
        largo = 1 + len(prev_tokens)
        for i in range(n - len(prev_tokens) - 1):
            prev_tokens = ["<s>"] + prev_tokens
        assert len(prev_tokens) == n - 1
        tokens = tuple(prev_tokens) + tuple([token])
        lambda_list = self.get_lambdas(tuple(prev_tokens))
        prob = 0.0
        for i in range(len(lambda_list)):
            num_qml = self.counts[tokens[i:]]
            den_qml = self.counts[tokens[i:-1]]
            try:
                prob = prob + (lambda_list[i] * float(num_qml)/den_qml)
            except ZeroDivisionError:
                prob = 0.0

        return prob


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


class BackOffNGram(AddOneNGram):
 
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
        self.addone = addone

        num_held_out = floor((len(sents)) * 0.1)
        if num_held_out == 0:
            num_held_out = 1

        if beta == None:
            print ("Calculando Beta")
            self.held_out = sents[len(sents)-num_held_out:]
            print(len(self.held_out))
            self.sents = sents[:len(sents)-num_held_out]
            lista_betas_perplexity = list()
            lista_parametros = [i*0.1 for i in range(1,11)]
            for i in range(len(lista_parametros)):
                print ("Calculando el BackOffNgram : ", i)
                backoff_model = BackOffNGram(n, self.sents, beta=lista_parametros[i], addone=addone)
                #Pasar esto a una clase.
                print ("Calculando log_prob")
                log_prob, m = backoff_model.log_prob(self.held_out)
                print ("Calculando cross_entropy")

                cross_entropy = backoff_model.cross_entropy(log_prob, m)
                print ("Calculando perplexitxy")

                perplexity = backoff_model.perplexity(cross_entropy)
                # ---

                lista_betas_perplexity.append(perplexity)
            print ("Calculando la mayor perplexity")
            index = lista_betas_perplexity.index(min(lista_betas_perplexity))
            self.beta = lista_parametros[index]
        else:
            self.beta = beta

        print ("Calculando tamaño del diccionario...")
        self.len_vocab = self.V()

        self.counts = defaultdict(int)
        self.next = defaultdict(list)
        print ("Calculando Counts")
        for i in range(1,n+1):
            print ("Calculando NGRAM", i)
            ngram = NGram(i, self.sents)
            ngram_c = ngram.counts
            self.counts.update(ngram_c)


      
    def cond_prob(self, token, prev_tokens=None):
        n = self.n
        if not prev_tokens:     
            prev_tokens = []
        if isinstance(token, tuple):
            t_token = token
        else:
            t_token = tuple([token])
        if len(prev_tokens) == 0:

            try:
                if not self.addone:
                    prob = float(self.counts[t_token]) / self.counts[()]
                else:
                    prob = (float(self.counts[t_token]) + 1) / (self.counts[()] + self.len_vocab)
            except KeyError:
                prob = 0.0
        elif len(prev_tokens) == 1:
            t_prev_tokens = tuple(prev_tokens)
            t_tokens = t_prev_tokens + t_token 
            try:
                c_estrella = self.counts[t_tokens] - self.beta
                prob = float(c_estrella) / float(self.counts[tuple(prev_tokens)])
            except KeyError:
                _A = self.A(t_prev_tokens)
                nexts = self.B(_A)
                
                try:
                    list_counts_nexts = [float(self.counts[tuple(n)])    / float(self.counts[tuple([])]) for n in nexts]
                    prob = self.alpha(t_prev_tokens) * ((float(self.counts[t_token])) / (float(self.counts[()]))) / (fsum(list_counts_nexts))
                except:
                    prob = 0.0
        else:
            t_prev_tokens = tuple(prev_tokens)
            t_tokens = t_prev_tokens + t_token 
            try:
                c_estrella = self.counts[(t_tokens)] - self.beta
                prob = float(c_estrella) / float(self.counts[t_prev_tokens])
            except KeyError:
                prob = self.alpha(t_prev_tokens)
                prob = prob * self.cond_prob(t_token, t_prev_tokens[1:])
                try:
                    prob = prob / self.denom(t_prev_tokens)
                except ZeroDivisionError:
                    prob = 0.0


        return prob

    def A(self, tokens):
        """Set of words with counts > 0 for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
        try:
            A = set(self.next[tokens])
        except KeyError:
            A = []
        return A

    def B(self, A):

        ngram = NGram(2, self.sents)
        aux = list(ngram.counts)

        for i in A:
            a = tuple([i])
            aux = [x for x in aux if x != a and len(x) < 2]
        return aux
 
    def alpha(self, tokens):
        """Missing probability mass for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
        list_of_next_words = self.A(tokens)

        try:
            alpha = float(len(list_of_next_words))*self.beta/self.counts[tokens]
        except KeyError:
            alpha = 0.0
        return alpha

    def denom(self, tokens):
        """Normalization factor for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """

        list_of_next_words = self.A(tokens)
        denom = 1.0
        list_estrella = list()
        for w in list_of_next_words:
            list_estrella.append(self.cond_prob(tuple([w]), tokens[1:]))

        denom = denom - sum(list_estrella)
        return denom

    def log_prob(self, t_sents):
        prob = 0.0
        count_tokens = 0
        x = 0
        for sent in t_sents:
            x += 1
            print (x)
            s_prob = self.sent_log_prob(sent)
            prob += s_prob
            count_tokens += len(sent)
        return prob, count_tokens

    def cross_entropy(self, log_prob, m):

        return -1*log_prob/float(m)

    def perplexity(self, cross_entropy):

        return pow(2, cross_entropy)