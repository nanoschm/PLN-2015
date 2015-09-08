# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from numpy import log2

class NGram(object):

    def __init__(self, n, sents):
        """ n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)

        for sent in sents:
            for i in range(n-1):
                sent = ["<s>"] + sent
            sent = sent + ["</s>"]
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1

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
        ngram = model
        
 
    def generate_sent(self):
        """Randomly generate a sentence."""
 
    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.
 
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """