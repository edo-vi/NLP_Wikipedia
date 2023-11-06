from collections import Counter
import numpy as np
import nltk
from nltk.stem.snowball import EnglishStemmer
import re


class MyCounter:
    def __init__(self, tokens, stemmed=False) -> None:
        self._stemmed = stemmed
        tokens = [t.lower() for t in tokens]
        self.counts = Counter(tokens)
        self.stemmer = EnglishStemmer()

    def update(self, other_counter):
        self.counts = self.counts + other_counter.counts

    def most_common(self, n):
        return self.counts.most_common(n)

    def c(self, key):
        if self._stemmed:
            key = self.stemmer.stem(key.lower()).lower()
        else:
            key = key.lower()
        try:
            return self.counts[key]
        except:
            return 0

    def contains(self, word):
        return word in self.counts

    def N(self):
        return np.array(list(self.counts.values())).sum()

    def V(self):
        return len(self.counts.keys())

    def log_likelihood(self, word, laplace_correction=True):
        return np.log(self.freq(word, laplace_correction=laplace_correction))

    def freq(self, word, laplace_correction=True):
        n_w = self.counts[word]
        V = self.N()

        if laplace_correction:
            return (n_w + 1) / (V + self.V())
        else:
            return (n_w) / V

    def log_likelihood_document(self, other):
        ll = 0
        for word in self.counts:
            ll += other.log_likelihood(word)
        return ll

    def remove_stopwords(self):
        stopwords = nltk.corpus.stopwords.words("english")
        new_counts = {}
        for word in self.counts:
            # If word is a stopword, or has length less than 2, or is ALL special characters
            if word in list(stopwords) or len(word) < 2 or re.match(r"^[_\W]+$", word):
                continue
            else:
                new_counts[word] = self.counts[word]
        del self.counts
        self.counts = new_counts
        return self

    def stem(self):
        self._stemmed = True

        stems = dict()
        for word in self.counts.keys():
            value = self.counts[word]

            stemmed = self.stemmer.stem(word).lower()
            if stemmed in stems.keys():
                value += stems[stemmed]
            stems[stemmed] = value
        del self.counts
        self.counts = Counter(stems)

        return self

    def keys(self):
        return list(self.counts.keys())

    def __str__(self):
        return self.counts.__str__()
