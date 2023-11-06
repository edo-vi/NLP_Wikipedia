from collections import Counter
import numpy as np
import nltk
from nltk.stem.snowball import EnglishStemmer


class MyCounter:
    def __init__(self, tokens) -> None:
        self._stemmed = False
        tokens = [t.lower() for t in tokens]
        self.counts = Counter(tokens)
        self.stemmer = EnglishStemmer()

    def update(self, other_counter):
        self.counts = self.counts + other_counter.counts

    def c(self, key):
        if self._stemmed:
            key = self.stemmer.stem(key.lower()).lower()
        else:
            key = key.lower()
        try:
            return self.counts[key]
        except:
            return 0

    def N(self):
        return np.array(list(self.counts.values())).sum()

    def remove_stopwords(self):
        stopwords = nltk.corpus.stopwords.words("english")
        new_counts = {}
        for word in self.counts:
            if not (word in list(stopwords) or len(word) < 2):
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
