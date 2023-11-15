from collections import Counter
import numpy as np
import nltk
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
import re


# I.e. a Bag Of Word representation
class BagOfWord:
    def __init__(self, tokens, stemmed=False) -> None:
        self._stemmed = stemmed
        tokens = [t.lower() for t in tokens]
        self.counts = Counter(tokens)
        self.stemmer = EnglishStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self._other_class_V = -1

    def update(self, other_counter):
        self.counts = self.counts + other_counter.counts

    def add_other_class_V(self, other_class_V):
        self._other_class_V = other_class_V

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

    # Number of running words in total
    def N(self):
        return np.array(list(self.counts.values())).sum()

    # Number of unique words, i.e. the size of the dictionary
    def V(self):
        return len(self.counts.keys())

    def log_likelihood(self, word, laplace_correction=True):
        return np.log(self.freq(word, laplace_correction=laplace_correction))

    def freq(self, word, laplace_correction=True):
        assert self._other_class_V != -1
        n_w = self.counts[word]

        # We don't add N of the other class to get the correct vocabulary N because it's still zero
        N = self.N()

        if laplace_correction:
            # we add the other |V| to get the correct vocaulary size, see pag 63 of book
            return (n_w + 1) / (N + self.V() + self._other_class_V)
        else:
            return (n_w) / N

    # Compute the log likelihood sum of all the words in 'this', based on the counts of other_document
    def log_likelihood_sum(self, other_document):
        ll = 0
        for word in self.counts:
            ll += other_document.log_likelihood(word)
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
        self.counts = Counter(new_counts)
        return self

    def stem(self):
        return self.lemmatize()

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

    def lemmatize(self):
        self._stemmed = True

        stems = dict()
        for word in self.counts.keys():
            value = self.counts[word]

            stemmed = self.lemmatizer.lemmatize(word)
            # Could already be present in the temporary dictionary: add the previous value to
            # the counts and then (line 104) update stems[stemmed] with the value (if already present it gets overwritten)
            if stemmed in stems.keys():
                value += stems[stemmed]
            stems[stemmed] = value
        # delete previous counts and add the stemmed/lemmatized versione
        del self.counts
        self.counts = Counter(stems)

        return self

    def keys(self):
        return list(self.counts.keys())

    def __str__(self):
        return self.counts.__str__()
