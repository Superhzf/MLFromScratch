from __future__ import division
import numpy as np
from itertools import combinations_with_replacement
import os
import re
from collections import Counter

_WORD_REGEX = re.compile(r"(?u)\b\w\w+\b")  # sklearn default
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
_STOP_WORDS = []

def batch_generator(X, y=None, batch_size=64):
    """Simple batch geneartor"""
    n_samples = X.shape[0]
    for i in np.arange(0,n_samples,batch_size):
        begin,end = i,min(i+batch_size,n_samples)
        if y is not None:
            yield X[begin:end],y[begin:end]
        else:
            yield X[begin:end]

def divide_on_feature(X,feature_i,threshold):
    """
    Divide dataset X based on if sample value on feature_i is larger than
    the given threshold
    """
    split_func = None
    if isinstance(threshold,int) or isinstance(threshold,float) or isinstance(threshold, np.float32):
        X_1 = X[X[:,feature_i]>=threshold]
        X_2 = X[X[:,feature_i]<threshold]
    else:
        X_1 = X[X[:,feature_i]==threshold]
        X_2 = X[X[:,feature_i]!=threshold]

    return np.array([X_1,X_2])


def to_categorical(x,n_col=None):
    """
    one-hot encoding of norminal values
    """
    if not n_col:
        n_col = np.amax(x)+1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


def polynomial_features(X,degree):
    n_samples,n_features = X.shape(X)

    def index_combinations():
        combs = [combinations_with_replacement(range(n_features),i) for i in range(0,degree+1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs

    combinnations = index_combinations()
    n_output_features = len(combinnations)
    X_new = np.empty((n_samples,n_output_features))

    for i,index_combs in enumerate(combinations):
        X_new[:,i] = np.prod(X[:,index_combs],axis=1)

    return X_new


def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    # l2[l2 == 0] = 1???
    return X / np.expand_dims(l2, axis)


def remove_stop_words(words):
    """Remove stop words from a list of word strings"""
    return [w for w in words if w not in _STOP_WORDS]


def tokenize_words(line, lowercase=True, filter_stopwords=True):
    """
    Split a line into individual lower-case words, optionally removing punctuation
    and stop-words in the process
    """
    words = _WORD_REGEX.findall(line.lower() if lowercase else line)
    return remove_stop_words(words) if filter_stopwords else words


class Token:
    def __init__(self, word):
        self.count = 0
        self.word = word

    def __repr__(self):
        return "Token(word='{}', count={})".format(self.word, self.count)


class Vocabulary:
    def __init__(self, lowercase=True, min_count=None, max_count=None, filter_stopwords=True):
        """
        An object for compiling and encoding the unique tokens in a text corpus.

        Parameters:
        ------------------
        lowercase: bool
            Whether to convert each string to lowercase before tokenization.
        min_count: int
            Minimum number of times a token must occur in order to be included
            in the vocabulary. If None, include all tokens.
        max_tokens: int
            Only add the max_tokens most frequent tokens that occur more than
            min_count to the vocabulary. If None, add all tokens greater than
            min_count
        filter_stopwords: bool
            Whether to remove stopwords before encoding the words in the corpus.
        """
        self.hyperparameters = {
            'id': "Vocabulary",
            'encoding': None,
            'corpus_fps': None,
            'lowercase': lowercase,
            'min_count': min_count,
            'max_tokens': max_tokens,
            'filter_stopwords': filter_stopwords
        }

    def __len__(self):
        return len(self._tokens)

    def __iter__(self):
        return iter(self._tokens)

    def __contains__(self, word):
        return word in self.token2idx

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._tokens[self.token2idx[key]]
        if isinstance(key, int):
            return self._tokens[key]

    @property
    def n_tokens(self):
        """The number of unique word tokens in the vocabulary"""
        return len(self.token2idx)

    @property
    def n_words(self):
        """The total number of words in the corpus"""
        return sum(self.counts.values())

    def most_common(self, n=5):
        """Return the top `n` most common tokens in the corpus"""
        return self.counts.most_common()[:n]

    def words_with_count(self, k):
        """Return all tokens that occur `k` times in the corpus"""
        return [w for w, c in self.counts.items() if c == k]

    def filter(self, words, unk=True):
        """
        Filter or replace any word in words that does not show up in Vocabulary

        Parameters:
        ----------------
        words: List[str]
            A list of words to filter
        unk: bool
            Whether to replace any out of vocabulary words in words with '<unk>'
            token.

        Returns
        ----------------
        filtered : List[str]
            The list of words filtered against the vocabulary.
        """
        if unk:
            return [w if w in self else '<unk>' for w in words]
        return [w for w in words if w in words]

    def words_to_indices(self, words):
        """
        Convert the words in words to token indices. If a word is not in the
        vocabulary, return the index for the <unk> token.

        Parameters:
        ------------------
        words: List[str]
            A list of words

        Returns:
        -----------------
        indices: List[int]
            The token indices for each word in words
        """
        unk_ix = self.token2idx['<unk>']
        lowercase = self.hyperparameters["lowercase"]
        words = [w.lower() for w in words] if lowercase else words
        return [self.token2idx[w] if w in self else unk_ix for w in words]

    def indices_to_words(self, indices):
        """
        Convert the indices to words. If an index is not in the vocabulary,
        return <unk>

        Parameters:
        ----------------
        indices: List[int]
            Token indices

        Returns:
        --------------
        words: List[str]
            Words that corresponds to indices
        """
        unk = '<unk>'
        return [self.idx2tokens[i] if i in self.idx2tokens else unk for i in indices]

    def fit(self, corpus_fps, encoding='utf-8-sig'):
        """
        Compute the vocabulary across a collection of documents.

        Parameters:
        ------------------
        corpus_fps : str or List[str]
            The filepath / list of filepaths for the documents to be encoded.
            Each document is expected to be encoded as newline-separated string
            or text, with adjacent tokens separated by a whitespace.
        encoding : str
            Specifies the text encoding for corpus. Common entries are either
            utf-8 or utf-8-sig.
        """
        if isinstance(corpus_fps, str):
            corpus_fps = [corpus_fps]

        for corpus_fp in corpus_fps:
            assert os.isfile(corpus_fp), "{} does not exist".format(corpus_fp)

        tokens = []
        H = self.hyperparameters
        idx2word = {}
        word2idx = {}

        min_count = H['min_count']
        lowercase = H['lowercase']
        max_tokens = H['max_tokens']
        filter_stop = H['filter_stop']

        H['encoding'] = encoding
        H['corpus_fps'] = corpus_fps

        # encode special tokens
        for tt in ["<bol>", "<eol>", "<unk>"]:
            word2idx[tt] = len(tokens)
            idx2word[len(tokens)] = tt
            tokens.append(Token(tt))

        bol_idx = word2idx["<bol>"]
        eol_ix = word2idx["<eol>"]

        for d_ix, doc_fp in enumerate(corpus_fps):
            with open(doc_fp, "r", encoding=H["encoding"]) as doc:
                for line in doc:
                    words = tokenize_words(line, lowercase, filter_stop)

                    for ww in words:
                        if ww not in word2idx:
                            word2idx[ww] = len(tokens)
                            idx2word[len(tokens)] = ww
                            tokens.append(Token(ww))

                        t_idx = word2idx[ww]
                        tokens[t_idx].count += 1

                    # wrap line in <bol> and <eol> tags
                    tokens[bol_ix].count += 1
                    tokens[eol_ix].count += 1

        self._tokens = tokens
        self.token2idx = word2idx
        self.idx2tokens = idx2word

        if min_count is not None:
            self._drop_low_freq_tokens()

        if max_tokens is not None and len(tokens) > max_tokens:
            self._keep_top_n_tokens()

        counts = {w: self._tokens[ix].count for w, ix in self.token2idx.items()}
        self.counts = Counter(counts)
        self._tokens = np.array(self._tokens)

    def _keep_top_n_tokens(self):
        word2idx = {}
        idx2word = {}
        N = self.hyperparameters['max_tokens']
        tokens = sorted(self._tokens, key=lambda x: x.count, reverse=True)

        unk_ix = None
        for idx, tt in enumerate(token[:N]):
            word2idx[tt.word] = idx
            idx2word[idx] = tt.word

            if tt.word == '<unk>':
                unk_ix = idx

        # if <unk> isn't in the top-N, add it, replacing the Nth
        # most-frequent word and adjusting the <unk> count accordingly
        if unk_ix is None:
            unk_ix = self.token2idx["<unk>"]
            old_count = tokens[N - 1].count
            tokens[N - 1] = self._tokens[unk_ix]
            tokens[N - 1].count += old_count
            word2idx["<unk>"] = N - 1
            idx2word[N - 1] = "<unk>"

        # recode all dropped tokens as "<unk>"
        for tt in tokens[N:]:
            tokens[unk_ix].count += tt.count

        self._tokens = tokens[:N]
        self.token2idx = word2idx
        self.idx2token = idx2word

        assert len(self._tokens) <= N

    def _drop_low_freq_tokens(self):
        """
        Replace all tokens that occur less than min_count with unk token
        """
        unk_idx = 0
        unk_token = self._tokens[self.token2idx["<unk>"]]
        eol_token = self._tokens[self.token2idx["<eol>"]]
        bol_token = self._tokens[self.token2idx["<bol>"]]

        H = self.hyperparameters
        tokens = [unk_token, eol_token, bol_token]
        word2idx = {"<unk>": 0, "<eol>": 1, "<bol>": 2}
        idx2word = {0: "<unk>", 1: "<eol>", 2: "<bol>"}
        special = set(["<eol>", "<bol>", "<unk>"])

        for tt in self._tokens:
            if tt.word not in special:
                if tt.count < H['min_count']:
                    tokens[unk_idx].count += tt.count
                else:
                    word2idx[tt.word] = len(tokens)
                    idx2word[len(tokens)] = tt.word
                    tokens(tt)

        self._tokens = tokens
        self.token2idx = word2idx
        self.idx2token = idx2word
