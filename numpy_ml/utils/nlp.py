import numpy as np
import re
import os
from collections import Counter

_WORD_REGEX = re.compile(r"(?u)\b\w\w+\b")  # sklearn default
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
_STOP_WORDS = {
    "a",
    "about",
    "above",
    "across",
    "after",
    "afterwards",
    "again",
    "against",
    "all",
    "almost",
    "alone",
    "along",
    "already",
    "also",
    "although",
    "always",
    "am",
    "among",
    "amongst",
    "amoungst",
    "amount",
    "an",
    "and",
    "another",
    "any",
    "anyhow",
    "anyone",
    "anything",
    "anyway",
    "anywhere",
    "are",
    "around",
    "as",
    "at",
    "back",
    "be",
    "became",
    "because",
    "become",
    "becomes",
    "becoming",
    "been",
    "before",
    "beforehand",
    "behind",
    "being",
    "below",
    "beside",
    "besides",
    "between",
    "beyond",
    "bill",
    "both",
    "bottom",
    "but",
    "by",
    "call",
    "can",
    "cannot",
    "cant",
    "co",
    "con",
    "could",
    "couldnt",
    "cry",
    "de",
    "describe",
    "detail",
    "do",
    "done",
    "down",
    "due",
    "during",
    "each",
    "eg",
    "eight",
    "either",
    "eleven",
    "else",
    "elsewhere",
    "empty",
    "enough",
    "etc",
    "even",
    "ever",
    "every",
    "everyone",
    "everything",
    "everywhere",
    "except",
    "few",
    "fifteen",
    "fifty",
    "fill",
    "find",
    "fire",
    "first",
    "five",
    "for",
    "former",
    "formerly",
    "forty",
    "found",
    "four",
    "from",
    "front",
    "full",
    "further",
    "get",
    "give",
    "go",
    "had",
    "has",
    "hasnt",
    "have",
    "he",
    "hence",
    "her",
    "here",
    "hereafter",
    "hereby",
    "herein",
    "hereupon",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "however",
    "hundred",
    "i",
    "ie",
    "if",
    "in",
    "inc",
    "indeed",
    "interest",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "keep",
    "last",
    "latter",
    "latterly",
    "least",
    "less",
    "ltd",
    "made",
    "many",
    "may",
    "me",
    "meanwhile",
    "might",
    "mill",
    "mine",
    "more",
    "moreover",
    "most",
    "mostly",
    "move",
    "much",
    "must",
    "my",
    "myself",
    "name",
    "namely",
    "neither",
    "never",
    "nevertheless",
    "next",
    "nine",
    "no",
    "nobody",
    "none",
    "noone",
    "nor",
    "not",
    "nothing",
    "now",
    "nowhere",
    "of",
    "off",
    "often",
    "on",
    "once",
    "one",
    "only",
    "onto",
    "or",
    "other",
    "others",
    "otherwise",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "part",
    "per",
    "perhaps",
    "please",
    "put",
    "rather",
    "re",
    "same",
    "see",
    "seem",
    "seemed",
    "seeming",
    "seems",
    "serious",
    "several",
    "she",
    "should",
    "show",
    "side",
    "since",
    "sincere",
    "six",
    "sixty",
    "so",
    "some",
    "somehow",
    "someone",
    "something",
    "sometime",
    "sometimes",
    "somewhere",
    "still",
    "such",
    "system",
    "take",
    "ten",
    "than",
    "that",
    "the",
    "their",
    "them",
    "themselves",
    "then",
    "thence",
    "there",
    "thereafter",
    "thereby",
    "therefore",
    "therein",
    "thereupon",
    "these",
    "they",
    "thick",
    "thin",
    "third",
    "this",
    "those",
    "though",
    "three",
    "through",
    "throughout",
    "thru",
    "thus",
    "to",
    "together",
    "too",
    "top",
    "toward",
    "towards",
    "twelve",
    "twenty",
    "two",
    "un",
    "under",
    "until",
    "up",
    "upon",
    "us",
    "very",
    "via",
    "was",
    "we",
    "well",
    "were",
    "what",
    "whatever",
    "when",
    "whence",
    "whenever",
    "where",
    "whereafter",
    "whereas",
    "whereby",
    "wherein",
    "whereupon",
    "wherever",
    "whether",
    "which",
    "while",
    "whither",
    "who",
    "whoever",
    "whole",
    "whom",
    "whose",
    "why",
    "will",
    "with",
    "within",
    "without",
    "would",
    "yet",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

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
            'encoding': None,
            'corpus_fps': None,
            'lowercase': lowercase,
            'min_count': min_count,
            'max_tokens': max_count,
            'filter_stopwords': filter_stopwords
        }

    def __len__(self):
        return len(self._tokens)

    @property
    def n_tokens(self):
        """The number of unique word tokens in the vocabulary"""
        return len(self.token2idx)

    @property
    def n_words(self):
        """The total number of words in the corpus"""
        return sum(self.counts.values())

    def list_all_tokens(self):
        return list(self.token2idx.keys())

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
        all_tokens = self.list_all_tokens()
        if unk:
            return [w if w in all_tokens else '<unk>' for w in words]
        return [w for w in words if w in all_tokens]

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
            assert os.path.isfile(corpus_fp), "{} does not exist".format(corpus_fp)

        # tokens stores all the tokens, each token has two features, word and count
        tokens = []
        H = self.hyperparameters
        idx2word = {}
        word2idx = {}

        min_count = H['min_count']
        lowercase = H['lowercase']
        max_tokens = H['max_tokens']
        filter_stop = H['filter_stopwords']

        H['encoding'] = encoding
        H['corpus_fps'] = corpus_fps

        # encode special tokens
        for tt in ["<bol>", "<eol>", "<unk>"]:
            word2idx[tt] = len(tokens)
            idx2word[len(tokens)] = tt
            tokens.append(Token(tt))

        bol_idx = word2idx["<bol>"]
        eol_idx = word2idx["<eol>"]

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
                    tokens[bol_idx].count += 1
                    tokens[eol_idx].count += 1

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

        unk_idx = None
        for idx, tt in enumerate(token[:N]):
            word2idx[tt.word] = idx
            idx2word[idx] = tt.word

            if tt.word == '<unk>':
                unk_idx = idx

        # if <unk> isn't in the top-N, add it, replace the Nth
        # most-frequent word with <unk> and adjust the <unk> count accordingly
        if unk_idx is None:
            unk_idx = self.token2idx["<unk>"]
            old_count = tokens[N - 1].count
            tokens[N - 1] = self._tokens[unk_idx]
            tokens[N - 1].count += old_count
            word2idx["<unk>"] = N - 1
            idx2word[N - 1] = "<unk>"

        # recode all dropped tokens as "<unk>"
        for tt in tokens[N:]:
            tokens[unk_idx].count += tt.count

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
                    # all the tokens which occur less than min_count will be
                    # considered as unknown
                    tokens[unk_idx].count += tt.count
                else:
                    word2idx[tt.word] = len(tokens)
                    idx2word[len(tokens)] = tt.word
                    tokens.append(tt)

        self._tokens = tokens
        self.token2idx = word2idx
        self.idx2token = idx2word
