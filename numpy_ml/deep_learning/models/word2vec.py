import numpy as np
from ..layers import Embedding
from ..loss_functions import NCELoss
from ...utils.nlp import Vocabulary, tokenize_words


class Word2Vec(object):
    def __init__(self,
                 context_len=5,
                 min_count=5,
                 skip_gram=False,
                 max_tokens=None,
                 embedding_dim=300,
                 filter_stopwords=True,
                 noise_dist_power=0.75,
                 num_negative_samples=64):
        """
        A word2vec model supporting both continuous bag of words (CBOW) and skip-gram
        architectures, with training via noise contrastive estimation

        Parameters:
        ----------------
        context_len: int
            The number of words to use as context during training.
        min_count: int
            The minimum number of times a token must occur in order to be included
            in vocab. If none, all the tokens will be included.
        skip_gram: bool
            If true, skip_gram model will be trained, CBOW otherwise.
        max_tokens: int
            Only add the first max_tokens most frequent tokens that occur more
            than min_count to the vocabulary. If none, all the tokens that are
            more than min_count will be included.
        embedding_dim: int
            The number of dimensions in the final word embeddings.
        filter_stopwords: bool
            Whether to remove stopwords before encoding the word in the corpus.
        noise_dist_power: float between (0, 1)
            The power the unigram count is raised to when computing the noise
            distribution for negative sampling. A value of 0 corresponds to a
            uniform distribution over tokens, and a value of 1 corresponds to
            a distribution proportional to the token unigram counts.
        num_negative_samples: int
            The number of negative samples to draw from the noise distribution
            for each positive training sample.
        """
        def __init__(self):
            self.optimizer = optimizer
            self.skip_gram = skip_gram
            self.min_count = min_count
            self.max_tokens = max_tokens
            self.context_len = context_len
            self.embedding_dim = embedding_dim
            self.filter_stopwords = filter_stopwords
            self.noise_dist_power = noise_dist_power
            self.num_negative_samples = num_negative_samples
            self.special_chars = set(["<unk>", "<eol>", "<bol>"])

        def initialize(self, optimizer):
            self._build_noise_distribution()

            self.embeddings = Embedding(vocab_size=self.vocab_size,
                                        n_out=self.embedding_dim,
                                        reduction=None if self.skip_gram else "mean",
                                        trainable=True)
            self.embeddings.initialize(optimizer)

            self.loss = NCELoss(n_classes=self.vocab_size,
                                n_in = self.embedding_dim,
                                subtract_log_label_prob=False,
                                noise_sampler=self._noise_sampler,
                                num_negative_samples=self.num_negative_samples,
                                trainable=True)
            self.loss.initialize(optimizer)

        def forward(self, X, target):
            """
            Evaluate the network on a single minibatch

            Parameters:
            ----------------
            X: numpy.array of shape (n_ex, n_in)
                The layer input with n_ex samples, and each observation has n_in
                integer word indices.
            targets: numpy.array of shape (n_ex,)
                Target word index for each example in the minibatch.
            """
            X_emb = self.embeddings.forward(X)
            loss, y_pred = self.loss.loss(X_emb, targets.flatten(),train=True)
            return loss, y_pred

        def backward_pass(self):
            dX_emb = self.loss.gradient()
            self.embeddings.backward_pass(dX_emb)

        def _build_noise_distribution():
            """
            Construct the noise distribution for use during negative sampling.
            """
            probs = np.zeros(len(self.vocab))
            for ix, token in enumerate(self.vocab):
                count = token.count
                probs[ix] = count ** self.noise_dist_power

            probs = probs/np.sum(probs)
            self._noise_sampler = DiscreteSampler(probs, log=False, with_replacement=False)

        def fit(self, corpus_path, optimizer=None, encoding="utf-8-sig", n_epochs=20, batch_size=128):
            self.n_epochs = n_epochs
            self.batch_size = batch_size

            self.vocab = Vocabulary(lowercase=True,
                                    min_count=self.min_count,
                                    max_tokens=self.max_tokens,
                                    filter_stopwords=self.filter_stopwords)
            self.vocab.fit(corpus_path, encoding=encoding)
            self.vocab_size = len(self.vocab)

            # ignore special characters when training the model
            for sp in self.special_chars:
                self.vocab.counts[sp] = 0

            # now that we know our vocabulary size, we can initialize the embeddings
            self.initialize(optimizer)

            prev_loss = np.inf
            for i in range(n_epochs):
                this_loss = self._train_epoch(corpus_path, encoding)

        def _train_epoch(self, corpus_path, encoding):
            total_loss = 0
            batch_generator = self.minibatcher(corpus_path, encoding)
            for idx, (X, target) in enumerate(batch_generator):
                this_loss = self._train_batch(X, target)
                total_loss += this_loss

            return total_loss / (ix + 1)


        def minibatcher(self, corpus_path, encoding):
            """
            A minibatch generator for skip-gram and CBOW models

            Parameters:
            -----------------
            corpus_fps : str or List[str]
                The filepath / list of filepaths for the documents to be encoded.
                Each document is expected to be encoded as newline-separated string
                or text, with adjacent tokens separated by a whitespace.
            encoding : str
                Specifies the text encoding for corpus. Common entries are either
                utf-8 or utf-8-sig.

            Returns:
            ----------------
            X: a list of length self.batch_size or numpy.array of shape (self.batch_size, n_in)
               If self.skip_gram is True,

            target: numpy.array of shape (self.batch_size, 1)
                The target IDs associated with each observation in X.
            """
            X_mb = []
            target_mb = []
            mb_ready = False

            for d_idx, doc_file_path in enumerate(corpus_path):
                with open(doc_file_path, "r", encoding=encoding) as doc:
                    for line in doc:
                        words = tokenize_words(line,
                                               lowercase=True,
                                               filter_stopwords=self.filter_stopwords)
                        word_idx = self.vocab.words_to_indices(self.vocab.filter(words,unk=False))
                        for word_loc, this_word in enumerate(word_idx):
                            # The closer a word is closed to this_word, the higher
                            # the probability for the word to be chosen
                            R = np.random.randint(1, self.context_len)
                            left = word_idx[max(word_loc - R, 0): word_loc]
                            right = word_idx[word_loc + 1: word_loc + 1 + R]
                            context = left + right

                            if context == 0:
                                continue

                            if self.skip_gram:
                                # in skip-gram, we use this_word to predict
                                # each of surrounding context
                                X_emb.extend([this_word]*len(context))
                                target_mb.extend(context)
                                mb_ready = len(target_mb) >= self.batch_size
                            else:
                                # In CBOW, we use surrounding context to predict
                                # this_word
                                context = np.array(context)
                                # each sample in X_mb has various length (ragged array)
                                X_mb.append(context)
                                target_mb.extend(word)
                                mb_ready= len(X_mb)==self.batch_size

                            if mb_ready:
                                mb_ready = False
                                X_batch = X_mb.copy()
                                target_batch = target_mb.copy()
                                X_mb = []
                                target_batch = []
                                if self.skip_gram:
                                    X_batch = np.array(X_batch)[:, None]
                                target_batch = np.array(target_batch)[:, None]
                                yield X_batch, target_batch
        if len(X_mb) > 0:
            if self.skip_gram:
                 X_mb = np.array(X_mb)[:, None]
            target_mb = np.array(target_mb)[:, None]
            yield X_mb, target_mb

    def _train_batch(self, X, target):
        loss, _ = self.forward(X, target)
        self.backward_pass()
        return loss
