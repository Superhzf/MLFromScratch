import numpy as np
from ..utils import Vocabulary,tokenize_words

class Word2Vec:
    def __init__(self,
                 context_len=5,
                 min_count=None,
                 skip_gram=False,
                 max_tokens=None,
                 embedding_dim=300,
                 filter_stopwords=True,
                 noise_dist_power=0.75,
                 num_negative_samples=64,
                 optimizer=None):
        """
        A word2vec model supporting both continuous bag of words (CBOW) and
        skip-gram architectures.

        Parameters:
        ------------------
        context_len: int
            The number of words to the left and right of the current word to use
            as context during training. Larger Larger values result in more
            training examples and thus can lead to higher accuracy at the expense
            of additional training time.
        min_count: int ot None
            Minimum number of times a token must occur in order to be included in
            vocabulary. If none, include all tokens from 'corpus_fp' in vocab.
        skip_gram: bool
            Whether to train the skip-gram or CBOW model. The skip-gram model
            is trained to predict the target word i given its surrounding context,
            words[i-context:i] and words[i+1:i+1+context] as input.
        max_tokens: int or None
            The maximum number of occurance for a word in order to be added into
            vocabulary. If none, all tokens that occur more than min_count will
            be added
        embedding_dim: int
            The number of dimensions in the final word embeddings
        filter_stopwords: bool
            Whether to remove stop words before encoding the words in the corpus.
        noise_dist_power: float
            The power the unigram count is raised to when computing the noise
            distribution for negative sampling. A value of 0 corresponds to a uniform
            distribution over tokens, and a value of 1 corresponds to a distribution
            proportional to the token unigram counts.
        num_negative_samples: int
            The number of negative samles to draw from the noise distribution for
            each positive training sample.
        optimizer: numpy_ml.neural_nets.optimizers object
            The optimization strategy
        """
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

    def initialize(self):
        self._dv = {}
        self._build_noise_distribution()
        self.embeddings = Embedding()
        self.loss = NCELoss()

    def _build_noise_distribution(self):
        """
        Construct the noise distribution for use during negative sampling.

        For a word w in the corpus, the noise distribution is:
            P_n(w) = Count(w) ** noise_dist_power / Z
        where Z is a normalizing constant and noise_dist_power is a hyperparameter
        """
        probs = np.zeros(len(self.vocab))
        power = self.noise_dist_power

        for idx, token in enumerate(self.vocab):
            count = token.count
            probs[idx] = count ** power

        probs = probs/np.sum(probs)
        self._noise_sampler = DiscreteSampler(probs, log=False, with_replacement=False)

    def minibatcher(self, corpus_fps, encoding):
        """
        A minibatch generator for skip-gram and CBOW models.

        Parameters:
        --------------------
        corpus_fps: str or List[str]
            The filepath / list of filepaths to the document(s) to be encoded.
            Each document is expected to be encoded as newline-separated
            string of text, with adjacent tokens separated by a whitespace

        encoding: str
            Specifies the text encoding for corpus.

        Returns:
        -------------------
        X: List of length of batchsize or NumPy ndarray of shape (batchsize, n_in)

        target: NumPy ndarray of shape (batchsize, 1)
            The target IDs associated with each example in X
        """
        batchsize = self.batchsize
        X_mb = []
        target_mb = []
        mb_ready = False
        for d_ix, doc_fp in enumerate(corpus_fps):
            with open(doc_fp, "r", encoding=encoding) as doc:
                for line in doc:
                    words = tokenize_words(line, lowercase=True, filter_stopwords=self.filter_stopwords)
                    word_ixs = self.vocab.words_to_indices(self.vocab.filter(words, unk=False))

                    for word_loc, word in enumerate(word_ixs):
                        # since more distant words are usually less related to
                        # the target word, we downweight them by sampling from
                        # them less frequently during training.
                        R = np.random.randint(1, self.context_len)
                        left = word_ixs[max(word_loc - R, 0) : word_loc]
                        right = word_ixs[word_loc + 1 : word_loc + 1 + R]
                        context = left + right

                        if len(context) == 0:
                            continue

                        # in the skip-gram architecture we use each of the
                        # surrounding context to predict `word` / avoid
                        # predicting negative samples
                        if self.skip_gram:
                            X_mb.extend([word] * len(context))
                            target_mb.extend(context)
                            mb_ready = len(target_mb) >= batchsize
                        # in the CBOW architecture we use the average of the
                        # context embeddings to predict the target `word` / avoid
                        # predicting the negative samples
                        else:
                            context = np.array(context)
                            X_mb.append(context)  # X_mb will be a ragged array
                            target_mb.append(word)
                            mb_ready = len(X_mb) == batchsize

        if len(X_mb) > 0:
            if self.skip_gram:
                X_mb = np.array(X_mb)[:, None]
            target_mb = np.array(target_mb)[:, None]
            yield X_mb, target_mb

    def _train_batch(self, X, target):
        loss, _ = self.forward(X, target)
        self.backward()
        self.update(loss)
        return loss

    def forward(sel, X, targets, retain_derived=True):
        """
        Evaluate the network on a single minibatch.

        Parameters
        ------------------------
        X: numpy ndarray of shape (n_ex, n_in)
            Layer input, representing a minibatch of n_ex examples, each
            consisting of n_in integer word indices
        targets: numpy ndarray of shape (n_ex, )
            Target word index for each example in the minibatch.
        retain_derived: bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through w.r.t. this input.

        Returns
        ----------------------
        loss: float
            The loss associated with the current minibatch
        y_pred: numpy ndarray of shape (n_ex,)
            The conditional probabilities of the words in targets given the
            corresponding example / context in X.
        """
        X_emb = self.embeddings.forward(X, retain_derived=True)
        loss, y_pred = self.loss.loss(X_emb, targets.flatten(), retain_derived=True)
        return loss, y_pred

    def backward(self):
        """
        Compute the gradient of the loss wrt the current network parameters.
        """
        dX_emb = self.loss.grad(retain_grads=True, update_params=False)
        self.embeddings.backward(dX_emb)

    def _train_epoch(self, corpus_fps, encoding):
        total_loss = 0
        batch_generator = self.minibatcher(corpus_fps, encoding)
        for ix, (X, target) in enumerate(batch_generator):
            loss = self._train_batch(X, target)
            total_loss += loss
            if self.verbose:
                smooth_loss = 0.99 * smooth_loss + 0.01 * loss if ix > 0 else loss
                fstr = "[Batch {}] Loss: {:.5f} | Smoothed Loss: {:.5f}"
                print(fstr.format(ix + 1, loss, smooth_loss))
        return total_loss / (ix + 1)

    def fit(self, corpus_fps, encoding='utf-8-sig', n_epochs=20, batchsize=128, verbose=True):
        """
        Learn word2vec embeddings for the examples in X_train

        Parameters:
        -------------------------
        corpus_fps: str or a list of strs
            The filepath / list of filepaths to the documents to be encoded.
            Each document is expected to be encoded as newline-separated string
            of text, with adjacent tokens separated by a whitespace.
        encoding: str
            Specifies the text encoding for corpus. Common entries are either
            utf-8 or utf-8-sig.
        n_epochs: int
            The maximum number of training epochs to run
        batchsize: int
            The desired number of examples in each training batch.
        verbose: bool
            Print batch information during training.
        """
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.batchsize = batchsize

        self.vocab = Vocabulary(
            lowercase = True,
            min_count=self.min_count,
            max_tokens=self.max_tokens,
            filter_stopwords=self.filter_stopwords,
        )
        self.vocab.fit(corpus_fps, encoding=encoding)
        self.vocab_size = len(self.vocab)

        # ignore special characters when training the model
        for sp in self.special_chars:
            self.vocab.counts[sp] = 0

        self.initialize()

        prev_loss = np.inf
        for i in range(n_epochs):
            loss = 0
            loss = self._train_epoch(corpus_fps, encoding)
            prev_loss = loss



class DiscreteSampler:
    def __init__(self, probs, log=False, with_replacement=True):
        """
        Sample from an abtribary multinominal PMF over the first N nonnegative
        integers using Vose's algorithm.

        Parameters
        ------------------
        probs: numpy.array of length N
            A list of probabilities of the N outcomes in the sample space.
            probs[i] returns the probability of outcome i
        log: bool
            Whether the probabilities in probs are in logspace.
        with_replacement: bool
            Whether to generate samples with replacement.
        """
        if not isinstance(probs, np.ndarray):
            raise Exception('The type of probs is not understood')

        self.log = log
        self.N = len(probs)
        self.probs = probs
        self.with_replacement = with_replacement

        alias = np.zeros(self.N)
        prob = np.zeros(self.N)
        scaled_probs = self.probs + np.log(self.N) if log else self.probs*self.N

        selector = scaled_probs < 0 if log else scaled_probs < 1
        small, large = np.where(selector)[0].tolist(), np.where(~selector)[0].tolist()
        while len(small) and len(large):
            l = small.pop()
            g = large.pop()

            alias[l] = g
            prob[l] = scaled_probs[l]

            if log:
                pg = np.log(np.exp(scaled_probs[g]) + np.exp(scaled_probs[l]) - 1)
            else:
                pg = scaled_probs[g] + scaled_probs[l] - 1

            scaled_probs[g] = pg
            to_small = pg < 0 if log else pg < 1
            if to_small:
                small.append(g)
            else:
                large.append(g)

        while len(large):
            prob[large.pop()] = 0 if log else 1

        while len(small):
            prob[small.pop()] = 0 if log else 1

        self.prob_table = prob
        self.alias_table = alias

    def __call__(self, n_samples=1):
        """
        Generate random draws from the probs distribution over integers in
        [0, N)

        Parameters:
        --------------------
        n_samples: int
            The number of samples to generate.

        Returns:
        -------------------
        sample: np.ndarray of shape (n_samples,)
            A collection of draws from the distribution defined by probs.
            Each sample is an int in the range [0, N)
        """
        return self.sample(n_samples)

    def sample(self, n_samples=1):
        """
        Generate random draws from the probs distribution over integers in
        [0, N)

        Parameters:
        --------------------
        n_samples: int
            The number of samples to generate

        Returns
        -----------------
        sample: np.ndarray of shape (n_samples, )
            A collection of draws from the distribution defined by probs.
            Each sample is an int in the range [0, N)
        """
        ixs = np.random.randint(0, self.N, n_samples)
        p = np.exp(self.prob_table[ixs]) if self.log else self.prob_table[ixs]
        flips = np.random.binomial(1, p)
        samples = [ix if f else self.alias_table[ix] for ix, f in zip(ixs, flips)]

        if not self.with_replacement:
            unique = list(set(samples))

            while len(samples) != len(unique):
                n_new = len(samples) - len(unique)
                samples = unique + self.sample(n_new).tolist()
                unique = list(set(samples))
        return np.array(samples, dtype=int)
