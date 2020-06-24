from numpy_ml.deep_learning.loss_functions import VAELoss

def test_VAE_loss(N=15):
    np.random.seed(12345)

    N = np.inf if N is None else N
    eps = np.finfo(float).eps

    i = 1
    while i < N:
        n_ex = np.random.randint(1, 10)
        t_dim = np.random.randint(2, 10)
        t_mean = random_tensor([n_ex, t_dim], standardize=True)
        t_log_var = np.log(np.abs(random_tensor([n_ex, t_dim], standardize=True) + eps))
        im_cols, im_rows = np.random.randint(2, 40), np.random.randint(2, 40)
        X = np.random.rand(n_ex, im_rows * im_cols)
        X_recon = np.random.rand(n_ex, im_rows * im_cols)

        mine = VAELoss()
        mine_loss = np.mean(mine.loss(X, X_recon, t_mean, t_log_var))

        dX_recon, dMean, dLogVar = mine.gradient(X, X_recon, t_mean, t_log_var)
        golds = TorchVAELoss().extract_grads(X, X_recon, t_mean, t_log_var)

        params = [
            (mine_loss, "loss"),
            (dX_recon, "dX_recon"),
            (dLogVar, "dt_log_var"),
            (dMean, "dt_mean"),
        ]
        print("\nTrial {}".format(i))
        for ix, (mine, label) in enumerate(params):
            np.testing.assert_allclose(
                mine,
                golds[label],
                err_msg=err_fmt(params, golds, ix),
                rtol=1e-5,
                atol=1e-5,
            )
            print("\tPASSED {}".format(label))
        i += 1
test_VAE_loss(10)
