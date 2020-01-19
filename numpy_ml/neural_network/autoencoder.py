import numpy as np

class Autoencoder():
    """
    An autoencoder with deep fully-connected neural nets.

    Training Data: MNIST Handwritten Digits (28*28 images)
    """
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.img_dim = self.img_rows*self.img_cols
        self.latent_dim = 128

        optimizer = Adam(learning_rate = 0.0002, b1 = 0.5)
        loss_function = SquareLoss

        self.encoder = self.build_encoder(optimizer, loss_function)
        self.decoder = self.build_decoder(optimizer, loss_function)

        self.autoencoder = NeuralNetwork(optimizer=optimizer, loss = loss_function)
        self.autoencoder.layers.extend(self.encoder.layers)
        self.autoencoder.layers.extend(self.decoder.layers)

        self.autoencoder.summary(name='Variational Autoencoder')

    def build_encoder(self,optimizer,loss_function):
        encoder = NeuralNetwork(optimizer=optimizer,loss = loss_function)
        encoder.add(Dense(512, input_shape=(self.img_dim,)))
        encoder.add(Activation('leaky_relu'))
        encoder.add(BatchNormalization(momentum = 0.8))
        encoder.add(Dense(256))
        encoder.add(Activation('leaky_relu'))
        encoder.add(BatchNormalization(momentum=0.8))
        encoder.add(Dense(self.latent_dim))

        return encoder

    def build_decoder(self,optimizer,loss_function):
        decoder = NeuralNetwork(optimizer=optimizer,loss=loss_function)
        decoder.add(Dense(256,input_shape=(self.latent_dim,)))
        decoder.add(Activation('leaky_relu'))
        decoder.add(BatchNormalization(momentum = 0.8))
        decoder.add(Dense(512))
        decoder.add(Activation('leaky_relu'))
        decoder.add(BatchNormalization(momentum=0.8))
        decoder.add(Dense(self.img_dim))
        decoder.add(Activation('tanh'))

        return decoder

    def train(self,n_epochs,batch_size=128):
        mnist = fetch_mldata('MNIST original')
        X = mnist.data
        y = mnist.target

        # Rescale [-1,1]
        X = (X.astype(np.float32) - 127.5)/127.5

        for epoch in range(n_epochs):

            # selecr a random half batch of images
            idx = np.random.randint(0,X.shape[0],batch_size)
            imgs = X[idx]

            # Train autoencoder
            loss, _ = self.autoencoder.train_on_batch(imgs,imgs)

            # Display the progress
            print ("%d [D loss: %f]" % (epoch, loss))


if __name__ = '__main__':
    ae = Autoencoder()
    ae.train(n_epochs=200000, batch_size=64, save_interval=400)
