import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np
from IPython.display import clear_output

from numpy_ml.deep_learning.optimizers import Adam
from numpy_ml.deep_learning.loss_functions import BinaryCrossEntropy
from numpy_ml.deep_learning.layers import Dense, Dropout, Activation, BatchNormalization
from numpy_ml.deep_learning import NeuralNetwork



class GAN():
    """
    A vanilla generative adversarial network

    Training Data: MNIST Handwritten Digits (28x28 images)
    """
    def __init__(self, print_loss=False):
        self.img_rows = 28
        self.img_cols = 28
        self.img_dim = self.img_rows * self.img_cols
        self.latent_dim = 100
        self.print_loss = print_loss

        optimizer = Adam(learning_rate=0.0002, b1=0.5)
        loss_function = BinaryCrossEntropy

        # Build the discriminator
        self.discriminator = self.build_discriminator(optimizer, loss_function)

        # Build the generator
        self.generator = self.build_generator(optimizer, loss_function)

        # Build the combined model
        self.combined = NeuralNetwork(optimizer=optimizer, loss=loss_function)
        self.combined.layers.extend(self.generator.layers)
        self.combined.layers.extend(self.discriminator.layers)

    def build_generator(self, optimizer, loss_function):
        model = NeuralNetwork(optimizer=optimizer, loss=loss_function)
        model.add(Dense(256, input_shape=(self.latent_dim,)))
        model.add(Activation('leaky_relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(Activation('leaky_relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(Activation('leaky_relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.img_dim))
        model.add(Activation('tanh'))

        return model

    def build_discriminator(self, optimizer, loss_function):
        model = NeuralNetwork(optimizer=optimizer, loss=loss_function)

        model.add(Dense(512, input_shape=(self.img_dim,)))
        model.add(Activation('leaky_relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Activation('leaky_relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2))
        model.add(Activation('softmax'))

        return model

    def train(self, n_epochs, batch_size=128, save_interval=50):
        self.n_epochs = n_epochs
        mnist = fetch_openml('mnist_784')
        X = mnist.data
        y = mnist.target

        # Rescale [-1, 1]
        X = (X.astype(np.float32) - 127.5) / 127.5

        half_batch = int(batch_size / 2)
        for epoch in range(self.n_epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.discriminator.set_trainable(True)
            # Select a random half batch of images
            idx = np.random.randint(0, X.shape[0], half_batch)
            imgs = X[idx]

            # Sample noise to use as generator input
            noise = np.random.normal(0, 1, (half_batch, self.latent_dim))

            # Generate a half batch of images
            gen_imgs = self.generator.predict(noise)

            combined_imgs = np.concatenate([imgs,gen_imgs])

            # Valid = [1, 0], Fake = [0, 1]
            valid = np.concatenate((np.ones((half_batch, 1)), np.zeros((half_batch, 1))), axis=1)
            fake = np.concatenate((np.zeros((half_batch, 1)), np.ones((half_batch, 1))), axis=1)

            combined_target = np.concatenate([valid,fake])

            # Train the discriminator
            d_loss = self.discriminator.train_on_batch(combined_imgs, combined_target)
            # d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            # d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            # d_loss = 0.5 * (d_loss_real + d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # We only want to train the generator for the combined model
            self.discriminator.set_trainable(False)

            # Sample noise and use as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # The generator wants the discriminator to label the generated samples as valid
            valid = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))), axis=1)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid)

            # Display the progress
            if self.print_loss:
                print ("epoch:{}, discriminator loss:{}, generator loss:{}".format(epoch, d_loss, g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5 # Grid size
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        # Generate images and reshape to image shape
        gen_imgs = self.generator.predict(noise).reshape((-1, self.img_rows, self.img_cols))

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        plt.suptitle("Generative Adversarial Network")
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        plt.show()
        clear_output(wait=True)
        if epoch == self.n_epochs - 1:
            fig.savefig("mnist_%d.png" % epoch)
        plt.close()


gan = GAN()
gan.train(n_epochs=200000, batch_size=64, save_interval=400)
