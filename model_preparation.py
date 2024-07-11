import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D
import tensorflow_addons as tfa
from keras.losses import BinaryCrossentropy
from keras.models import Model
from keras.callbacks import Callback

from Data_Load import data_load

def build_generator():
    model = Sequential([
        # Takes in random values and reshapes it to 7x7x128
        Dense(7*7*128, input_dim=128), LeakyReLU(0.2), Reshape((7, 7, 128)),
        # For upsampling block 1
        UpSampling2D(), Conv2D(128, 5, padding='same'), LeakyReLU(0.2),
        # Conv layer 1
        Conv2D(128, 5, padding='same'), LeakyReLU(0.2),
        # Conv layer 2
        Conv2D(128, 5, padding='same'), LeakyReLU(0.2),
        # For upsampling block 2
        UpSampling2D(), Conv2D(128, 4, padding='same'), LeakyReLU(0.2),
        # Conv layer 3
        Conv2D(128, 4, padding='same'), LeakyReLU(0.2),
        # Conv layer 4
        Conv2D(128, 5, padding='same'), LeakyReLU(0.2),
        # Final Conv layer
        Conv2D(1, 4, padding='same', activation='sigmoid')
    ])
    return model
    
def build_discriminator():
    model = Sequential([
        Conv2D(64, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding='same'), LeakyReLU(alpha=0.2),
        Conv2D(128, kernel_size=3, strides=2, padding='same'), LeakyReLU(alpha=0.2),
        Conv2D(256, kernel_size=3, strides=2, padding='same'), LeakyReLU(alpha=0.2),
        Conv2D(512, kernel_size=3, strides=2, padding='same'), LeakyReLU(alpha=0.2),
        Flatten(), Dense(1, activation='sigmoid')
    ])
    return model

def print_img(model):
    img = model.predict(np.random.randn(4,128,1))
    # Setup the subplot formatting
    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    # Loop four times and get images
    for idx, img in enumerate(img):
        # Plot the image using a specific subplot
        ax[idx].imshow(np.squeeze(img))
        # Appending the image label as the plot title
        ax[idx].title.set_text(idx)

class ModelMonitor(Callback):
    def __init__(self, num_img=3, latent_dim=128, save_freq=5):
        super().__init__()
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.save_freq = save_freq  # Save every `save_freq` epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == 0:  # Save images every `save_freq` epochs
            random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim, 1))
            generated_images = self.model.generator(random_latent_vectors)
            generated_images *= 255
            generated_images = generated_images.numpy()
            if not os.path.exists('images'):
                os.makedirs('images')
            for i in range(self.num_img):
                img = np.array_to_img(generated_images[i])
                img.save(os.path.join('images', f'generated_img_{epoch}_{i}.png'))

class GAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generator = generator
        self.discriminator = discriminator
    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        super().compile(*args, **kwargs)

        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss
    def train_step(self, batch):
        real_img = batch
        fake_img = self.generator(tf.random.normal((6,128,1)), training=False)

        #train the discriminator
        with tf.GradientTape() as d_tape:
            #pass the real and fake images
            yhat_real = self.discriminator(real_img, training=True)
            yhat_fake = self.discriminator(fake_img, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            #create labels for real and fake images
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_real)], axis=0)

            #add some noise to the true outputs
            noise_real= 0.15*tf.random.uniform(tf.shape(yhat_real))
            noise_fake= -0.15*tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            #calculate loss
            total_d_loss = self.d_loss(yhat_realfake, yhat_realfake)

        #apply backpropagation
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        #train the generator
        with tf.GradientTape() as g_tape:
            #generate new images
            gen_img = self.generator(tf.random.normal((128,128,1)), training=True)
            #create predicted labels
            predicted_labels = self.discriminator(gen_img, training=False)

            #calculate loss
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        #apply backpropagation
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {"d_loss":total_d_loss, "g_loss": total_g_loss}

if __name__ == '__main__':
    # Optimizers and Losses for the Generator and the Discriminator
    g_opt = tfa.optimizers.AdamW(learning_rate=6e-7, weight_decay=5e-4, epsilon=1e-7)
    d_opt = tfa.optimizers.AdamW(learning_rate=6e-7, weight_decay=5e-4, epsilon=1e-7)
    g_loss = BinaryCrossentropy()
    d_loss = BinaryCrossentropy()
    
    # Data Preparation
    ds = data_load()
    data, keys = ds.download_data()
    data = data.map(ds.scale_data)
    data = data.cache()
    data = data.shuffle(60000)
    data = data.batch(128)
    data = data.prefetch(32)

    # Calling the generator and the discriminator
    generator = build_generator()
    discriminator = build_discriminator()

    # Creating an instance for GAN model for training
    gan = GAN(generator=generator,discriminator=discriminator)
    gan.compile(g_opt=g_opt, g_loss=g_loss, d_opt=d_opt, d_loss=d_loss)

    # Example usage with your GAN model
    gan = GAN(generator, discriminator)
    gan.compile(g_opt, d_opt, g_loss, d_loss)
    monitor = ModelMonitor(num_img=3, latent_dim=128, save_freq=1)
    history = gan.fit(data, epochs=1)

    plt.suptitle('loss')
    plt.plot(history.history['d_loss'], label='d_loss')
    plt.plot(history.history['g_loss'], label='g_loss')
    plt.legend()
    plt.show()

    generator.save('generator.h5')
    discriminator.save('discriminator.h5')
    
    imgs = generator.predict(np.random.randn(16,128,1))

    fig, ax = plt.subplots(ncols=4, figsize=(10,10))
    for r in range(4):
        ax[r].imshow(imgs[r])