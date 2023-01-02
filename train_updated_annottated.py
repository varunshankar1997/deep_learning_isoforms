import os
import sys
import keras
import numpy as np
import pandas as pd
from keras.layers import Dense
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from keras.models import Model, load_model

import keras
from keras import layers
from keras import backend as K
from sklearn.model_selection import train_test_split


Path_Isoforms = '/dtu-compute/datasets/iso_02456/gtex_isoform_expression_transposed.tsv'

Path = '/dtu-compute/datasets/iso_02456/gtex_gene_expression_transposed.tsv' #CD to the folder -> pwd -> copy paste
latent_dim = 64
intermediate_dim = 128
epochz= 200
batch_size = 200
# df_iso = pd.read_csv('/zhome/bf/7/164671/gtex_isoform_expression_subset.tsv', sep ="\t").T


#applying annotations to the dataset
dict ={}
dict ={}
with open('/dtu-compute/datasets/iso_02456/gtex_annot.tsv','r') as infile:
        for line in infile:
                key=line.split()[0]
                value=line.split()[1]

                dict[key] = value


print("started")
def load_data(DATA_PATH):
    """Loads the data and preprocesses it."""
    row_names = []
    array = []
    with open(DATA_PATH, 'r', encoding='utf-8') as infile:
        next(infile)
        for line in infile:
            line = line.split("\t")
            row_names.append(line[0])
            array.append(line[1:])

    expr = np.asarray(array, dtype=np.float32)
    expr = np.log2(1+expr[:])
    expr = expr / np.max(expr, axis=1, keepdims=True)
    expr = np.nan_to_num(expr)
    return expr, row_names


x, row_names = load_data(Path)
original_dim = x.shape[1]

x_iso, row_names_iso = load_data(Path_Isoforms)
original_dim_iso = x.shape[1]

def train(autoencoder, data , epochs=epochz):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for x in data:
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            opt.step()
    return autoencoder


def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon


def vae_loss(x, x_decoded_mean):

  xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
  kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
  return xent_loss + kl_loss

inputs = keras.Input(shape=(original_dim,))
outputs = layers.Dense(original_dim)(x) # No activation
opt = Adam(learning_rate=0.0001, clipnorm=0.001)


class VAE(keras.Model):
    def __init__(self, original_dim=original_dim, epochs = 10, intermediate_dim = 100, latent_dim = 100):
        super(VAE, self).__init__()
        inputs = keras.Input(shape=(original_dim,))
        h = layers.Dense(intermediate_dim)(inputs) # delete act

        z_mean = layers.Dense(latent_dim)(h)
        z_log_sigma = layers.Dense(latent_dim)(h)
        z = layers.Lambda(sampling)([z_mean, z_log_sigma])

        # Create encoder
        encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
        self.encoder = encoder
        # Create decoder
        latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
        x = layers.Dense(intermediate_dim)(latent_inputs) #relu


        outputs = layers.Dense(original_dim)(x) #delete act
        decoder = keras.Model(latent_inputs, outputs, name='decoder')
        self.decoder = decoder

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = keras.Model(inputs, outputs, name='vae_mlp')

        # loss
        reconstruction_loss = keras.losses.mean_squared_error(inputs, outputs) #mean and a varriance
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(0.9*(reconstruction_loss) + 0.1*(kl_loss))
        vae.add_loss(vae_loss)

        opt = Adam(learning_rate=0.001 ,clipnorm=0.001)

        vae.compile(optimizer= opt, loss='mean_squared_error', metrics=['accuracy'])
        vae.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, x_test))
        vae.save("vae_pretrained")# RUN THE SCRIPT FROM /HOME SO THE PRETRAINED WOULD BE SAVED IN THE DIRECTORY THAT YOU CAN WRITE!


x_val_iso = x_val_iso = np.array(x_iso)
x_train = x_test = np.array(x)
vae = VAE(intermediate_dim = intermediate_dim, latent_dim = latent_dim, epochs=epochz)


x_test_encoded = np.array(vae.encoder.predict(x_test , batch_size=batch_size))
x_val_encoded = np.array(vae.encoder.predict(x_val_iso , batch_size=batch_size))



tolerance = 0.1
accuracy = (np.abs(x_test_encoded - x_val_encoded) < tolerance ).mean()
print(accuracy)
print("ended")
