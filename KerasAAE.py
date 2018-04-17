from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense
from keras.utils import plot_model
from keras.datasets import mnist
from keras.optimizers import Adam
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
from datetime import datetime
from sklearn.manifold import TSNE


def create_model(input_dim, latent_dim, verbose=False, save_graph=False):

	autoencoder_input = Input(shape=(input_dim,))
	generator_input = Input(shape=(input_dim,))

	encoder = Sequential()
	encoder.add(Dense(1000, input_shape=(input_dim,), activation='relu'))
	encoder.add(Dense(1000, activation='relu'))
	encoder.add(Dense(latent_dim, activation=None))
	
	decoder = Sequential()
	decoder.add(Dense(1000, input_shape=(latent_dim,), activation='relu'))
	decoder.add(Dense(1000, activation='relu'))
	decoder.add(Dense(input_dim, activation='sigmoid'))

	discriminator = Sequential()
	discriminator.add(Dense(1000, input_shape=(latent_dim,), activation='relu'))
	discriminator.add(Dense(1000, activation='relu'))
	discriminator.add(Dense(1, activation='sigmoid'))

	autoencoder = Model(autoencoder_input, decoder(encoder(autoencoder_input)))
	autoencoder.compile(optimizer=Adam(lr=1e-4), loss="mean_squared_error")
	
	discriminator.compile(optimizer=Adam(lr=1e-3), loss="binary_crossentropy")

	discriminator.trainable = False
	generator = Model(generator_input, discriminator(encoder(generator_input)))
	generator.compile(optimizer=Adam(lr=1e-3), loss="binary_crossentropy")

	if verbose:
		print("Autoencoder Architecture")
		print(autoencoder.summary())
		print("Discriminator Architecture")
		print(discriminator.summary())
		print("Generator Architecture")
		print(generator.summary())

	if save_graph:
		plot_model(autoencoder, to_file="autoencoder_graph.png")
		plot_model(discriminator, to_file="discriminator_graph.png")
		plot_model(generator, to_file="generator_graph.png")

	return autoencoder, discriminator, generator, encoder, decoder

def train(n_samples, batch_size, n_epochs):
	global LATENT_DIM

	autoencoder, discriminator, generator, encoder, decoder = create_model(input_dim=784, latent_dim=LATENT_DIM)
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train[:n_samples].reshape(n_samples, 784)
	normalize = colors.Normalize(0., 255.)
	x_train = normalize(x_train)

	x_sample = x_train[42,:].reshape(-1, 784)

	past = datetime.now()
	for epoch in np.arange(1, n_epochs + 1):
		autoencoder_losses = []
		discriminator_losses = []
		generator_losses = []
		for batch in np.arange(len(x_train) / batch_size):
			start = int(batch * batch_size)
			end = int(start + batch_size)
			samples = x_train[start:end]
			autoencoder_history = autoencoder.fit(x=samples, y=samples, epochs=1, batch_size=batch_size, validation_split=0.0, verbose=0)
			fake_latent = encoder.predict(samples)
			discriminator_input = np.concatenate((fake_latent, np.random.randn(batch_size, LATENT_DIM) * 5.))
			discriminator_labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))
			discriminator_history = discriminator.fit(x=discriminator_input, y=discriminator_labels, epochs=1, batch_size=batch_size, validation_split=0.0, verbose=0)
			generator_history = generator.fit(x=samples, y=np.ones((batch_size, 1)), epochs=1, batch_size=batch_size, validation_split=0.0, verbose=0)
			
			autoencoder_losses.append(autoencoder_history.history["loss"])
			discriminator_losses.append(discriminator_history.history["loss"])
			generator_losses.append(generator_history.history["loss"])
		now = datetime.now()
		print("\nEpoch {}/{} - {:.1f}s".format(epoch, n_epochs, (now - past).total_seconds()))
		print("Autoencoder Loss: {}".format(np.mean(autoencoder_losses)))
		print("Discriminator Loss: {}".format(np.mean(discriminator_losses)))
		print("Generator Loss: {}".format(np.mean(generator_losses)))
		past = now

		if epoch % 50 == 0:
			print("\nSaving models...")
			autoencoder.save('autoencoder.h5')
			discriminator.save('discriminator.h5')
			generator.save('generator.h5')
			encoder.save('encoder.h5')
			decoder.save('decoder.h5')

	autoencoder.save('autoencoder.h5')
	discriminator.save('discriminator.h5')
	generator.save('generator.h5')
	encoder.save('encoder.h5')
	decoder.save('decoder.h5')

def reconstruct(n_samples):
	encoder = load_model('encoder.h5')
	decoder = load_model('decoder.h5')
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	choice = np.random.choice(np.arange(n_samples))
	original = x_train[choice].reshape(1, 784)
	normalize = colors.Normalize(0., 255.)
	original = normalize(original)
	latent = encoder.predict(original)
	reconstruction = decoder.predict(latent)
	draw([{"title": "Original", "image": original}, {"title": "Reconstruction", "image": reconstruction}])

def generate(latent=None):
	global LATENT_DIM
	decoder = load_model('decoder.h5')
	if latent is None:
		latent = np.random.randn(1, LATENT_DIM)
	sample = decoder.predict(latent.reshape(1, LATENT_DIM))
	draw([{"title": "Sample", "image": sample}])

def draw(samples):
	fig = plt.figure(figsize=(5 * len(samples), 5))
	gs = gridspec.GridSpec(1, len(samples))
	for i, sample in enumerate(samples):
		ax = plt.Subplot(fig, gs[i])
		ax.imshow((sample["image"] * 255.).reshape(28, 28), cmap='gray')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_aspect('equal')
		ax.set_title(sample["title"])
		fig.add_subplot(ax)
	plt.show(block=False)
	raw_input("Press Enter to Exit")

def plot(n_samples):
	global LATENT_DIM
	encoder = load_model('encoder.h5')
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train[:n_samples].reshape(n_samples, 784)
	y_train = y_train[:n_samples]
	normalize = colors.Normalize(0., 255.)
	x_train = normalize(x_train)
	latent = encoder.predict(x_train)
	if LATENT_DIM > 2:
		tsne = TSNE()
		print("\nFitting t-SNE, this will take awhile...")
		latent = tsne.fit_transform(latent)
	fig, ax = plt.subplots()
	for label in np.arange(10):
		ax.scatter(latent[(y_train == label), 0], latent[(y_train == label), 1], label=label, s=3)
	ax.legend()
	plt.show(block=False)
	raw_input("Press Enter to Exit")


if __name__ == "__main__":
	
	LATENT_DIM = 2

	parser = argparse.ArgumentParser(description="Adversarial Autoencoder implemented with Keras")
	parser.add_argument("--train", action="store_true", help="Train AAE model")
	parser.add_argument("-e", "--epochs", action="store", type=int, default=5000, help="Number of training epochs")
	parser.add_argument("-bs", "--batchsize", action="store", type=int, default=500, help="Batch size for training")
	parser.add_argument("-s", "--samples", action="store", type=int, default=60000, help="Number of samples to use from the MNIST dataset")
	parser.add_argument("--reconstruct", action="store_true", help="Reconstruct a random MNIST image with trained AAE model")
	parser.add_argument("--generate", action="store_true", help="Generate an image with a latent input and the trained AAE model")
	parser.add_argument("-l", "--latent", nargs='+', type=float)
	parser.add_argument("--plot", action="store_true", help="Plot latent space")
	args = parser.parse_args()

	if args.train:
		train(n_samples=args.samples, batch_size=args.batchsize, n_epochs=args.epochs)
	elif args.reconstruct:
		reconstruct(n_samples=args.samples)
	elif args.generate:
		if args.latent:
			assert len(args.latent) == LATENT_DIM, "Latent vector provided is of dim {}; required dim is {}".format(len(args.latent), LATENT_DIM)
			generate(args.latent)
		else:
			generate()
	elif args.plot:
		plot(args.samples)
	else:
		parser.print_help()
