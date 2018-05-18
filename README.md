# keras-aae

Reproduces Adversarial Autoencoder architecture from [Makhzani, Alireza, et al. "Adversarial autoencoders." arXiv preprint arXiv:1511.05644 (2015)](https://arxiv.org/abs/1511.05644) with Keras.

## Summary

The Adversarial Autoencoder behaves similarly to [Variational Autoencoders](https://arxiv.org/abs/1312.6114), forcing the latent space of an autoencoder to follow a predefined prior. In the case of the Adversarial Autoencoder, this latent space can be defined arbitrarily and easily sampled and fed into the Discriminator in the network.

<div>
<img src="https://raw.githubusercontent.com/greentfrapp/keras-aae/master/images/aae_latent.png" alt="Latent space from Adversarial Autoencoder" width="whatever" height="300px" style="display: inline-block;">
<img src="https://raw.githubusercontent.com/greentfrapp/keras-aae/master/images/regular_latent.png" alt="Latent space from regular Autoencoder" width="whatever" height="300px" style="display: inline-block;">
</div>

*The left image shows the latent space of an unseen MNIST test set after training with an Adversarial Autoencoder for 50 epochs, which follows a 2D Gaussian prior. Contrast this with the latent space of the regular Autoencoder trained under the same conditions, with a far more irregular latent distribution.*

## Instructions

To train a model just run

```
$ python keras-aae.py --train
```

For more parameters, run with `--help` flag.

For comparison with a regular autoencoder, run

```
$ python regular-ae.py --train --noadversarial
```