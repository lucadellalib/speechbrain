"""Convolutional autoencoder.

Author
 * Luca Della Libera 2022
"""

import torch
from torch import nn

from speechbrain.nnet.CNN import Conv1d, ConvTranspose1d
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.linear import Linear


__all__ = [
    "AutoEncoder",
]


class AutoEncoder(nn.Module):
    """Convolutional autoencoder model.

    Arguments
    ---------
    input_size : int
        The input size.
    latent_size : int
        The latent space size.

    Examples
    --------
    >>> model = AutoEncoder(8000, 128)
    >>> input = torch.rand(4, 8000, 1)
    >>> output = model(input)

    """

    def __init__(self, input_size, latent_size=128):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.encoder_cnn = Sequential(
            Conv1d(8, 5, in_channels=1, stride=1, padding="valid"),
            nn.ReLU(),
            Conv1d(4, 3, in_channels=8, stride=1, padding="valid"),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten(start_dim=1)
        example_in = torch.ones(1, input_size, 1)
        example_encoder_cnn_out = self.encoder_cnn(example_in)
        example_flatten_out = self.flatten(example_encoder_cnn_out)
        self.encoder_linear = Linear(
            input_size=example_flatten_out.shape[1], n_neurons=latent_size
        )

        self.decoder_linear = Linear(
            input_size=latent_size, n_neurons=example_flatten_out.shape[1]
        )
        self.unflatten = nn.Unflatten(1, example_encoder_cnn_out.shape[1:])
        self.decoder_cnn = Sequential(
            nn.ReLU(),
            ConvTranspose1d(8, 3, in_channels=4, stride=1, padding="valid"),
            nn.ReLU(),
            ConvTranspose1d(1, 5, in_channels=8, stride=1, padding="valid"),
        )

    def forward(self, x):
        while x.ndim < 3:
            x = x.unsqueeze(-1)
        out = torch.nn.functional.pad(
            x, [0, 0, 0, self.input_size - x.shape[-2]]
        )
        out = self.encoder_cnn(out)
        out = self.flatten(out)
        out = self.encoder_linear(out)
        out = self.decoder_linear(out)
        out = self.unflatten(out)
        out = self.decoder_cnn(out)
        return out[:, : x.shape[-2], :]
