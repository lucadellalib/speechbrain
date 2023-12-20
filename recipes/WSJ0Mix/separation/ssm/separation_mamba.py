"""Mamba-based separation models.

Install the required packages:
pip install causal-conv1d<=1.0.2
pip install mamba-ssm

Authors
 * Luca Della Libera
"""

import torch
from mamba_ssm import Mamba
from torch import nn


__all__ = ["MambaNet", "MambaMaskNet"]


class MambaNet(nn.Module):
    def __init__(
        self,
        d_input,
        n_layers=1,
        d_state=64,
        d_conv=4,
        expand=2,
        **kwargs,
    ):
        super().__init__()
        self.d_input = d_input
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.layers = nn.ModuleList(
            [
                Mamba(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=d_input,  # Model dimension d_model
                    d_state=d_state,  # SSM state expansion factor
                    d_conv=d_conv,  # Local convolution width
                    expand=expand,  # Block expansion factor
                    **kwargs
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, input, *args, **kwargs):
        # input: B x L x C
        output = input
        for layer in self.layers:
            output = layer(output, *args, **kwargs)
        return output


class MambaMUNet(nn.Module):
    def __init__(
        self,
        d_input,
        num_spks=2,
        n_layers=1,
        d_state=64,
        d_conv=4,
        expand=2,
        output_activation=torch.nn.ReLU(),
        **kwargs,
    ):
        super().__init__()
        self.d_input = d_input
        self.num_spks = num_spks
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.layers = nn.ModuleList(
            [
                Mamba(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=d_input,  # Model dimension d_model
                    d_state=d_state,  # SSM state expansion factor
                    d_conv=d_conv,  # Local convolution width
                    expand=expand,  # Block expansion factor
                    **kwargs
                )
                for _ in range(n_layers - 1)
            ]
        )
        self.output_linear = nn.Linear(d_input, num_spks * d_input)
        self.output_activation = output_activation

    def forward(self, input, *args, **kwargs):
        # input: B x C x L
        output = input.movedim(-1, -2)
        for layer in self.layers:
            output = layer(output, *args, **kwargs)
        output = self.output_linear(output)
        output = self.output_activation(output)
        B, L, C = output.shape
        output = (
            output.movedim(-1, 0)
            .reshape(self.num_spks, -1, B, L)
            .movedim(1, -2)
        )
        return output


class MambaMaskNet(nn.Module):
    def __init__(
        self,
        d_input,
        num_spks=2,
        n_layers=1,
        d_state=64,
        d_conv=4,
        expand=2,
        output_activation=torch.nn.ReLU(),
        **kwargs,
    ):
        super().__init__()
        self.d_input = d_input
        self.num_spks = num_spks
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.layers = nn.ModuleList(
            [
                Mamba(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=d_input,  # Model dimension d_model
                    d_state=d_state,  # SSM state expansion factor
                    d_conv=d_conv,  # Local convolution width
                    expand=expand,  # Block expansion factor
                    **kwargs
                )
                for _ in range(n_layers - 1)
            ]
        )
        self.output_linear = nn.Linear(d_input, num_spks * d_input)
        self.output_activation = output_activation

    def forward(self, input, *args, **kwargs):
        # input: B x C x L
        output = input.movedim(-1, -2)
        for layer in self.layers:
            output = layer(output, *args, **kwargs)
        output = self.output_linear(output)
        output = self.output_activation(output)
        B, L, C = output.shape
        output = (
            output.movedim(-1, 0)
            .reshape(self.num_spks, -1, B, L)
            .movedim(1, -2)
        )
        return output


def test_mamba_net():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 4
    sequence_length = 10246
    d_input = 50
    lr = 0.1

    model = MambaNet(d_input, n_layers=4).to(device)
    torch.manual_seed(0)

    # Create an optimizer with the general parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    input = torch.rand(batch_size, sequence_length, d_input, device=device)
    output = model(input)
    loss = output.sum()
    loss.backward()
    print(output)
    optimizer.step()


def test_mamba_mask_net():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 4
    sequence_length = 10246
    d_input = 50
    lr = 0.1

    model = MambaMaskNet(d_input, num_spks=2, n_layers=4).to(device)
    torch.manual_seed(0)

    # Create an optimizer with the general parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    input = torch.rand(batch_size, d_input, sequence_length, device=device)
    output = model(input)
    loss = output.sum()
    loss.backward()
    print(output)
    optimizer.step()


if __name__ == "__main__":
    test_mamba_net()
    test_mamba_mask_net()
