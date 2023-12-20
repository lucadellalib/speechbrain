"""S4-based separation models.

Install the required CUDA extensions:
pip install git+https://github.com/HazyResearch/state-spaces.git@d589d982216485cce0a46bbe7605fe75c03d3223#subdirectory=extensions/cauchy

Authors
 * Luca Della Libera
"""

import math

import torch
#from s5 import S5Block
from torch import nn

try:
    from .s4 import Activation, S4
    from .sashimi import FFBlock, LinearActivation, ResidualBlock, Sashimi
except ImportError:
    from s4 import Activation, S4
    from sashimi import FFBlock, LinearActivation, ResidualBlock, Sashimi


__all__ = ["S4MaskNet", "S4Net", "SashimiMaskNet"]


class S4Block(nn.Module):
    def __init__(self, d_input, d_output, d_state=64, mode="nplr", **kwargs):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_input)
        self.s4 = S4(
            d_input,
            d_state,
            transposed=False,
            activation="gelu",
            postact="id",
            mode=mode,
            **kwargs,
        )
        self.linear1 = nn.Linear(d_input, d_input)
        self.layer_norm2 = nn.LayerNorm(d_input)
        self.linear = nn.Linear(d_input, d_input)
        self.gelu = nn.GELU()
        self.projection = nn.Linear(d_input, d_output)
        self.output_linear = nn.Linear(d_input, d_output)

    def forward(self, input, *args, **kwargs):
        # B x L x S
        output1 = self.layer_norm1(input)
        output1, state = self.s4(output1, *args, **kwargs)
        output1 = self.linear1(output1)
        output1 += input
        output2 = self.layer_norm2(output1)
        output2 = self.linear(output2)
        output2 = self.gelu(output2)
        output2 = self.output_linear(output2)
        output2 += self.projection(output1)
        return output2, state


class S4Net(nn.Module):
    def __init__(
        self,
        d_input,
        n_layers=1,
        d_state=64,
        mode="nplr",
        **kwargs,
    ):
        super().__init__()
        self.d_input = d_input
        self.n_layers = n_layers
        self.d_state = d_state
        self.layers = nn.ModuleList(
            [
                S4Block(d_input, d_input, d_state, mode=mode, **kwargs)
                for _ in range(n_layers)
            ]
        )

    def forward(self, input, *args, **kwargs):
        # input: B x L x C
        output = input
        for layer in self.layers:
            output, _ = layer(output, *args, **kwargs)
        return output


class S4MaskNet(nn.Module):
    def __init__(
        self,
        d_input,
        num_spks=2,
        n_layers=1,
        d_state=64,
        output_activation="relu",
        mode="nplr",
        **kwargs,
    ):
        super().__init__()
        self.d_input = d_input
        self.num_spks = num_spks
        self.n_layers = n_layers
        self.d_state = d_state
        self.output_activation = Activation(output_activation)
        self.layers = nn.ModuleList(
            [
                S4Block(d_input, d_input, d_state, mode=mode, **kwargs)
                for _ in range(n_layers - 1)
            ]
        )
        self.layers.append(
            S4Block(d_input, num_spks * d_input, d_state, mode=mode, **kwargs)
        )

    def forward(self, input, *args, **kwargs):
        # input: B x C x L
        output = input.movedim(-1, -2)
        for layer in self.layers:
            output, _ = layer(output, *args, **kwargs)
        output = self.output_activation(output)
        B, L, C = output.shape
        output = (
            output.movedim(-1, 0)
            .reshape(self.num_spks, -1, B, L)
            .movedim(1, -2)
        )
        return output


"""
class S5MaskNet(nn.Module):
    def __init__(
        self,
        d_input,
        num_spks=2,
        n_layers=1,
        d_state=64,
        block_count=8,
        bidirectional=False,
        output_activation="relu",
        **kwargs,
    ):
        super().__init__()
        self.d_input = d_input
        self.num_spks = num_spks
        self.n_layers = n_layers
        self.d_state = d_state
        self.output_activation = Activation(output_activation)
        self.layers = nn.ModuleList(
            [
                S5Block(d_input, d_state, block_count=block_count, bidir=bidirectional, **kwargs)
                for _ in range(n_layers - 1)
            ]
        )
        self.layers.append(
            S5Block(d_input, d_state, block_count=block_count, bidir=bidirectional, **kwargs)
        )

    def forward(self, input, *args, **kwargs):
        # input: B x C x L
        output = input.movedim(-1, -2)
        for layer in self.layers:
            output = layer(output, *args, **kwargs)
        output = self.output_activation(output)
        B, L, C = output.shape
        output = (
            output.movedim(-1, 0)
            .reshape(self.num_spks, -1, B, L)
            .movedim(1, -2)
        )
        return output


class S5Net(nn.Module):
    def __init__(
        self,
        d_input,
        n_layers=1,
        d_state=64,
        block_count=8,
        bidirectional=False,
        **kwargs,
    ):
        super().__init__()
        self.d_input = d_input
        self.n_layers = n_layers
        self.d_state = d_state
        self.layers = nn.ModuleList(
            [
                S5Block(d_input, d_state, block_count=block_count, bidir=bidirectional, **kwargs)
                for _ in range(n_layers)
            ]
        )

    def forward(self, input, *args, **kwargs):
        # input: B x L x C
        output = input
        for layer in self.layers:
            output = layer(output, *args, **kwargs)
        return output
"""


class SashimiMaskNet(Sashimi):
    def __init__(
        self,
        d_input,
        num_spks=2,
        pool=(4, 4),
        output_activation="relu",
        mode="nplr",
        **kwargs,
    ):
        super().__init__(pool=pool, d_model=d_input, mode=mode, **kwargs)
        self.num_spks = num_spks
        self.pool = pool
        self.projection = LinearActivation(
            self.d_output,
            num_spks * self.d_output,
            transposed=False,
            activation=output_activation,
            activate=True,
        )

    def forward(self, input, **kwargs):
        # B x C x L
        output = input
        pool_factor = math.prod(self.pool)
        pad = None
        if output.shape[-1] % pool_factor != 0:
            pad = [
                0,
                math.ceil(output.shape[-1] / pool_factor) * pool_factor
                - output.shape[-1],
            ]
            output = torch.nn.functional.pad(output, pad, value=0)
        # B x L x C
        output = output.movedim(-1, -2)
        output, _ = super().forward(output)
        if pad is not None:
            output = output[:, : -pad[1], :]
        output = self.projection(output)
        B, L, C = output.shape
        output = (
            output.movedim(-1, 0)
            .reshape(self.num_spks, -1, B, L)
            .movedim(1, -2)
        )
        return output


def test_s4_net():
    batch_size = 4
    sequence_length = 10246
    d_input = 50
    lr = 0.1

    model = S4Net(d_input, n_layers=4)
    torch.manual_seed(0)

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s)
        for s in sorted(
            list(dict.fromkeys(frozenset(hp.items()) for hp in hps))
        )
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})
    print(optimizer)

    input = torch.rand(batch_size, sequence_length, d_input)
    output = model(input)
    loss = output.sum()
    loss.backward()
    print(output)
    optimizer.step()


"""
def test_s5_net():
    batch_size = 4
    sequence_length = 10246
    d_input = 50
    lr = 0.1

    model = S5Net(d_input, n_layers=4)
    torch.manual_seed(0)

    # Create an optimizer with the general parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    input = torch.rand(batch_size, sequence_length, d_input)
    output = model(input)
    loss = output.sum()
    loss.backward()
    print(output)
    optimizer.step()
"""


def test_s4_mask_net():
    batch_size = 4
    sequence_length = 10246
    d_input = 50
    lr = 0.1

    model = S4MaskNet(d_input, num_spks=2, n_layers=4)
    torch.manual_seed(0)

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s)
        for s in sorted(
            list(dict.fromkeys(frozenset(hp.items()) for hp in hps))
        )
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})
    print(optimizer)

    input = torch.rand(batch_size, d_input, sequence_length)
    output = model(input)
    loss = output.sum()
    loss.backward()
    print(output)
    optimizer.step()


def test_sashimi_mask_net():
    batch_size = 4
    sequence_length = 10246
    d_input = 50
    lr = 0.1

    model = SashimiMaskNet(d_input, num_spks=2, n_layers=4, pool=[4, 4])
    torch.manual_seed(0)

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s)
        for s in sorted(
            list(dict.fromkeys(frozenset(hp.items()) for hp in hps))
        )
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})
    print(optimizer)

    input = torch.rand(batch_size, d_input, sequence_length)
    output = model(input)
    loss = output.sum()
    loss.backward()
    print(output)
    optimizer.step()


if __name__ == "__main__":
    test_s4_net()
    #test_s5_net()
    test_s4_mask_net()
    test_sashimi_mask_net()
