"""SepS4 models.

Install the required CUDA extensions:
pip install git+https://github.com/HazyResearch/state-spaces.git@d589d982216485cce0a46bbe7605fe75c03d3223#subdirectory=extensions/cauchy

Authors
 * Luca Della Libera
"""

import math

import torch

try:
    from .sashimi import Sashimi, LinearActivation
except ImportError:
    from sashimi import Sashimi, LinearActivation


__all__ = ["SashimiMaskNet"]


class SashimiMaskNet(Sashimi):
    def __init__(self, num_spks=2, output_activation="relu", pool=(4, 4), **kwargs):
        super().__init__(pool=pool, **kwargs)
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
            pad = [0, math.ceil(output.shape[-1] / pool_factor) * pool_factor - output.shape[-1]]
            output = torch.nn.functional.pad(output, pad, value=0)
        # B x L x C
        output = output.movedim(-1, -2)
        output, _ = super().forward(output)
        if pad is not None:
            output = output[:, :-pad[1], :]
        output = self.projection(output)
        B, L, C = output.shape
        output = (
            output.movedim(-1, 0)
            .reshape(self.num_spks, -1, B, L)
            .movedim(1, -2)
        )
        return output


# Quick test
if __name__ == "__main__":
    batch_size = 4
    sequence_length = 10246
    input_size = 50
    lr = 0.1

    model = SashimiMaskNet(num_spks=2, d_model=input_size, n_layers=4, pool=[4, 4]).cuda()
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

    input = torch.rand(batch_size, input_size, sequence_length).cuda()
    output = model(input).cuda()
    loss = output.sum()
    loss.backward()
    print(output)
    optimizer.step()
