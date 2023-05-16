"""S5-based separation models.

Authors
 * Luca Della Libera
"""

import torch
torch.jit.script = lambda fn, *args, **kwargs: fn  # Monkey patch unpickable torch.jit.script
from s5 import S5Block
from torch import nn


__all__ = ["SBTransformerBlock_wnormandskip"]


class SBTransformerBlock_wnormandskip(nn.Module):
    def __init__(
        self,
        d_model,
        num_layers=1,
        d_state=64,
        block_count=8,
        bidirectional=False,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_state = d_state
        self.layers = nn.ModuleList(
            [
                S5Block(d_model, d_state, block_count=block_count, bidir=bidirectional, **kwargs)
                for _ in range(num_layers)
            ]
        )

    def forward(self, input, *args, **kwargs):
        # input: B x L x C
        output = input
        for layer in self.layers:
            output = layer(output, *args, **kwargs)
        return output


def test_SBTransformerBlock_wnormandskip():
    batch_size = 7
    sequence_length = 150
    d_model = 128
    lr = 0.1

    model = SBTransformerBlock_wnormandskip(d_model, num_layers=4).cuda()
    torch.manual_seed(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    input = torch.rand(batch_size, sequence_length, d_model).cuda()
    output = model(input)
    loss = output.sum()
    loss.backward()
    print(input.shape)
    print(output.shape)
    optimizer.step()


if __name__ == "__main__":
    test_SBTransformerBlock_wnormandskip()
