"""GSS-based separation models.

Authors
 * Luca Della Libera
"""

import torch
from gated_state_spaces_pytorch import GSS
from torch import nn


__all__ = ["SBTransformerBlock_wnormandskip"]


class SBTransformerBlock_wnormandskip(nn.Module):
    def __init__(
        self,
        d_model,
        num_layers=1,
        dim_expansion_factor=4,  # hidden dimension (expansion factor x d_model) = 2048
        dss_kernel_N=512,
        dss_kernel_H=256,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GSS(
                    dim=d_model,
                    dim_expansion_factor=dim_expansion_factor,
                    dss_kernel_N=dss_kernel_N,
                    dss_kernel_H=dss_kernel_H,
                    **kwargs,
                )
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
