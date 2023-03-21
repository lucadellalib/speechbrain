"""H3-based separation models.

Install the required CUDA extensions:
pip install git+https://github.com/HazyResearch/H3.git@8ebedd61275770b1fca6e0f8a31e642529d8aa97#subdirectory=csrc/cauchy
pip install git+https://github.com/HazyResearch/H3.git@8ebedd61275770b1fca6e0f8a31e642529d8aa97#subdirectory=csrc/fftconv

Authors
 * Luca Della Libera
"""

import torch
from torch import nn


try:
    from .h3 import H3
except ImportError:
    from h3 import H3


class H3MaskNet(nn.Module):
    def __init__(
        self,
        d_input,
        num_spks=2,
        n_layers=1,
        d_state=64,
        output_activation=nn.functional.relu,
        use_fast_fftconv=False,
        head_dim=1,
        mode="diag",
        measure="diag-lin",
        **kwargs,
    ):
        super().__init__()
        self.d_input = d_input
        self.num_spks = num_spks
        self.n_layers = n_layers
        self.d_state = d_state
        self.output_activation = output_activation
        self.layers = nn.ModuleList(
            [
                H3(
                    d_input,
                    d_state,
                    head_dim=head_dim,
                    use_fast_fftconv=use_fast_fftconv,
                    layer_idx=i,
                    mode=mode,
                    measure=measure,
                    **kwargs,
                )
                for i in range(n_layers - 1)
            ]
        )
        self.output_proj = nn.Linear(d_input, num_spks * d_input)

    def forward(self, input, *args, **kwargs):
        # input: B x C x L
        output = input.movedim(-1, -2)
        for layer in self.layers:
            output = layer(output, *args, **kwargs)
        output = self.output_proj(output)
        output = self.output_activation(output)
        B, L, C = output.shape
        output = (
            output.movedim(-1, 0)
            .reshape(self.num_spks, -1, B, L)
            .movedim(1, -2)
        )
        return output


def test_h3_mask_net():
    batch_size = 4
    sequence_length = 10246
    d_input = 50
    lr = 0.1

    model = H3MaskNet(d_input, num_spks=2, n_layers=4).cuda()
    torch.manual_seed(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    input = torch.rand(batch_size, d_input, sequence_length, device="cuda")
    output = model(input)
    loss = output.sum()
    loss.backward()
    print(output)
    optimizer.step()


if __name__ == "__main__":
    test_h3_mask_net()
