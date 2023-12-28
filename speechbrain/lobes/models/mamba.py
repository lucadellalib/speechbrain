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


__all__ = ["MambaNet"]


from mamba_ssm.models.mixer_seq_simple import layer_norm_fn, rms_norm_fn, create_block, RMSNorm, partial, _init_weights


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input, inference_params=None):
        hidden_states = input
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states



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


if __name__ == "__main__":
    test_mamba_net()
