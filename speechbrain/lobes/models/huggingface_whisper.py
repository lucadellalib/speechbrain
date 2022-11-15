"""This lobe enables the integration of Hugging Face pretrained Whisper models.

Reference: https://cdn.openai.com/papers/whisper.pdf
Transformer from Hugging Face needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Luca Della Libera 2022
 * Pooneh Mousavi 2022
"""

import logging
import os
import pathlib
from typing import Type

import torch
from torch import nn
from torch.nn import functional as F
from huggingface_hub import model_info

from speechbrain.pretrained.fetching import fetch


__all__ = [
    "HuggingFaceWhisper",
]


try:
    import transformers
    from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        _compute_mask_indices,
    )
except ImportError:
    raise ImportError(
        "Please install `transformers` from Hugging Face to use Whisper\n"
        "e.g. run `pip install transformers`"
    )


_LOGGER = logging.getLogger(__name__)


class HuggingFaceWhisper(nn.Module):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained Whisper models.

    Source paper: https://cdn.openai.com/papers/whisper.pdf
    Transformer from Hugging Face needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        Hugging Face hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    freeze_feature_extractor :  bool (default: False)
        When freeze = False and freeze_feature_extractor True, the featue_extractor module of the model is Frozen. If False
        all the wav2vec model will be trained including featue_extractor module.
    apply_spec_augment : bool (default: False)
        If True, the model will apply spec augment on the output of feature extractor
        (inside huggingface Wav2VecModel() class).
        If False, the model will not apply spec augment. We set this to false to prevent from doing it twice.
    output_all_hiddens : bool (default: False)
        If True, the forward function outputs the hidden states from all transformer layers.
        For example wav2vec2-base has 12 transformer layers and the output is of shape (13, B, T, C),
        where a projection of the CNN output is added to the beginning.
        If False, the forward function outputs the hidden states only from the last transformer layer.

    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "openai/whisper-base"
    >>> save_path = "savedir"
    >>> model = HuggingFaceWhisper(model_hub, save_path)
    >>> outputs = model(inputs)
    """

    def __init__(
        self,
        source: "str",
        save_path: "str",
        output_norm: "bool" = True,
        freeze: "bool" = True,
        freeze_feature_extractor: "bool" = False,
        apply_spec_augment: "bool" = False,
        output_all_hiddens: "bool" = False,
    ) -> "None":
        super().__init__()

        # Download the extractor from Hugging Face
        # The extractor is only used to retrieve the normalization information
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            source, cache_dir=save_path
        )

        # Download and load the model
        self._from_pretrained(
            source, config=WhisperConfig, model=WhisperModel, save_path=save_path
        )

        # Set apply_spec_augment
        self.model.config.apply_spec_augment = apply_spec_augment

        # We check if inputs need to be normalized w.r.t pretrained wav2vec2
        self.normalize_wav = self.feature_extractor.do_normalize

        self.freeze = freeze
        self.freeze_feature_extractor = freeze_feature_extractor
        self.output_norm = output_norm
        if self.freeze:
            _LOGGER.warning(
                "speechbrain.lobes.models.huggingface_wav2vec - wav2vec 2.0 is frozen."
            )
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            if self.freeze_feature_extractor:
                self.model.feature_extractor._freeze_parameters()
        self.output_all_hiddens = output_all_hiddens

    def _from_pretrained(self, source: "str", config: "Type[WhisperConfig]", model: "Type[WhisperModel]", save_path: "str") -> "None":
        """This function manages the source checking and loading of the params.
        # 1. Is the model from HF or a local path
        # 2. Is the model pretrained with HF or SpeechBrain
        # 3. Download (if appropriate) and load with respect to 1. and 2.
        """
        is_sb, ckpt_file = self._check_model_source(source)
        if is_sb:
            config = config.from_pretrained(source, cache_dir=save_path)
            self.model = model(config)
            self.model.gradient_checkpointing_disable()  # Required by DDP
            # Fetch the checkpoint file
            ckpt_full_path = fetch(
                filename=ckpt_file, source=source, savedir=save_path
            )
            # We transfer the parameters from the checkpoint
            self._load_sb_pretrained_w2v2_parameters(ckpt_full_path)
        else:
            self.model = model.from_pretrained(source, cache_dir=save_path)

    def _load_sb_pretrained_w2v2_parameters(self, path: "str") -> "None":
        """Loads the parameter of a w2v2 model pretrained with SpeechBrain and the
        HuggingFaceWav2Vec2Pretrain Object. It is necessary to perform a custom
        loading because HuggingFace adds a level to the checkpoint when storing
        the model breaking the compatibility between HuggingFaceWav2Vec2Pretrain
        and HuggingFaceWav2Vec2.

        In practice a typical HuggingFaceWav2Vec2 checkpoint for a given parameter
        would be: model.conv.weight.data while for HuggingFaceWav2Vec2Pretrain it
        is: model.wav2vec2.weight.data (wav2vec2 must be removed before loading).
        """

        modified_state_dict = {}
        orig_state_dict = torch.load(path, map_location="cpu")

        # We remove the .wav2vec2 in the state dict.
        for key, params in orig_state_dict.items():
            if "wav2vec2." in key:
                save_key = key.replace("model.wav2vec2.", "")
                modified_state_dict[save_key] = params

        incompatible_keys = self.model.load_state_dict(
            modified_state_dict, strict=False
        )
        for missing_key in incompatible_keys.missing_keys:
            _LOGGER.warning(
                f"During parameter transfer to {self.model} loading from "
                + f"{path}, the transferred parameters did not have "
                + f"parameters for the key: {missing_key}"
            )
        for unexpected_key in incompatible_keys.unexpected_keys:
            _LOGGER.warning(
                f"The param with the key: {unexpected_key} is discarded as it "
                + "is useless for wav2vec 2.0 finetuning."
            )

    def _check_model_source(self, path: "str"):
        """Checks if the pretrained model has been trained with SpeechBrain and
        is hosted locally or on a Hugging Face hub.
        """
        checkpoint_filename = ""
        source = pathlib.Path(path)
        is_local = True
        is_sb = True

        # If path is a Hugging Face hub.
        if not source.exists():
            is_local = False

        if is_local:
            # Test for Hugging Face model
            if any(File.endswith(".bin") for File in os.listdir(path)):
                is_sb = False
                return is_sb, checkpoint_filename

            # Test for SpeechBrain model and get the filename
            for file in os.listdir(path):
                if file.endswith(".ckpt"):
                    checkpoint_filename = os.path.join(path, file)
                    is_sb = True
                    return is_sb, checkpoint_filename
        else:
            files = model_info(
                path
            ).siblings  # Get list of files of the hub

            # Test if it's an Hugging Face model or a SB one
            for file in files:
                if file.rfilename.endswith(".ckpt"):
                    checkpoint_filename = file.rfilename
                    is_sb = True
                    return is_sb, checkpoint_filename

            for file in files:
                if file.rfilename.endswith(".bin"):
                    checkpoint_filename = file.rfilename
                    is_sb = False
                    return is_sb, checkpoint_filename

        err_msg = f"{path} does not contain a .bin or .ckpt checkpoint !"
        raise FileNotFoundError(err_msg)

    def forward(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        # If we freeze, we simply remove all grads and features from the graph
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(wav)  # .detach() Useless...

        return self.extract_features(wav)

    def extract_features(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape)

        # Extract Whisper output
        out = self.model(wav, output_hidden_states=True)

        if self.output_all_hiddens:
            out = torch.stack(list(out[2]), dim=0)
            norm_shape = out.shape[-3:]
        else:
            out = out[0]
            norm_shape = out.shape

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, norm_shape)

        return out
