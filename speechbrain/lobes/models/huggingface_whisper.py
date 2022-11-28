# Adapted from:
# https://github.com/Adel-Moumen/speechbrain/blob/80005b82b195b3d8860b0ff56cbdd3135001f528/speechbrain/lobes/models/huggingface_whisper.py

"""This lobe enables the integration of huggingface pretrained whisper model.

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Adel Moumen 2022
 * Titouan Parcollet 2022
 * Luca Della Libera 2022
"""

import torch
import logging
from torch import nn

try:
    from transformers import WhisperModel
    from transformers import WhisperFeatureExtractor

except ImportError:
    MSG = "Please install transformers from HuggingFace to use Whisper\n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)

logger = logging.getLogger(__name__)


class HuggingFaceWhisper(nn.Module):
    """This lobe enables the integration of HuggingFace pretrained Whisper model.
    Source paper whisper:
        https://cdn.openai.com/papers/whisper.pdf
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be finetuned. It will download automatically the model from
    HuggingFace or use a local path.
    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "openai/whisper-tiny"
    save_path : str
        Path (dir) of the downloaded model.
    Example
    -------
    >>> model_hub = "openai/whisper-tiny"
    >>> save_path = "savedir"
    >>> sampling_rate = 16000
    >>> model = HuggingFaceWhisper(model_hub, save_path, sampling_rate)
    >>> tokens = torch.tensor([[1, 1]]) * model.model.config.decoder_start_token_id
    >>> inputs = torch.randn([1, 93680])
    >>> outputs = model(inputs, tokens)
    """

    def __init__(
        self,
        source,
        save_path,
        sampling_rate=16000,
        encoder_only=False,
        freeze=False,
        freeze_encoder=False,
    ):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.encoder_only = encoder_only
        self.freeze = freeze
        self.freeze_encoder = freeze_encoder

        # Download the extractor from HuggingFace.
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            source, cache_dir=save_path, sampling_rate=sampling_rate,
        )
        self._n_fft = feature_extractor.n_fft
        self._hop_length = feature_extractor.hop_length
        self._nb_max_frames = feature_extractor.nb_max_frames
        self.register_buffer(
            "_mel_filters", torch.as_tensor(feature_extractor.mel_filters)
        )

        self.model = WhisperModel.from_pretrained(source, cache_dir=save_path)

        if self.freeze:
            if self.encoder_only:
                logger_msg = "whisper encoder is frozen."
            else:
                logger_msg = "whisper encoder-decoder is frozen."
            logger.warning(
                "speechbrain.lobes.models.huggingface_whisper - " + logger_msg
            )
            self.model.train()  # we keep it to train to have dropout and LN computed adequaly
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            if self.freeze_encoder:
                logger.warning(
                    "speechbrain.lobes.models.huggingface_whisper - whisper encoder is frozen."
                )
                for param in self.model.encoder.parameters():
                    param.requires_grad = False

        if self.encoder_only:
            logger.warning(
                "speechbrain.lobes.models.huggingface_whisper - whisper encoder only, removing the decoder."
            )
            self.model.decoder = None  # TODO: del or None?

    def forward(self, wav, tokens=None):
        """Perform mel transformation and one step of the whisper (encoder-decoder).

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        tokens : torch.Tensor
            A batch of whisper decoder input ids. This is only necessary if the decoder is used.
        """
        if self.freeze:
            with torch.no_grad():
                out_encoder = self.forward_encoder(wav)
                if self.encoder_only:
                    return out_encoder
                out = self.forward_decoder(out_encoder, tokens)
                return out
        else:
            if self.encoder_only:
                return self.forward_encoder(wav)
            else:
                out_encoder = self.forward_encoder(wav)
                return self.forward_decoder(out_encoder, tokens)

    def forward_encoder(self, wav):
        """Perform one step of the whisper encoder.
        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        if self.freeze_encoder:
            with torch.no_grad():
                mel = self._get_mel(wav)
                return self.model.encoder(mel).last_hidden_state
        else:
            mel = self._get_mel(wav)
            return self.model.encoder(mel).last_hidden_state

    def _get_mel(self, wav):
        """Takes an input waveform and return its corresponding mel spectrogram.
        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """
        mels = self._log_mel_spectrogram(wav)
        mels = self._pad_or_trim(mels)
        return mels

    # Adapted from:
    # https://github.com/openai/whisper/blob/eff383b27b783e280c089475852ba83f20f64998/whisper/audio.py#L92
    def _log_mel_spectrogram(self, audio):
        """Compute the log-Mel spectrogram of a batch of input waveforms.

        Arguments
        ----------
        audio : torch.Tensor
            A batch of audio waveforms in 16 kHz.

        Returns
        -------
        torch.Tensor
            A tensor that contains the batch of log-Mel spectrograms.
        """
        window = torch.hann_window(self._n_fft).to(audio.device)
        stft = torch.stft(
            audio,
            self._n_fft,
            self._hop_length,
            window=window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2

        filters = self._mel_filters
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    # Adapted from:
    # https://github.com/openai/whisper/blob/eff383b27b783e280c089475852ba83f20f64998/whisper/audio.py#L52
    def _pad_or_trim(self, array, axis=-1):
        """Pad or trim the log-Mel spectrograms as expected by the encoder
        (see https://github.com/openai/whisper/blob/eff383b27b783e280c089475852ba83f20f64998/whisper/transcribe.py#L92,
        https://github.com/openai/whisper/blob/eff383b27b783e280c089475852ba83f20f64998/whisper/transcribe.py#L177, and
        https://github.com/openai/whisper/blob/eff383b27b783e280c089475852ba83f20f64998/whisper/audio.py#L19).

        Arguments
        ----------
        array : torch.Tensor
            A tensor that contains the batch of Mel spectrograms.
        axis :
            The axis along which to pad.

        Returns
        -------
        torch.Tensor
            The padded tensor.
        """
        if array.shape[axis] > self._nb_max_frames:
            array = array.index_select(
                dim=axis,
                index=torch.arange(self._nb_max_frames, device=array.device),
            )

        if array.shape[axis] < self._nb_max_frames:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, self._nb_max_frames - array.shape[axis])
            array = nn.functional.pad(
                array, [pad for sizes in pad_widths[::-1] for pad in sizes]
            )

        return array

    def forward_decoder(self, audio_features, tokens):
        """Perform one step of the whisper decoder.
        Arguments
        ---------
        audio_features : torch.Tensor
            A batch of audio features (mel + whisper encoding).
        tokens : torch.Tensor (TO DO: ARE MORE INFO IT S NOT CLEAR)
            A batch of whisper decoder input ids.
        """
        return self.model.decoder(
            encoder_hidden_states=audio_features, input_ids=tokens
        ).last_hidden_state
