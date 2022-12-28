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
    from transformers.models.whisper.tokenization_whisper import (
        LANGUAGES,
        TASK_IDS,
        TO_LANGUAGE_CODE,
        WhisperTokenizer,
    )
except ImportError:
    MSG = "Please install transformers from HuggingFace to use Whisper\n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)

logger = logging.getLogger(__name__)


class CustomWhisperTokenizer(WhisperTokenizer):
    # override
    @property
    def prefix_tokens(self):
        # all_special_ids = self.all_special_ids
        bos_token_id = 50258  # all_special_ids[-106]
        translate_token_id = 50358  # all_special_ids[-6]
        transcribe_token_id = 50359  # all_special_ids[-5]
        notimestamps_token_id = 50363  # all_special_ids[-1]
        # langs = tuple(LANGUAGES.keys())

        if self.language is not None:
            self.language = self.language.lower()
            if self.language in TO_LANGUAGE_CODE:
                language_id = TO_LANGUAGE_CODE[self.language]
            else:
                raise ValueError(
                    f"Unsupported language: {self.language}. Language should be in: {TO_LANGUAGE_CODE.keys()}"
                )

        if self.task is not None:
            if self.task not in TASK_IDS:
                raise ValueError(
                    f"Unsupported task: {self.task}. Task should be in: {TASK_IDS}"
                )

        bos_sequence = [bos_token_id]
        if self.language is not None:
            # Need to replace with custom code because language ID is hardcoded...
            # bos_sequence.append(bos_token_id + 1 + langs.index(language_id))
            bos_sequence.append(
                self.encode(f"<|{language_id}|>", add_special_tokens=False)[0]
            )
        if self.task is not None:
            bos_sequence.append(
                transcribe_token_id
                if self.task == "transcribe"
                else translate_token_id
            )
        if not self.predict_timestamps:
            bos_sequence.append(notimestamps_token_id)
        return bos_sequence


class HuggingFaceWhisper(nn.Module):
    """This lobe enables the integration of HuggingFace pretrained Whisper model.
    Source paper whisper:
        https://cdn.openai.com/papers/whisper.pdf
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html
    Some part of the code also cis adapted from the official OpenAI repository:
    https://github.com/openai/whisper
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
        output_attentions=True,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.encoder_only = encoder_only
        self.freeze = freeze
        self.freeze_encoder = freeze_encoder
        self.output_attentions = output_attentions

        self.tokenizer = None
        # Download the tokenizer only if we are going to use the Decoder.
        if not encoder_only:
            self.tokenizer = CustomWhisperTokenizer.from_pretrained(
                source,
                language=None,
                task="transcribe",
                predict_timestamps=False,
            )
            self.tokenizer.supported_languages = LANGUAGES
            self.tokenizer.to_language_codes = TO_LANGUAGE_CODE

        # Download the extractor from HuggingFace.
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            source, cache_dir=save_path, sampling_rate=sampling_rate,
        )
        self._n_fft = feature_extractor.n_fft
        self._hop_length = feature_extractor.hop_length
        self._n_samples = feature_extractor.n_samples
        self.register_buffer(
            "_mel_filters", torch.as_tensor(feature_extractor.mel_filters)
        )

        self.model = WhisperModel.from_pretrained(source, cache_dir=save_path)

        if self.freeze:
            logger.warning(
                "speechbrain.lobes.models.huggingface_whisper - whisper encoder-decoder is frozen."
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

    def forward(self, wav, decoder_input_ids=None):
        """Perform mel transformation and one step of the whisper (encoder-decoder).
        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        decoder_input_ids : torch.Tensor
            This is necessary if we want to use the decoder.
            A batch of decoder inputs tokens.
            The first tokens need to dictacte the behavior of the decoder.
            It needs to start with the bos_token, the language token,
            the task token, and finally the timestamp token.
            Please refer to the whisper paper for more details or go to the
            seq2seq2.py file in SpeechBrain to see how to generate the tokens
            with Greedy Search and/or Beam Search.
        """
        if self.freeze:
            with torch.no_grad():
                out_encoder = self.forward_encoder(wav)
                if self.encoder_only:
                    return out_encoder
                logits, attn = self.forward_decoder(
                    out_encoder, decoder_input_ids
                )
                return out_encoder, logits, attn
        else:
            if self.encoder_only:
                return self.forward_encoder(wav)
            else:
                out_encoder = self.forward_encoder(wav)
                logits, attn = self.forward_decoder(
                    out_encoder, decoder_input_ids
                )
                return out_encoder, logits, attn

    def forward_encoder(self, wav):
        """Perform one step of the whisper encoder with Mel FBANKs as Input.
        Arguments
        ---------
        wav : torch.Tensor (FBANKs)
            A batch of Mel FBANK from HF to transform to features.
        """

        if self.freeze_encoder:
            with torch.no_grad():
                mel = self._get_mel(wav)
                return self.model.encoder(mel).last_hidden_state
        else:
            mel = self._get_mel(wav)
            return self.model.encoder(mel).last_hidden_state

    def _get_mel(self, wav):
        """Takes an input waveform and return its corresponding mel spectrogram
        according to HuggingFace implementation. WARNING: it's slow! Better push this
        in the DataLoader.
        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """
        mels = self._pad_or_trim(wav)
        mels = self._log_mel_spectrogram(mels)
        return mels

    def _log_mel_spectrogram(self, audio):
        """Compute the Mel spectrogram of a batch of input waveforms.
        Reference: adapted from
        https://github.com/openai/whisper/blob/eff383b27b783e280c089475852ba83f20f64998/whisper/audio.py#L92
        Arguments
        ---------
        audio : torch.Tensor
            A batch of audio waveforms in 16 kHz.
        Returns
        -------
        torch.Tensor
            A tensor that contains the batch of Mel spectrograms.
        """
        window = torch.hann_window(self._n_fft, device=audio.device)
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
        log_spec = torch.maximum(
            log_spec,
            (log_spec.flatten(start_dim=1).max(dim=-1)[0] - 8.0)[:, None, None],
        )
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def _pad_or_trim(self, array, axis=-1):
        """Pad or trim the Mel spectrograms as expected by the encoder.
        Reference: adapted from
        https://github.com/openai/whisper/blob/eff383b27b783e280c089475852ba83f20f64998/whisper/audio.py#L52
        Arguments
        ---------
        array : torch.Tensor
            A tensor that contains the batch of Mel spectrograms.
        axis : int
            The axis along which to pad.
        Returns
        -------
        torch.Tensor
            The padded tensor.
        """
        if array.shape[axis] > self._n_samples:
            array = array.index_select(
                dim=axis,
                index=torch.arange(self._n_samples, device=array.device),
            )

        if array.shape[axis] < self._n_samples:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (
                0,
                self._n_samples - array.shape[axis],
            )
            array = nn.functional.pad(
                array, [pad for sizes in pad_widths[::-1] for pad in sizes]
            )

        return array

    def forward_decoder(self, audio_features, decoder_input_ids):
        """Perform one step of the whisper decoder.
        Arguments
        ---------
        audio_features : torch.Tensor
            A batch of audio features (mel + whisper encoding).
        decoder_input_ids : torch.Tensor
            A batch of decoder inputs tokens.
            The first tokens need to dictacte the behavior of the decoder.
            It needs to start with the bos_token, the language token,
            the task token, and finally the timestamp token.
            Please refer to the whisper paper for more details or go to the
            seq2seq2.py file in SpeechBrain to see how to generate the tokens
            with Greedy Search and/or Beam Search.
        """
        output_states = self.model.decoder(
            encoder_hidden_states=audio_features,
            input_ids=decoder_input_ids,
            output_attentions=self.output_attentions,
        )

        attn = output_states.attentions[-1]
        attn = attn.view(attn.shape[0] * attn.shape[1], *attn.shape[2:])
        output_states = output_states.last_hidden_state

        logits = (
            output_states
            @ torch.transpose(
                self.model.decoder.embed_tokens.weight.to(output_states.dtype),
                0,
                1,
            )
        ).to(audio_features.dtype)

        return logits, attn

    @torch.no_grad()
    def generate(
        self,
        wav=None,
        audio_features=None,
        forced_decoder_locale=None,
        max_gen_tokens=445,
        strategy="greedy",
    ):
        if wav is None and audio_features is None:
            raise ValueError(
                "Either `wav` or `audio_features` argument should be given"
            )
        if audio_features is None:
            audio_features = self.forward_encoder(wav)
        batch_size = audio_features.shape[0]
        (
            startoftranscript_id,
            transcribe_id,
            notimestamps_id,
        ) = self.tokenizer.prefix_tokens
        pad_id = self.model.config.pad_token_id
        endoftext_id = self.tokenizer.eos_token_id

        hyps = torch.full(
            (batch_size, max_gen_tokens + 4),
            pad_id,
            dtype=torch.long,
            device=audio_features.device,
        )
        if forced_decoder_locale is None:
            # Compute most likely language token IDs
            all_lang_tokens = [
                f"<|{l}|>" for l in self.tokenizer.supported_languages
            ]
            all_lang_tokens_ids = self.tokenizer.convert_tokens_to_ids(
                all_lang_tokens
            )
            hyps[:, 0] = startoftranscript_id
            logits, _ = self.forward_decoder(audio_features, hyps[:, :1])
            lang_mask = torch.zeros(
                logits.shape[-1], device=logits.device, dtype=torch.bool
            )
            lang_mask[all_lang_tokens_ids] = True
            logits[:, :, ~lang_mask] = -float("inf")
            lang_tokens_ids = logits.argmax(dim=-1)[:, 0]
        else:
            if forced_decoder_locale.lower() == "zh-cn":
                forced_decoder_locale = "zh"
            if forced_decoder_locale.lower() not in LANGUAGES:
                raise NotImplementedError(
                    f"Unsupported language: {forced_decoder_locale}"
                )
            lang_tokens_ids = self.tokenizer.convert_tokens_to_ids(
                f"<|{forced_decoder_locale.lower()}|>"
            )

        # Prepare initial tokens in the right format
        hyps[:, 0] = startoftranscript_id
        hyps[:, 1] = lang_tokens_ids
        hyps[:, 2] = transcribe_id
        hyps[:, 3] = notimestamps_id

        # Autoregressive loop
        num_gen_tokens = 0
        unfinished_mask = torch.ones(
            len(hyps), dtype=torch.bool, device=audio_features.device
        )
        while True:
            logits, _ = self.forward_decoder(
                audio_features[unfinished_mask],
                hyps[unfinished_mask, : num_gen_tokens + 4],
            )
            # Prepare suppress mask
            suppress_mask = torch.ones(
                logits.shape[-1], device=audio_features.device, dtype=torch.bool
            )
            suppress_mask[self.model.config.suppress_tokens] = False
            logits[:, :, ~suppress_mask] = -float("inf")
            gen_tokens = logits.argmax(dim=-1)[:, -1]
            hyps[unfinished_mask, num_gen_tokens + 4] = gen_tokens
            unfinished_mask[unfinished_mask == True] = (
                gen_tokens != endoftext_id
            )
            num_gen_tokens += 1
            if (not unfinished_mask.any()) or (
                num_gen_tokens >= max_gen_tokens
            ):
                break
        return hyps[:, 4 : num_gen_tokens + 3]
