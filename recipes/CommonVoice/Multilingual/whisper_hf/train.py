#!/usr/bin/env python3

# ==============================================================================
# Copyright 2022 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import evaluate
import torch
from datasets import Audio, load_dataset
from torch import Tensor
from transformers import (
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from transformers.models.whisper.tokenization_whisper import (
    LANGUAGES,
    TO_LANGUAGE_CODE,
)

from common_voice_prepare import prepare_common_voice


__all__ = [
    "fine_tune_whisper",
]


KNOWN_LANGUAGES = list(LANGUAGES.keys())


class WhisperForLanguageTranscription(WhisperForConditionalGeneration):
    def __init__(
        self,
        config: "WhisperConfig",
        available_language_token_ids: "Optional[Sequence[int]]" = None,
        forced_language_id: "Optional[int]" = None,
    ) -> "None":
        super().__init__(config)
        self.available_language_token_ids = available_language_token_ids
        self.forced_language_id = (
            forced_language_id  # Can be changed after model initialization
        )
        self._startoftranscript_id = 50258
        self._transcribe_id = 50359
        self._notimestamps_id = 50363

    def generate(
        self, inputs: "Optional[torch.Tensor]" = None, **kwargs: "Any"
    ) -> "Any":
        if (
            self.available_language_token_ids is not None
            and self.forced_language_id is None
        ):
            self.config.forced_decoder_ids = None
            (
                self.predicted_language_token_id,
                decoder_input_ids,
            ) = self._predict_language_token_id(
                inputs, self.available_language_token_ids,
            )
            kwargs["decoder_input_ids"] = decoder_input_ids
        else:
            self.predicted_language_token_id = None
            # If self.forced_language_id is None leave default value
            if self.forced_language_id is not None:
                self.config.forced_decoder_ids = [
                    (1, self.forced_language_id),
                    (2, self._transcribe_id),
                    (3, self._notimestamps_id),
                ]
        return super().generate(inputs, **kwargs)

    # Adapted from:
    # https://discuss.huggingface.co/t/language-detection-with-whisper/26003/2
    def _predict_language_token_id(
        self,
        input_features: "Tensor",
        available_language_token_ids: "Sequence[int]",
    ) -> "Tuple[Tensor, Tensor]":
        # Compute logits
        logits = self.forward(
            input_features,
            decoder_input_ids=torch.full(
                (input_features.shape[0], 1),
                self._startoftranscript_id,
                device=input_features.device,
            ),
        ).logits
        mask = torch.ones(
            logits.shape[-1], device=logits.device, dtype=torch.bool
        )
        mask[available_language_token_ids] = False
        logits[:, :, mask] = -float("inf")

        # Compute most likely language token
        predicted_language_token_id = logits.argmax(dim=-1)

        # Prepare decoder input ids
        decoder_input_ids = torch.empty(
            (logits.shape[0], 4), device=logits.device
        )
        decoder_input_ids[:, 0] = self._startoftranscript_id
        decoder_input_ids[:, 1] = predicted_language_token_id[:, 0]
        decoder_input_ids[:, 2] = self._transcribe_id
        decoder_input_ids[:, 3] = self._notimestamps_id

        return predicted_language_token_id.long(), decoder_input_ids.long()


# Adapted from:
# https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb
def fine_tune_whisper(
    dataset_size: "str",
    whisper_model: "str" = "openai/whisper-tiny",
    test_only: "bool" = False,
    prepare_common_voice_kwargs: "Dict[str, Any]" = None,
    training_kwargs: "Dict[str, Any]" = None,
) -> "None":
    """Fine-tune Whisper model on Common Voice dataset.

    Parameters
    ----------
    dataset_size:
        The Common Voice dataset size. Must be one of the following:
        - "small":   1h train, 15 min dev, 15 min test;
        - "medium": 10h train, 15 min dev, 15 min test;
        - "large": full train, 15 min dev, 15 min test;
        - "full":  full train,   full dev,   full test.
    whisper_model:
        The path to a directory containing the Whisper
        model checkpoint.
    test_only:
        True to skip fine-tuning, False otherwise.
    prepare_common_voice_kwargs:
        The keyword arguments to pass to `prepare_common_voice`
        (see common_voice_prepare.py).
    training_kwargs:
        The keyword arguments to pass to `Seq2SeqTrainingArguments`
        (see https://github.com/huggingface/transformers/blob/e82c1cb78e178519060b9391214727be75a218ca/src/transformers/training_args.py#L121).

    Examples
    --------
    >>> fine_tune_whisper("small", whisper_model="openai/whisper-tiny")

    """
    prepare_common_voice_kwargs = prepare_common_voice_kwargs or {}
    training_kwargs = training_kwargs or {}

    # FIXME:
    prepare_common_voice_kwargs[
        "manifest_dir"
    ] = f"../data/common_voice_10_0_{dataset_size}"

    # Prepare data
    prepare_common_voice(dataset_size, **prepare_common_voice_kwargs)

    # Set default values as in `prepare_common_voice`
    dataset_version = prepare_common_voice_kwargs.get("dataset_version", "10_0")
    dataset_dir = prepare_common_voice_kwargs.get(
        "dataset_dir", os.path.join("data", f"common_voice_{dataset_version}")
    )
    manifest_dir = prepare_common_voice_kwargs.get(
        "manifest_dir", f"{dataset_dir}_{dataset_size}"
    )
    locales = prepare_common_voice_kwargs.get("locales") or []

    # Build pipeline
    processor = WhisperProcessor.from_pretrained(
        whisper_model, task="transcribe"
    )
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    # Build model
    available_language_tokens = [f"<|{l}|>" for l in LANGUAGES]
    available_language_token_ids = tokenizer.convert_tokens_to_ids(
        available_language_tokens
    )
    model = WhisperForLanguageTranscription.from_pretrained(
        whisper_model,
        available_language_token_ids=available_language_token_ids,
    )
    model.config.use_cache = False

    # No tokens are forced as decoder outputs, no tokens are suppressed during generation
    model.forced_language_id = None
    model.config.suppress_tokens = []

    # Build dataset
    data_files = {
        "test": os.path.join(manifest_dir, "test.csv"),
    }
    if not test_only:
        data_files.update(
            {
                "train": os.path.join(manifest_dir, "train.csv"),
                "dev": os.path.join(manifest_dir, "dev.csv"),
            }
        )
    dataset = load_dataset("csv", data_files=data_files)
    dataset = dataset.remove_columns(
        [
            "ID",
            # "accents",
            "age",
            "client_id",
            "down_votes",
            "duration",
            "gender",
            "segment",
            "up_votes",
        ]
    )

    def resolve_root_dir(sample: "Dict[str, Any]") -> "Dict[str, Any]":
        sample["mp3"] = sample["mp3"].replace("$root_dir", manifest_dir)
        return sample

    dataset = dataset.map(resolve_root_dir)
    dataset = dataset.rename_columns({"mp3": "audio", "wrd": "sentence"})
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Add `diff` dummy languages to make `tokenizer.prefix_tokens`
    # return the correct id for new added languages
    # (see https://github.com/huggingface/transformers/blob/95754b47a6d4fbdad3440a45762531e8c471c528/src/transformers/models/whisper/tokenization_whisper.py#L413)
    last_language = list(LANGUAGES)[-1]
    last_language_index = tokenizer.additional_special_tokens.index(
        f"<|{last_language}|>"
    )
    diff = len(tokenizer.additional_special_tokens) - last_language_index - 1
    for i in range(diff):
        LANGUAGES[i] = i

    def add_unknown_locales(sample: "Dict[str, Any]") -> "Dict[str, Any]":
        # Add locale if unknown
        locale = sample["locale"].lower()
        if locale not in LANGUAGES:
            TO_LANGUAGE_CODE[locale] = LANGUAGES[locale] = locale
            tokenizer.add_tokens(f"<|{locale}|>", special_tokens=True)
        return sample

    dataset.map(
        add_unknown_locales,
        remove_columns=[
            n for n in dataset.column_names["test"] if n != "locale"
        ],
    )

    def prepare_dataset(sample: "Dict[str, Any]") -> "Dict[str, Any]":
        # Load and resample audio data from 48kHz to 16kHz
        audio = sample["audio"]

        # Compute log-Mel input features from input audio array
        sample["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"],
        ).input_features[0]

        # Set locale
        locale = sample["locale"].lower()
        language = LANGUAGES[locale]
        tokenizer.set_prefix_tokens(
            language, task="transcribe", predict_timestamps=False
        )

        # Encode target text to label ids
        sample["labels"] = tokenizer(sample["sentence"]).input_ids
        return sample

    dataset = dataset.map(prepare_dataset, remove_columns=["locale"],)

    # Build data collator
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: "WhisperProcessor"

        def __call__(
            self, features: "List[Dict[str, Union[List[int], Tensor]]]"
        ) -> "Dict[str, Tensor]":
            # Split inputs and labels since they have to be of different lengths and need different padding
            # methods first treat the audio inputs by simply returning torch tensors
            input_features = [
                {"input_features": feature["input_features"]}
                for feature in features
            ]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )

            # Get the tokenized label sequences
            label_features = [
                {"input_ids": feature["labels"]} for feature in features
            ]

            # Pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )

            # Replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # If BOS token is appended in previous tokenization step,
            # cut BOS token here as it's append later anyway
            if (
                (labels[:, 0] == self.processor.tokenizer.bos_token_id)
                .all()
                .cpu()
                .item()
            ):
                labels = labels[:, 1:]

            batch["labels"] = labels

            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Set default training arguments
    training_kwargs.setdefault("seed", 1234)
    training_kwargs.setdefault(
        "output_dir",
        os.path.join(
            "results",
            "multilingual" if not locales else "_".join(locales),
            dataset_size,
            os.path.basename(whisper_model) + ("-ft" if not test_only else ""),
            str(training_kwargs["seed"]),
        ),
    )
    training_kwargs.setdefault("per_device_train_batch_size", 2)
    training_kwargs.setdefault("gradient_accumulation_steps", 8)
    training_kwargs.setdefault("learning_rate", 1e-4)
    training_kwargs.setdefault("num_train_epochs", 20)
    training_kwargs.setdefault("gradient_checkpointing", True)
    training_kwargs.setdefault("dataloader_num_workers", 4)
    training_kwargs.setdefault("fp16", True)
    training_kwargs.setdefault("group_by_length", True)
    training_kwargs.setdefault("evaluation_strategy", "epoch")
    training_kwargs.setdefault("per_device_eval_batch_size", 2)
    training_kwargs.setdefault("predict_with_generate", True)
    training_kwargs.setdefault("generation_max_length", 225)
    training_kwargs.setdefault("save_strategy", "epoch")
    training_kwargs.setdefault("logging_strategy", "epoch")
    training_kwargs.setdefault("report_to", ["tensorboard"])
    training_kwargs.setdefault("load_best_model_at_end", True)
    training_kwargs.setdefault("metric_for_best_model", "loss")
    training_kwargs.setdefault("greater_is_better", False)
    training_kwargs.setdefault("push_to_hub", False)
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    # Build performance metrics
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    # Build normalizer
    with open(os.path.join(os.path.dirname(__file__), "english.json")) as f:
        english_spelling_mapping = json.load(f)
    normalizer = EnglishTextNormalizer(english_spelling_mapping)

    def compute_metrics(pred: "EvalPrediction") -> "Dict[str, float]":
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # We do not want to group tokens when computing the metrics
        preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        languages = tokenizer.batch_decode(
            [s[1] for s in pred_ids], skip_special_tokens=False
        )

        # Normalize
        preds = [normalizer(s) for s in preds]
        labels = [normalizer(s) for s in labels]

        # Write transcription file
        output_dir = training_kwargs["output_dir"]
        output_file = os.path.join(output_dir, "transcriptions.csv")
        i = 0
        while os.path.exists(f"{output_file}_{i}.csv"):
            i += 1
        with open(f"{output_file}_{i}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["language", "transcription", "reference"])
            for language, transcription, reference in zip(
                languages, preds, labels
            ):
                writer.writerow([language, transcription, reference])

        cer = 100 * cer_metric.compute(predictions=preds, references=labels)
        wer = 100 * wer_metric.compute(predictions=preds, references=labels)

        return {"cer": cer, "wer": wer}

    # Build trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"] if "train" in dataset else None,
        eval_dataset=dataset["dev"] if "dev" in dataset else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    processor.save_pretrained(training_args.output_dir)

    # Train
    lines = []
    if not test_only:
        trainer.train()

        # Write training log in SpeechBrain format
        log_history = trainer.state.log_history
        for i, metrics in enumerate(log_history):
            if "eval_loss" in metrics:
                train_metrics = log_history[i - 1]
                lr = train_metrics["learning_rate"]
                train_loss = train_metrics["loss"]
                epoch = metrics["epoch"]
                valid_loss = metrics["eval_loss"]
                valid_cer = metrics["eval_cer"]
                valid_wer = metrics["eval_wer"]
                line = (
                    f"epoch: {int(epoch)}, lr: {lr:.2e} - train loss: {train_loss:.2f} - valid loss: {valid_loss:.2f}, "
                    f"valid CER: {valid_cer:.2f}, valid WER: {valid_wer:.2f}"
                )
                lines.append(line)

    # If single known language, set it in the decoder before testing
    if len(locales) == 1 and locales[0] in KNOWN_LANGUAGES:
        model.forced_language_id = tokenizer.convert_tokens_to_ids(
            f"<|{locales[0].lower()}|>"
        )

    # Test
    test_metrics = trainer.evaluate(dataset["test"])
    epoch = test_metrics.get("epoch", 0) + 1
    test_loss = test_metrics["eval_loss"]
    test_cer = test_metrics["eval_cer"]
    test_wer = test_metrics["eval_wer"]
    line = (
        f"epoch: {int(epoch)} - test loss: {test_loss:.2f}, "
        f"test CER: {test_cer:.2f}, test WER: {test_wer:.2f}"
    )
    lines.append(line)

    # Write log file
    with open(
        os.path.join(training_kwargs["output_dir"], "train_log.txt"), "w"
    ) as f:
        content = "\n".join(lines)
        f.write(content)
        print(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper on Common Voice 10.0"
    )
    parser.add_argument(
        "dataset_size",
        choices=["small", "medium", "large", "full"],
        help="dataset size",
    )
    parser.add_argument(
        "-m",
        "--whisper_model",
        help="path to a directory containing the Whisper model checkpoint",
    )
    parser.add_argument(
        "-t", "--test_only", action="store_true", help="skip fine-tuning"
    )
    parser.add_argument(
        "-p",
        "--prepare_common_voice_kwargs",
        default="{}",
        type=lambda x: eval(x.replace("`", "'")),
        help="`prepare_common_voice` keyword arguments as Python code (see common_voice_prepare.py)",
    )
    parser.add_argument(
        "-c",
        "--training_kwargs",
        default="{}",
        type=lambda x: eval(x.replace("`", "'")),
        help=(
            "training keyword arguments as Python code "
            "(see https://github.com/huggingface/transformers/blob/e82c1cb78e178519060b9391214727be75a218ca/src/transformers/training_args.py#L121)"
        ),
    )

    args = parser.parse_args()
    fine_tune_whisper(
        args.dataset_size,
        args.whisper_model,
        args.test_only,
        args.prepare_common_voice_kwargs,
        args.training_kwargs,
    )
