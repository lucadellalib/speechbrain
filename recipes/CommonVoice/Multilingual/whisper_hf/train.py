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
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
from datasets import Audio, load_dataset
from torch import Tensor
from transformers import (
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from transformers.models.whisper.tokenization_whisper import LANGUAGES

from common_voice_prepare import prepare_common_voice


__all__ = [
    "fine_tune_whisper",
]


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
    locales = prepare_common_voice_kwargs.get("locales", [])

    # Build pipeline
    language = LANGUAGES[locales[0]] if len(locales) == 1 else None
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
    tokenizer = WhisperTokenizer.from_pretrained(
        whisper_model, language=language, task="transcribe",
    )
    processor = WhisperProcessor.from_pretrained(
        whisper_model, language=language, task="transcribe"
    )

    # Build dataset
    dataset = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(manifest_dir, "train.csv"),
            "dev": os.path.join(manifest_dir, "dev.csv"),
            "test": os.path.join(manifest_dir, "test.csv"),
        },
    )
    dataset = dataset.remove_columns(
        [
            "ID",
            # "accents",
            "age",
            "client_id",
            "down_votes",
            "duration",
            "gender",
            "locale",
            "segment",
            "up_votes",
        ]
    )

    def resolve_root_dir(sample: "Dict[str, Any]") -> "Dict[str, Any]":
        sample["mp3"] = sample["mp3"].replace("$root_dir", manifest_dir)
        return sample

    dataset = dataset.map(resolve_root_dir, num_proc=8)
    dataset = dataset.rename_columns({"mp3": "audio", "wrd": "sentence"})
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(sample: "Dict[str, Any]") -> "Dict[str, Any]":
        # Load and resample audio data from 48kHz to 16kHz
        audio = sample["audio"]

        # Compute log-Mel input features from input audio array
        sample["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"],
        ).input_features[0]

        # Encode target text to label ids
        sample["labels"] = tokenizer(sample["sentence"]).input_ids
        return sample

    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names["train"],
        num_proc=8,
    )

    # Build data collator
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: "Any"

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
            # cut BOS token here as it's append later anyways
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
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize
        pred_str = [normalizer(s) for s in pred_str]
        label_str = [normalizer(s) for s in label_str]

        print("Example transcription/label:")
        print(pred_str[0])
        print(label_str[0])

        cer = 100 * cer_metric.compute(
            predictions=pred_str, references=label_str
        )
        wer = 100 * wer_metric.compute(
            predictions=pred_str, references=label_str
        )

        return {"cer": cer, "wer": wer}

    # Build model
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model)

    # No tokens are forced as decoder outputs, no tokens are suppressed during generation
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Set default training arguments
    training_kwargs.setdefault("seed", 1234)
    training_kwargs.setdefault(
        "output_dir",
        os.path.join(
            "results",
            "multilingual" if not locales else "_".join(locales),
            dataset_size,
            os.path.basename(whisper_model),
            str(training_kwargs["seed"]),
        ),
    )
    training_kwargs.setdefault("per_device_train_batch_size", 2)
    training_kwargs.setdefault("gradient_accumulation_steps", 8)
    training_kwargs.setdefault("learning_rate", 1e-4)
    training_kwargs.setdefault("num_train_epochs", 20)
    training_kwargs.setdefault("gradient_checkpointing", False)
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

    # Build trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
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

    # If single language, set it in the decoder before testing
    if len(locales) == 1:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=locales[0], task="transcribe"
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
        type=json.loads,
        help="`prepare_common_voice` keyword arguments in JSON format (see common_voice_prepare.py)",
    )
    parser.add_argument(
        "-c",
        "--training_kwargs",
        default="{}",
        type=json.loads,
        help=(
            "training keyword arguments in JSON format "
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
