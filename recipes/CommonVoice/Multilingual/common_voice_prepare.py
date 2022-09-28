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

import csv
import logging
import os
import re
import shutil
import unicodedata
from datetime import datetime
from logging import StreamHandler
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Sequence, Tuple

import datasets
import torchaudio


__all__ = [
    "prepare_common_voice",
]


_BASENAME = os.path.basename(__file__).replace(".py", "")

_LOG_DIR = os.path.join(
    "logs", _BASENAME, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)

_LOGGER = logging.getLogger(__name__)

_MAX_DURATIONS_S = {
    "small": {"train": 1 * 60 * 60, "dev": 15 * 60, "test": 15 * 60},
    "medium": {"train": 10 * 60 * 60, "dev": 15 * 60, "test": 15 * 60},
    "large": {"train": None, "dev": 15 * 60, "test": 15 * 60},
    "full": {"train": None, "dev": None, "test": None},
}

os.makedirs(_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(funcName)s - %(message)s",
    handlers=[
        StreamHandler(),
        RotatingFileHandler(
            os.path.join(_LOG_DIR, f"{_BASENAME}.log"),
            maxBytes=512 * 1024,
            backupCount=100,
        ),
    ],
)


def prepare_common_voice(
    dataset_size: "str",
    dataset_version: "str" = "10_0",
    dataset_dir: "Optional[str]" = None,
    manifest_dir: "Optional[str]" = None,
    remove_accents: "bool" = True,
) -> "None":
    """Prepare the data manifest CSV files for Common Voice dataset
    (see https://commonvoice.mozilla.org/en/datasets).

    Parameters
    ----------
    dataset_size:
        The Common Voice dataset size. Must be one of the following:
        - "small":   1h train, 15 min dev, 15 min test;
        - "medium": 10h train, 15 min dev, 15 min test;
        - "large": full train, 15 min dev, 15 min test;
        - "full":  full train,   full dev,   full test.
    dataset_version:
        The Common Voice dataset version.
    dataset_dir:
        The path to the Common Voice dataset directory.
        If empty, the dataset (~510 GB) is downloaded from Hugging Face Hub
        (requires a Hugging Face account, see https://huggingface.co/docs/huggingface_hub/quick-start).
        Default to ``f"data/common_voice_{dataset_version}"``.
    manifest_dir:
        The path to the directory where the data manifest CSV files
        (and intermediate TSV files) are stored. If empty, the data
        manifest CSV files (and intermediate TSV files) are created
        on the fly.
        Default to ``f"{dataset_dir}_{dataset_size}".
    remove_accents:
        True to transform accented letters to the closest
        corresponding non-accented letters, False otherwise.

    Examples
    --------
    >>> prepare_common_voice("small", "10_0")

    """
    if dataset_size not in _MAX_DURATIONS_S:
        raise ValueError(
            f"`dataset_size` ({dataset_size}) must be one of {list(_MAX_DURATIONS_S)}"
        )

    dataset_name = f"mozilla-foundation/common_voice_{dataset_version}"
    if dataset_dir is None:
        dataset_dir = os.path.join("data", f"common_voice_{dataset_version}")
    if manifest_dir is None:
        manifest_dir = f"{dataset_dir}_{dataset_size}"

    # Get dataset metadata from Hugging Face Hub
    locales = datasets.get_dataset_config_names(
        dataset_name, use_auth_token=True
    )
    locales = ["ro", "ar"]

    output_tsv_files = []
    for i, locale in enumerate(locales):
        _LOGGER.log(logging.INFO, "----------------------------------------------------------------------")
        _LOGGER.log(logging.INFO, f"Locale: {locale}")
        locale_dir = os.path.join(dataset_dir, locale)
        if not os.path.isdir(locale_dir):
            _LOGGER.log(logging.INFO, "Downloading dataset...")
            cache_dir = os.path.join(dataset_dir, "cache")
            datasets.load_dataset(
                dataset_name, locale, cache_dir=cache_dir, use_auth_token=True
            )
            extracted_dir = os.path.join(cache_dir, "downloads", "extracted")
            extracted_dir = os.path.join(
                extracted_dir, os.listdir(extracted_dir)[0]
            )
            extracted_dir = os.path.join(
                extracted_dir, os.listdir(extracted_dir)[0]
            )
            shutil.move(
                os.path.join(extracted_dir, locale), locale_dir,
            )
            shutil.rmtree(cache_dir)
        else:
            _LOGGER.log(logging.INFO, "Dataset already downloaded")

        max_duration_s = _MAX_DURATIONS_S[dataset_size]

        _LOGGER.log(logging.INFO, f"Building subdataset {dataset_size}...")
        input_train_tsv_file = os.path.join(dataset_dir, locale, "train.tsv")
        output_train_tsv_file = os.path.join(
            manifest_dir, locale, f"train.tsv"
        )
        if not os.path.isfile(output_train_tsv_file):
            _LOGGER.log(logging.INFO, f"Creating {output_train_tsv_file}...")
            trim_tsv_file(
                input_train_tsv_file,
                output_train_tsv_file,
                max_duration_s["train"],
                remove_accents,
            )
        else:
            _LOGGER.log(
                logging.INFO, f"{output_train_tsv_file} already created"
            )

        input_dev_tsv_file = os.path.join(dataset_dir, locale, "dev.tsv")
        output_dev_tsv_file = os.path.join(
            manifest_dir, locale, f"dev.tsv"
        )
        if not os.path.isfile(output_dev_tsv_file):
            _LOGGER.log(logging.INFO, f"Creating {output_dev_tsv_file}...")
            trim_tsv_file(
                input_dev_tsv_file,
                output_dev_tsv_file,
                max_duration_s["dev"],
                remove_accents,
            )
        else:
            _LOGGER.log(logging.INFO, f"{output_dev_tsv_file} already created")

        input_test_tsv_file = os.path.join(dataset_dir, locale, "test.tsv")
        output_test_tsv_file = os.path.join(
            manifest_dir, locale, f"test.tsv"
        )
        if not os.path.isfile(output_test_tsv_file):
            _LOGGER.log(logging.INFO, f"Creating {output_test_tsv_file}...")
            trim_tsv_file(
                input_test_tsv_file,
                output_test_tsv_file,
                max_duration_s["test"],
                remove_accents,
            )
        else:
            _LOGGER.log(logging.INFO, f"{output_test_tsv_file} already created")

        # Cleanup if at least one of the 3 TSV files was
        # not created due to insufficient total duration
        output_tsv_files += [
            output_train_tsv_file,
            output_dev_tsv_file,
            output_test_tsv_file,
        ]
        if not all(os.path.isfile(file) for file in output_tsv_files[-3:]):
            _LOGGER.log(logging.INFO, "Cleaning up...")
            for file in output_tsv_files[-3:]:
                try:
                    os.remove(file)
                    _LOGGER.log(logging.INFO, f"Removing {file}...")
                except Exception:
                    pass
            for _ in range(3):
                output_tsv_files.pop(-1)

    _LOGGER.log(logging.INFO, f"Creating data manifest CSV files...")

    train_csv_file = os.path.join(manifest_dir, f"train.csv")
    if not os.path.isfile(train_csv_file):
        _LOGGER.log(logging.INFO, f"Creating {train_csv_file}...")
        merge_tsv_files(
            [f for f in output_tsv_files if "train" in f], train_csv_file
        )
    else:
        _LOGGER.log(logging.INFO, f"{train_csv_file} already created")

    dev_csv_file = os.path.join(manifest_dir, f"dev.csv")
    if not os.path.isfile(dev_csv_file):
        _LOGGER.log(logging.INFO, f"Creating {dev_csv_file}...")
        merge_tsv_files(
            [f for f in output_tsv_files if "dev" in f], dev_csv_file
        )
    else:
        _LOGGER.log(logging.INFO, f"{dev_csv_file} already created")

    test_csv_file = os.path.join(manifest_dir, f"test.csv")
    if not os.path.isfile(test_csv_file):
        _LOGGER.log(logging.INFO, f"Creating {test_csv_file}...")
        merge_tsv_files(
            [f for f in output_tsv_files if "test" in f], test_csv_file
        )
    else:
        _LOGGER.log(logging.INFO, f"{test_csv_file} already created")

    _LOGGER.log(logging.INFO, "Done!")


# Cache file: durations_s, total_duration_s to improve performance
# when `prepare_common_voice` is called multiple times
_TRIM_TSV_FILE_CACHE: "Dict[str, Tuple[List[float], float]]" = {}


# Adapted from:
# https://github.com/speechbrain/speechbrain/blob/v0.5.13/recipes/CommonVoice/common_voice_prepare.py
def trim_tsv_file(
    input_tsv_file: "str",
    output_tsv_file: "str",
    max_total_duration_s: "Optional[float]",
    remove_accents: "bool" = True,
) -> "None":
    """Deterministically remove rows from an input TSV file until
    `total_duration_s` <= `max_total_duration_s`, where `total_duration_s`
    is the sum of the durations in seconds of the listed audio clips (one for each row).

    Parameters
    ----------
    input_tsv_file:
        The path to the input TSV file.
    output_tsv_file:
        The path to the output TSV file.
    max_total_duration_s:
        The maximum total duration in seconds.
        Default to `total_duration_s`.
    remove_accents:
        True to transform accented letters to the closest
        corresponding non-accented letters, False otherwise.

    Examples
    --------
    >>> trim_tsv_file("data/common_voice_10_0/en/test.tsv", "data/common_voice_10_0_small/en/test.tsv", 15 * 60)

    """
    # Setting backend to sox-io (needed to read MP3 files)
    if torchaudio.get_audio_backend() != "sox_io":
        torchaudio.set_audio_backend("sox_io")

    # Header: client_id path sentence up_votes down_votes age gender accents locale segment
    _LOGGER.log(logging.INFO, f"Reading input TSV file ({input_tsv_file})...")
    try:
        durations_s, total_duration_s = _TRIM_TSV_FILE_CACHE[input_tsv_file]
    except KeyError:
        with open(input_tsv_file) as f:
            tsv_reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            _ = next(tsv_reader)
            durations_s = []
            total_duration_s = 0.0
            for row in tsv_reader:
                # NOTE: info returns incorrect num_frames on torchaudio>0.11.0
                clip_file = os.path.join(
                    os.path.dirname(input_tsv_file), "clips", row[1]
                )
                info = torchaudio.info(clip_file)
                duration_s = info.num_frames / info.sample_rate
                durations_s.append(duration_s)
                total_duration_s += duration_s
                _LOGGER.log(
                    logging.DEBUG,
                    f"Reading {clip_file} (duration: {duration_s})...",
                )
            _TRIM_TSV_FILE_CACHE[input_tsv_file] = durations_s, total_duration_s

    if max_total_duration_s is None:
        max_total_duration_s = total_duration_s
    if total_duration_s < max_total_duration_s:
        _LOGGER.log(
            logging.INFO,
            f"Too few data, skipping (total_duration_s: {total_duration_s})...",
        )
        return

    _LOGGER.log(
        logging.INFO,
        f"Total duration in seconds (before trimming): {total_duration_s}",
    )
    num_removed_rows = 0
    while total_duration_s > max_total_duration_s:
        duration_s = durations_s[num_removed_rows]
        total_duration_s -= duration_s
        num_removed_rows += 1
        _LOGGER.log(logging.DEBUG, f"Removing row {num_removed_rows}...")
    _LOGGER.log(logging.INFO, f"Removed {num_removed_rows} rows")
    _LOGGER.log(
        logging.INFO,
        f"Total duration in seconds (after trimming): {total_duration_s}",
    )

    _LOGGER.log(logging.INFO, f"Writing output TSV file ({output_tsv_file})...")
    os.makedirs(os.path.dirname(output_tsv_file), exist_ok=True)
    with open(input_tsv_file) as fr, open(output_tsv_file, "w") as fw:
        tsv_reader = csv.reader(fr, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(tsv_reader)
        # Rename "path" and "sentence" fields
        header[1], header[2] = "mp3", "wrd"
        tsv_writer = csv.writer(fw, delimiter="\t")
        # Add "ID" and "duration" fields
        tsv_writer.writerow(["ID"] + header + ["duration"])
        for i, row in enumerate(tsv_reader):
            if i < num_removed_rows:
                continue

            sentence = row[2]
            locale = row[8]
            sentence_id = clip_file = row[1]
            clip_file = os.path.join("$root_dir", locale, "clips", clip_file)
            # !! Language specific cleaning !!
            # Important: feel free to specify the text
            # normalization corresponding to your alphabet
            if locale in ["en", "fr", "it", "rw"]:
                sentence = re.sub(
                    "[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", sentence
                ).upper()

            if locale == "fr":
                # Replace J'y D'hui etc by J_ D_hui
                sentence = sentence.replace("'", " ")
                sentence = sentence.replace("’", " ")
            elif locale == "ar":
                HAMZA = "\u0621"
                ALEF_MADDA = "\u0622"
                ALEF_HAMZA_ABOVE = "\u0623"
                letters = (
                    "ابتةثجحخدذرزسشصضطظعغفقكلمنهويىءآأؤإئ"
                    + HAMZA
                    + ALEF_MADDA
                    + ALEF_HAMZA_ABOVE
                )
                sentence = re.sub("[^" + letters + " ]+", "", sentence).upper()
            elif locale == "ga-IE":
                # Irish lower() is complicated, but upper() is nondeterministic, so use lowercase
                def pfxuc(a: "str") -> "bool":
                    return len(a) >= 2 and a[0] in "tn" and a[1] in "AEIOUÁÉÍÓÚ"

                def galc(w: "str") -> "str":
                    return (
                        w.lower()
                        if not pfxuc(w)
                        else w[0] + "-" + w[1:].lower()
                    )

                sentence = re.sub("[^-A-Za-z'ÁÉÍÓÚáéíóú]+", " ", sentence)
                sentence = " ".join(map(galc, sentence.split(" ")))

            # Remove accents if specified
            if remove_accents:
                sentence = unicodedata.normalize("NFD", sentence)
                sentence = sentence.replace("'", " ")
                sentence = sentence.replace("’", " ")

            # Remove multiple spaces
            sentence = re.sub(" +", " ", sentence)

            # Remove spaces at the beginning and the end of the sentence
            sentence = sentence.lstrip().rstrip()

            # Remove too short sentences (or empty)
            if len(sentence.split(" ")) < 3:
                _LOGGER.log(
                    logging.DEBUG,
                    f"Sentence for row {i + 1} is too short, removing...",
                )
                continue

            row[1], row[2] = clip_file, sentence
            tsv_writer.writerow([sentence_id] + row + [durations_s[i]])

    _LOGGER.log(logging.INFO, "Done!")


def merge_tsv_files(
    input_tsv_files: "Sequence[str]", output_csv_file: "str",
) -> "None":
    """Merge input TSV files into a single output CSV file.

    Parameters
    ----------
    input_tsv_files:
        The paths to the input TSV files.
    output_csv_file:
        The path to the output CSV file.

    Examples
    --------
    >>> merge_tsv_files(
    >>>     ["data/common_voice_10_0_small/en/test.tsv", "data/common_voice_10_0_small/fa/test.tsv"],
    >>>     "data/common_voice_10_0_small/test.csv",
    >>> )

    """
    _LOGGER.log(logging.INFO, f"Writing output CSV file ({output_csv_file})...")
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
    num_clips, total_duration_s = 0, 0.0
    with open(output_csv_file, "w") as fw:
        csv_writer = csv.writer(fw)
        write_header = True
        for input_tsv_file in input_tsv_files:
            # Header: ID client_id mp3 wrd up_votes down_votes age gender accents locale segment duration
            _LOGGER.log(
                logging.INFO, f"Reading input TSV file ({input_tsv_file})..."
            )
            with open(input_tsv_file) as fr:
                tsv_reader = csv.reader(
                    fr, delimiter="\t", quoting=csv.QUOTE_NONE
                )
                header = next(tsv_reader)
                if write_header:
                    csv_writer.writerow(header)
                    write_header = False
                for row in tsv_reader:
                    num_clips += 1
                    total_duration_s += float(row[-1])
                    csv_writer.writerow(row)

    with open(output_csv_file.replace(".csv", ".stats"), "w") as fw:
        fw.write(f"Number of samples: {num_clips}\n")
        fw.write(f"Total duration in seconds: {total_duration_s}")

    _LOGGER.log(logging.INFO, "Done!")


if __name__ == "__main__":
    prepare_common_voice("small")
    prepare_common_voice("medium")
    prepare_common_voice("large")
    prepare_common_voice("full")
