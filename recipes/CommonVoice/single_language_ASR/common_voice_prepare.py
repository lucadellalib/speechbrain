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

import os
import csv
import re
import logging
import torchaudio
import unicodedata
from tqdm.contrib import tzip

__all__ = [
    "prepare_common_voice",
]

logger = logging.getLogger(__name__)

# Random indices are not generated on the fly but statically read from a predefined
# file to avoid reproducibility issues on different platforms and/or Python versions
with open(os.path.join(os.path.dirname(__file__), "random_idxes.txt")) as f:
    _RANDOM_IDXES = [int(line) for line in f]

_MAX_DURATIONS_S = {
    "small": {"train": 1 * 60 * 60, "dev": 15 * 60, "test": 15 * 60},
    "medium": {"train": 10 * 60 * 60, "dev": 15 * 60, "test": 15 * 60},
    "large": {"train": None, "dev": 15 * 60, "test": 15 * 60},
    "full": {"train": None, "dev": None, "test": None},
}



def prepare_common_voice(
    dataset_size: "str",
    data_folder: "Optional[str]" = None,
    save_folder: "Optional[str]" = None,
    train_tsv_file=None,
    dev_tsv_file=None,
    test_tsv_file=None,
    accented_letters=False,
    dataset_version: "str" = "10_0",
    language="en",
    copy_clips: "bool" = True,
    debug: "bool" = False,
    skip_prep=False,
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
        If empty, the dataset (~510 GB, more if `copy_clips` is True)
        is downloaded from Hugging Face Hub (requires a Hugging Face
        account, see https://huggingface.co/docs/huggingface_hub/quick-start).
        Default to ``f"data/common_voice_{dataset_version}"``.
    manifest_dir:
        The path to the directory where the data manifest CSV files
        (and intermediate TSV files) are stored. If empty, the data
        manifest CSV files (and intermediate TSV files) are created
        on the fly.
        Default to ``f"{dataset_dir}_{dataset_size}".
    copy_clips:
        True to copy the original audio clips in `manifest_dir`,
        False otherwise.
    debug:
        True to enable debug mode, False otherwise.

    Examples
    --------
    >>> prepare_common_voice("small", "10_0")

    """
    if skip_prep:
        return

    # TODO: download data if not exist!!!!! & generate modified tsv file.
    
    if dataset_size not in _MAX_DURATIONS_S:
        raise ValueError(
            f"`dataset_size` ({dataset_size}) must be one of {list(_MAX_DURATIONS_S)}"
        )

    # If not specified point toward standard location w.r.t CommonVoice tree
    if train_tsv_file is None:
        train_tsv_file = data_folder + "/train.tsv"
    else:
        train_tsv_file = train_tsv_file

    if dev_tsv_file is None:
        dev_tsv_file = data_folder + "/dev.tsv"
    else:
        dev_tsv_file = dev_tsv_file

    if test_tsv_file is None:
        test_tsv_file = data_folder + "/test.tsv"
    else:
        test_tsv_file = test_tsv_file

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_csv_train = save_folder + "/train.csv"
    save_csv_dev = save_folder + "/dev.csv"
    save_csv_test = save_folder + "/test.csv"


    if skip(save_csv_train, save_csv_dev, save_csv_test):

        msg = "%s already exists, skipping data preparation!" % (save_csv_train)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_dev)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_test)
        logger.info(msg)
        return

    # Additional checks to make sure the data folder contains Common Voice
    check_commonvoice_folders(data_folder)

    # Creating csv files for {train, dev, test} data
    file_pairs = zip(
        [train_tsv_file, dev_tsv_file, test_tsv_file],
        [save_csv_train, save_csv_dev, save_csv_test],
    )
    for tsv_file, save_csv in file_pairs:
        create_csv(
            tsv_file, save_csv, data_folder, accented_letters, language,
        )


def create_csv(
    orig_tsv_file, csv_file, data_folder, accented_letters=False, language="en"
):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    orig_tsv_file : str
        Path to the Common Voice tsv file (standard file).
    data_folder : str
        Path of the CommonVoice dataset.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.

    Returns
    -------
    None
    """

    # Check if the given files exists
    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    loaded_csv = open(orig_tsv_file, "r").readlines()[1:]
    nb_samples = str(len(loaded_csv))

    msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    # Adding some Prints
    msg = "Creating csv lists in %s ..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav", "spk_id", "wrd"]]

    # Start processing lines
    total_duration = 0.0
    for line in tzip(loaded_csv):

        line = line[0]

        # Path is at indice 1 in Common Voice tsv files. And .mp3 files
        # are located in datasets/lang/clips/
        mp3_path = os.path.join( data_folder, "clips", line.split("\t")[1])
        file_name = mp3_path.split(".")[-2].split("/")[-1]
        spk_id = line.split("\t")[0]
        snt_id = file_name
        duration=float(line.split("\t")[-1].split("\n")[0])

        # # Setting torchaudio backend to sox-io (needed to read mp3 files)
        # if torchaudio.get_audio_backend() != "sox_io":
        #     logger.warning("This recipe needs the sox-io backend of torchaudio")
        #     logger.warning("The torchaudio backend is changed to sox_io")
        #     torchaudio.set_audio_backend("sox_io")

        # # Reading the signal (to retrieve duration in seconds)
        # if os.path.isfile(mp3_path):
        #     info = torchaudio.info(mp3_path,format="mp3")
        # else:
        #     msg = "\tError loading: %s" % (str(len(file_name)))
        #     logger.info(msg)
        #     continue

        # duration = info.num_frames / info.sample_rate
        total_duration += duration

        # Getting transcript
        words = line.split("\t")[2]

        # Unicode Normalization
        words = unicode_normalisation(words)

        # !! Language specific cleaning !!
        # Important: feel free to specify the text normalization
        # corresponding to your alphabet.

        if language in ["en", "fr", "it", "rw"]:
            words = re.sub(
                "[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", words
            ).upper()

        if language == "fr":
            # Replace J'y D'hui etc by J_ D_hui
            words = words.replace("'", " ")
            words = words.replace("’", " ")

        elif language == "ar":
            HAMZA = "\u0621"
            ALEF_MADDA = "\u0622"
            ALEF_HAMZA_ABOVE = "\u0623"
            letters = (
                "ابتةثجحخدذرزسشصضطظعغفقكلمنهويىءآأؤإئ"
                + HAMZA
                + ALEF_MADDA
                + ALEF_HAMZA_ABOVE
            )
            words = re.sub("[^" + letters + " ]+", "", words).upper()
        elif language == "ga-IE":
            # Irish lower() is complicated, but upper() is nondeterministic, so use lowercase
            def pfxuc(a):
                return len(a) >= 2 and a[0] in "tn" and a[1] in "AEIOUÁÉÍÓÚ"

            def galc(w):
                return w.lower() if not pfxuc(w) else w[0] + "-" + w[1:].lower()

            words = re.sub("[^-A-Za-z'ÁÉÍÓÚáéíóú]+", " ", words)
            words = " ".join(map(galc, words.split(" ")))

        # Remove accents if specified
        if not accented_letters:
            words = strip_accents(words)
            words = words.replace("'", " ")
            words = words.replace("’", " ")

        # Remove multiple spaces
        words = re.sub(" +", " ", words)

        # Remove spaces at the beginning and the end of the sentence
        words = words.lstrip().rstrip()

        # Getting chars
        chars = words.replace(" ", "_")
        chars = " ".join([char for char in chars][:])

        # Remove too short sentences (or empty):
        if len(words.split(" ")) < 3:
            continue

        # Composition of the csv_line
        csv_line = [snt_id, str(duration), mp3_path, spk_id, str(words)]

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

    # Writing the csv lines
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(len(loaded_csv)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)


def unicode_normalisation(text):

    try:
        text = unicode(text, "utf-8")
    except NameError:  # unicode is a default on python 3
        pass
    return str(text)

def strip_accents(text):

    text = (
        unicodedata.normalize("NFD", text)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    return str(text)


def skip(save_csv_train, save_csv_dev, save_csv_test):
    """
    Detects if the Common Voice data preparation has been already done.

    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = False

    if (
        os.path.isfile(save_csv_train)
        and os.path.isfile(save_csv_dev)
        and os.path.isfile(save_csv_test)
    ):
        skip = True

    return skip

def check_commonvoice_folders(data_folder):
    """
    Check if the data folder actually contains the Common Voice dataset.

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain Common Voice dataset.
    """

    files_str = "/clips"

    # Checking clips
    if not os.path.exists(data_folder + files_str):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the Common Voice dataset)" % (data_folder + files_str)
        )
        raise FileNotFoundError(err_msg)
