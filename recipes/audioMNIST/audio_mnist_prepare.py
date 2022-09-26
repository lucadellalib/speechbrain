"""
Downloads and creates dataset manifest files for audioMNIST (digit recognition).
For digit recognition, different digits of different speakers should appear in
train, validation, and test sets. In this case, these sets are thus derived from
splitting the original set into three chunks.

Author
 * Luca Della Libera, 2022
"""

import json
import logging
import os
import shutil

from sklearn.model_selection import train_test_split

from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.data_utils import download_file, get_all_files


__all__ = ["prepare_audio_mnist"]


_LOGGER = logging.getLogger(__name__)

_AUDIO_MNIST_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/tags/v1.0.10.tar.gz"

_SAMPLE_RATE = 8000


def prepare_audio_mnist(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    valid_ratio=0.10,
):
    """Prepares the JSON files for the audioMNIST dataset.

    Downloads the dataset if it is not found in `data_folder`.

    Arguments
    ---------
    data_folder : str
        The path to the directory where the audioMNIST dataset is stored.
    save_json_train : str
        The path to the train dataset specification JSON file will be saved.
    save_json_valid : str
        The path to the validation dataset specification JSON file will be saved.
    save_json_test : str
        The path to the test dataset specification JSON file will be saved.
    valid_ratio : str
        The ratio of the original training set to use for validation
        (the test set is fixed, see https://github.com/Jakobovski/free-spoken-digit-dataset/blob/v1.0.10/utils/train-test-split.py)

    Examples
    --------
    >>> data_folder = "/path/to/audioMNIST"
    >>> prepare_audio_mnist(data_folder, "train.json", "valid.json", "test.json")

    """

    # Check if this phase is already done (if so, skip it)
    if all(
        os.path.exists(p)
        for p in [save_json_train, save_json_valid, save_json_test]
    ):
        _LOGGER.info("Preparation completed in previous run, skipping.")
        return

    # If the dataset doesn't exist yet, download it
    if not os.path.exists(data_folder):
        _download_audio_mnist(data_folder)

    # List files and create manifest from list
    _LOGGER.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    wav_files = get_all_files(data_folder, match_and=[".wav"])

    # Random split the signal list into train, valid, and test sets
    train_valid_X, train_valid_y = [], []
    test_X = []
    for wav_file in wav_files:
        label, _, idx = wav_file[:-4].split(
            "_"
        )  # {digitLabel}_{speakerName}_{index}.wav
        if int(idx) <= 4:
            test_X.append(wav_file)
        else:
            train_valid_X.append(wav_file)
            train_valid_y.append(label)

    # Train-validation split in a stratified fashion
    train_X, valid_X = train_test_split(
        train_valid_X,
        test_size=valid_ratio,
        random_state=42,
        stratify=train_valid_y,
    )

    # Create JSON files
    _create_json(train_X, save_json_train)
    _create_json(valid_X, save_json_valid)
    _create_json(test_X, save_json_test)


def _download_audio_mnist(destination):
    """Downloads audioMNIST dataset.

    Arguments
    ---------
    destination : str
        The path to the destination directory.

    """
    archive = os.path.join(
        destination, "free-spoken-digit-dataset-1.0.10.tar.gz"
    )
    download_file(_AUDIO_MNIST_URL, archive)
    shutil.unpack_archive(archive, destination)
    shutil.copytree(
        os.path.join(archive.replace(".tar.gz", ""), "recordings"),
        os.path.join(destination, "data"),
    )
    shutil.rmtree(archive.replace(".tar.gz", ""))


def _create_json(wav_files, json_file):
    """Creates the JSON file given a sequence of WAV files.

    Arguments
    ---------
    wav_files : list[str]
        The sequence of paths to the WAV files.
    json_file : str
        The path to the JSON file.

    """
    json_dict = {}
    for wav_file in wav_files:
        # Read the signal (to retrieve length in seconds)
        signal = read_audio(wav_file)
        length = signal.shape[0] / _SAMPLE_RATE

        # Manipulate path to get utt_id, label, spk_id, and relative path
        wav_filename = os.path.basename(wav_file)
        utt_id = wav_filename[:-4]
        label, spk_id, _ = utt_id.split(
            "_"
        )  # {digitLabel}_{speakerName}_{index}.wav
        relative_path = os.path.join("{data_root}", "data", wav_filename)

        # Create entry for this utterance
        json_dict[utt_id] = {
            "wav": relative_path,
            "length": length,
            "spk_id": spk_id,
            "label": label,
        }

    # Write the dictionary to the JSON file
    with open(json_file, mode="w") as f:
        json.dump(json_dict, f, indent=2)

    _LOGGER.info(f"{json_file} successfully created!")
