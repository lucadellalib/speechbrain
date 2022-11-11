import os
import json
import torch
import logging
import whisper
import torchaudio
import numpy as np
from collections import defaultdict
import torchaudio.transforms as at
from common_voice_prepare import prepare_common_voice  # noqa
from datasets import load_dataset

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '[%(asctime)s - %(funcName)12s() ] >>> %(message)s',
    '%H:%M'
))
logger.addHandler(handler)
logger.setLevel(logging.INFO)



def load_manifests(dataset_size,dataset_dir,manifest_dir,lang):
    locales=[lang]
    prepare_common_voice(
        dataset_size,
        dataset_dir=dataset_dir,
        manifest_dir=manifest_dir,
        locales=locales,)


    manifests = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(manifest_dir, "train.csv"),
            "dev": os.path.join(manifest_dir, "dev.csv"),
            "test": os.path.join(manifest_dir, "test.csv"),
        },
    )

    return manifests


def load_wave(wave_path, sample_rate=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


class CommonVoiceDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, tokenizer) -> None:
        super().__init__()
        self.manifest = manifest
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, id):
        d = self.manifest[id]
        audio_filepath, text = d['mp3'], d['sentence']
        sig = load_wave(audio_filepath, sample_rate=16000)
        sig = whisper.pad_or_trim(sig.flatten())
        mel = whisper.log_mel_spectrogram(sig)

        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]

        return {
            'input_ids': mel,
            'labels': labels,
            'dec_input_ids': text,
            'audio_filepath': audio_filepath
        }



class WhisperDataCollatorWithPadding:
    def __call__(self, features: dict):
        input_ids, labels, dec_input_ids, audio_filepaths = [], [], [], []
        for f in features:
            audio_filepaths.append(f['audio_filepath'])
            input_ids.append(f['input_ids'])
            labels.append(f['labels'])
            dec_input_ids.append(f['dec_input_ids'])

        input_ids = torch.concat([input_ids[None, :] for input_ids in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_lengths = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_lengths)

        labels = [
            np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100)
            for lab, lab_len in zip(labels, label_lengths)
        ]
        dec_input_ids = [
            np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257)
            for e, e_len in zip(dec_input_ids, dec_input_ids_lengths)
        ]

        batch = {
            'labels': labels,
            'dec_input_ids': dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch['input_ids'] = input_ids
        batch['audio_filepaths'] = audio_filepaths

        return batch
