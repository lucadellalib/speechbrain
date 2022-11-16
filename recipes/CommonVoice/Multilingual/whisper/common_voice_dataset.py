from datasets import load_dataset,Audio
import os
import soundfile as sf
import torch
from jiwer import wer,cer
import torchaudio
import logging


import whisper
import numpy as np
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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

__all__ = [
    "CommonVoice","WhisperDataCollatorWithPadding"
]

def load_manifests(dataset_size,dataset_dir,locales):
    manifest_dir= f"{dataset_dir}_{dataset_size}"
    prepare_common_voice(
            dataset_size,
            dataset_dir=dataset_dir,
            manifest_dir=manifest_dir,
            locales=locales)


    manifests  = load_dataset(
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
    """
    A simple class to wrap commonVoice and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, dataset_size,dataset_dir,manifests,tokenizer, device=DEVICE):
        self.manifest_dir= f"{dataset_dir}_{dataset_size}"
        self.tokenizer = tokenizer

        self.dataset = self.preprocess_dataset(manifests)
        self.device = device
    
    
    def preprocess_dataset(self,dataset):
        # dataset = dataset.remove_columns(
        # [
        #     "ID",
        #     "age",
        #     "client_id",
        #     "down_votes",
        #     "duration",
        #     "gender",
        #     "locale",
        #     "segment",
        #     "up_votes",
        # ]
        # )
        dataset = dataset.map(self.resolve_root_dir, num_proc=4)
        dataset = dataset.rename_columns({"mp3": "audio_filepath", "wrd": "sentence"})
        # dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        return dataset
    
    def resolve_root_dir(self,sample: "Dict[str, Any]") -> "Dict[str, Any]":
        sample["mp3"] = sample["mp3"].replace("$root_dir", self.manifest_dir)
        return sample

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, id):
        d = self.dataset[id]
        audio_filepath, text = d['audio_filepath'], d['sentence']
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




# locales=['en']
# wtokenizer = whisper.tokenizer.get_tokenizer(True, language=locales[0], task='transcribe')
# manifests= load_manifests('small','data/common_voice_10_0',locales)
# dataset = CommonVoiceDataset('small','data/common_voice_10_0',manifests,wtokenizer,split='dev')
    
# loader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=WhisperDataCollatorWithPadding())

# for items in loader:
#     print("ahan")

