from datasets import load_dataset,Audio
import os
import soundfile as sf
import torch
from jiwer import wer,cer
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm

import whisper

from common_voice_prepare import prepare_common_voice  # noqa

from datasets import load_dataset
from whisper.normalizers import EnglishTextNormalizer



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CommonVoice(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, dataset_size,dataset_dir,locales, device=DEVICE):
        self.manifest_dir= f"{args.dataset_dir}_{args.dataset_size}"
        prepare_common_voice(
            dataset_size,
            dataset_dir=dataset_dir,
            manifest_dir=self.manifest_dir,
            locales=locales,)


        ds  = load_dataset(
            "csv",
            data_files={
                "train": os.path.join(self.manifest_dir, "train.csv"),
                "dev": os.path.join(self.manifest_dir, "dev.csv"),
                "test": os.path.join(self.manifest_dir, "test.csv"),
            },
         )
        self.dataset = self.preprocess_dataset(ds)
        self.device = device
    
    
    def preprocess_dataset(self,dataset):
        dataset = dataset.remove_columns(
        [
            "ID",
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
        dataset = dataset.map(self.resolve_root_dir, num_proc=4)
        dataset = dataset.rename_columns({"mp3": "audio", "wrd": "sentence"})
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        return dataset
    
    def resolve_root_dir(self,sample: "Dict[str, Any]") -> "Dict[str, Any]":
        sample["mp3"] = sample["mp3"].replace("$root_dir", self.manifest_dir)
        return sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio =self.dataset[item]['audio']
        text =self.dataset[item]['sentence']
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        
        return (mel, text)

def inference_whisper(dataset_size,dataset_dir,locales):
    dataset = CommonVoice(dataset_size,dataset_dir,locales,)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    model = whisper.load_model(args.whisper_model)
    print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )
        # predict without timestamps for short-form transcription
    if locals != None:
        options = whisper.DecodingOptions(language=locals[0], without_timestamps=True)
    else:
        options = whisper.DecodingOptions(without_timestamps=True)
    
    hypotheses = []
    references = []

    for mels, texts in tqdm(loader):
        results = model.decode(mels, options)
        hypotheses.extend([result.text for result in results])
        references.extend(texts)
    
    normalizer = EnglishTextNormalizer()
    
    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]



    print(f"Error rates for language: {locales[0]}, size: {dataset_size} and whisper model: {args.whisper_model}")
    print(f"WER:", wer(list(data["reference_clean"]), list(data["hypothesis_clean"])))
    print(f"CER:", cer(list(data["reference_clean"]), list(data["hypothesis_clean"])))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Used pre-traned Whisper on Common Voice 10.0"
    )
    parser.add_argument(
        "whisper_model",
        help="path to a directory containing the Whisper model checkpoint",
    )
    parser.add_argument(
        "dataset_size",
        choices=["small", "medium", "large", "full"],
         default="small",
        help="dataset size",
    )
    parser.add_argument(
        "-i",
        "--dataset_dir",
        default="data/common_voice_10_0",
        help="path to the dataset directory",
    )
    parser.add_argument(
        "-l",
        "--locales",
        nargs="+",
        default=['rw'],
        help="locales to include (e.g. 'en', 'it', etc.), default to all the locales in Common Voice 10.0",
    )

    args = parser.parse_args()
    # load model and processor
    
    
    

