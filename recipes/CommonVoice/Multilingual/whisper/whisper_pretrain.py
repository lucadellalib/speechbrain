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
    A simple class to wrap commonVoice and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, dataset_size,dataset_dir,locales,split="test", device=DEVICE):
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
        self.dataset = self.preprocess_dataset(ds[split])
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
#             "locale",
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
        audio = torch.from_numpy(self.dataset[item]['audio']['array'])
        text =self.dataset[item]['sentence']
        locale =self.dataset[item]['locale']
        audio = whisper.pad_or_trim(audio).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        
        return (mel, text,locale)

def inference_whisper(dataset_size,dataset_dir,output_dir,locales):
    dataset = CommonVoice(dataset_size,dataset_dir,locales,)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    model = whisper.load_model(args.whisper_model)
    print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )
        # predict without timestamps for short-form transcription
    if locales != None :
        if locales[0] in whisper.tokenizer.LANGUAGES.keys():
            options = whisper.DecodingOptions(language=locales[0], without_timestamps=True)
        else:
            options = whisper.DecodingOptions(without_timestamps=True)
            print(f"{locales[0]} is not among supported languages in whisper.")
            
    else:
        options = whisper.DecodingOptions(without_timestamps=True)
      
    
    hypotheses = []
    references = []
    languages=[]

    for mels, texts,locales in tqdm(loader):
        results = model.decode(mels, options)
        hypotheses.extend([result.text for result in results])
        references.extend(texts)
        languages.extend(locales)
    
    normalizer = EnglishTextNormalizer()
    
    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references,language=languages))
    
    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]
    
    test_cer= 100*cer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
    test_wer=100*wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
    
    data.to_csv(os.path.join(output_dir, "test_result.csv"))

    print(f"WER:", test_wer)
    print(f"CER:",test_cer)
    
    lines = []
    line = (
        f"test CER: {test_cer:.2f}, test WER: {test_wer:.2f}"
    )
    lines.append(line)

    # Write log file
    with open(os.path.join(output_dir, "train_log.txt"), "w") as f:
        f.write("\n".join(lines))





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
        "-d",
        "--dataset_dir",
        default=None,
        help="path to Common Voice 10.0 dataset directory",
    )
    parser.add_argument(
        "-o", 
        "--output_dir", 
        default=None, help="path to the output directory",
    )

    parser.add_argument(
        "-l",
        "--locales",
        nargs="+",
        default=None,
        help="locales to include (e.g. 'en', 'it', etc.), default to all the locales in Common Voice 10.0",
    )

    args = parser.parse_args()
    inference_whisper(args.dataset_size,args.dataset_dir,args.output_dir,args.locales)
    # load model and processor
    
    
    

