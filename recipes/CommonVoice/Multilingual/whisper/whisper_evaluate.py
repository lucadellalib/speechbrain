from datasets import load_dataset,Audio
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor,WhisperFeatureExtractor
import soundfile as sf
import torch
from jiwer import wer,cer
import pandas as pd
import argparse


from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch



def resolve_root_dir(sample: "Dict[str, Any]") -> "Dict[str, Any]":
    sample["mp3"] = sample["mp3"].replace("$root_dir", manifest_dir)
    return sample

def map_to_pred(batch):
    audio = batch["audio"]
    # batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    batch['labels'] = processor.tokenizer._normalize(batch['sentence'])
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens = True,normalize = True)
    batch["transcription"] = transcription[0]
    return batch

def evaluate_whisper(dataset_size,locales):
    # Dataset preparation (parsing CommonVoice)

    from common_voice_prepare import prepare_common_voice  # noqa
    prepare_common_voice(
        dataset_size,
        dataset_dir=dataset_dir,
        manifest_dir=manifest_dir,
        locales=locales,)


    common_voice_data = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(manifest_dir, "train.csv"),
            "dev": os.path.join(manifest_dir, "dev.csv"),
            "test": os.path.join(manifest_dir, "test.csv"),
        },
    )


# 


    for locale in locales:
        common_voice_data = common_voice_data.map(resolve_root_dir, num_proc=4)
        lan=None
        if f"<|{locale}|>"  in processor.tokenizer.additional_special_tokens:
            model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=locale, task = "transcribe")
        common_voice_data = common_voice_data.rename_columns({"mp3": "audio", "wrd": "sentence"})
        common_voice_data = common_voice_data.cast_column("audio", Audio(sampling_rate=16000))
        result = common_voice_data['test'].map(map_to_pred, remove_columns=common_voice_data['test'].column_names,num_proc=1)
        print(f"Error rates for language: {locale}, size: {dataset_size} and whisper model: {args.whisper_model}")
        print("WER:", wer(result["labels"], result["transcription"]))
        print("CER:", cer(result["labels"], result["transcription"]))



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
         default=None,
        help="dataset size",
    )
    parser.add_argument(
        "-i",
        "--dataset_dir",
        default=None,
        help="path to the dataset directory",
    )
    parser.add_argument(
        "-l",
        "--locales",
        nargs="+",
        default=None,
        help="locales to include (e.g. 'en', 'it', etc.), default to all the locales in Common Voice 10.0",
    )

    args = parser.parse_args()
    # load model and processor
    processor = WhisperProcessor.from_pretrained(args.whisper_model)
    model = WhisperForConditionalGeneration.from_pretrained(args.whisper_model)
    dataset_dir=args.dataset_dir
    manifest_dir = f"{args.dataset_dir}_{args.dataset_size}"
    evaluate_whisper(
        args.dataset_size,
        args.locales
    )
