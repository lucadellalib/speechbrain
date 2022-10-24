#!/usr/bin/env python3

"""Recipe for training a connectionist temporal classification ASR system with Common Voice.

To run this recipe, do the following:
> python train.py hparams/<path-to-config>.yaml

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split, and many other possible variations.

Authors
 * Titouan Parcollet 2021
 * Luca Della Libera 2022
"""

import os
import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        if hasattr(self.modules, "wav2vec2"):
            # Add augmentation if specified
            if stage == sb.Stage.TRAIN:
                if hasattr(self.hparams, "augmentation"):
                    wavs = self.hparams.augmentation(wavs, wav_lens)

            # Forward pass
            feats = self.modules.wav2vec2(wavs)
        else:
            # Forward pass
            feats = self.hparams.compute_features(wavs)
            feats = self.modules.normalize(feats, wav_lens)

            # Add augmentation if specified
            if stage == sb.Stage.TRAIN:
                if hasattr(self.hparams, "augmentation"):
                    feats = self.hparams.augmentation(feats)

        x = self.modules.enc(feats)
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC) given predictions and targets."""

        p_ctc, wav_lens = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens

        loss = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )

            predicted_words = self.tokenizer(sequence, task="decode_from_list")

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        if self.auto_mix_prec:

            if hasattr(self.hparams, "wav2vec2"):
                if not self.hparams.wav2vec2.freeze:
                    self.optimizer_wav2vec2.zero_grad()
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            if hasattr(self.hparams, "wav2vec2"):
                if not self.hparams.wav2vec2.freeze:
                    self.scaler.unscale_(self.optimizer_wav2vec2)
            self.scaler.unscale_(self.optimizer)

            if self.check_gradients(loss):
                if hasattr(self.hparams, "wav2vec2"):
                    if not self.hparams.wav2vec2.freeze:
                        self.scaler.step(self.optimizer_wav2vec2)
                self.scaler.step(self.optimizer)

            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            loss.backward()

            if self.check_gradients(loss):
                if hasattr(self.hparams, "wav2vec2"):
                    if not self.hparams.wav2vec2.freeze:
                        self.optimizer_wav2vec2.step()
                self.optimizer.step()

            if hasattr(self.hparams, "wav2vec2"):
                if not self.hparams.wav2vec2.freeze:
                    self.optimizer_wav2vec2.zero_grad()
            self.optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            if hasattr(self.hparams, "wav2vec2"):
                (
                    old_lr_wav2vec2,
                    new_lr_wav2vec2,
                ) = self.hparams.lr_annealing_wav2vec2(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            if hasattr(self.hparams, "wav2vec2"):
                if not self.hparams.wav2vec2.freeze:
                    sb.nnet.schedulers.update_learning_rate(
                        self.optimizer_wav2vec2, new_lr_wav2vec2
                    )

            self.hparams.train_logger.log_stats(
                stats_meta=(
                    {
                        "epoch": epoch,
                        "lr": old_lr,
                        "lr_wav2vec": old_lr_wav2vec2,
                    }
                    if hasattr(self.hparams, "wav2vec2")
                    else {"epoch": epoch, "lr": old_lr}
                ),
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def init_optimizers(self):
        """Initializes the wav2vec2 optimizer and model optimizer"""
        if hasattr(self.hparams, "wav2vec2"):
            if not self.hparams.wav2vec2.freeze:
                self.optimizer_wav2vec2 = self.hparams.opt_class_wav2vec2(
                    self.modules.wav2vec2.parameters()
                )
                if self.checkpointer is not None:
                    self.checkpointer.add_recoverable(
                        "optimizer_wav2vec2", self.optimizer_wav2vec2
                    )
        super().init_optimizers()


# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["manifest_dir"], "train.csv"),
        replacements={"root_dir": hparams["manifest_dir"]},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["manifest_dir"], "dev.csv"),
        replacements={"root_dir": hparams["manifest_dir"]},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["manifest_dir"], "test.csv"),
        replacements={"root_dir": hparams["manifest_dir"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("mp3")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(mp3):
        info = torchaudio.info(mp3)
        sig = sb.dataio.dataio.read_audio(mp3)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "tokens"],
    )
    return train_data, valid_data, test_data


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation (parsing CommonVoice)
    from common_voice_prepare import prepare_common_voice  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_dir"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Due to DDP, we do the preparation ONLY on the main python process
    run_on_main(
        prepare_common_voice,
        kwargs={
            "dataset_size": hparams["dataset_size"],
            "dataset_version": hparams["dataset_version"],
            "dataset_dir": hparams["dataset_dir"],
            "manifest_dir": hparams["manifest_dir"],
            "locales": hparams["locales"],
        },
    )

    # Defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_dir"],
        vocab_size=hparams["output_neurons"],
        annotation_train=os.path.join(hparams["manifest_dir"], "train.csv"),
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
    )

    # Here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        opt_class=hparams["opt_class"],
        checkpointer=hparams["checkpointer"],
    )

    # Adding objects to trainer:
    asr_brain.tokenizer = tokenizer

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    asr_brain.hparams.wer_file = hparams["output_dir"] + "/wer_test.txt"
    asr_brain.evaluate(
        test_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    asr_brain.hparams.wer_file = hparams["output_dir"] + "/wer_valid.txt"
    asr_brain.evaluate(
        valid_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
