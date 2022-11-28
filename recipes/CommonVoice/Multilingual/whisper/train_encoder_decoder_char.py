#!/usr/bin/env python3

"""Recipe for fine-tuning Whisper encoder-decoder with character vocabulary for ASR on Common Voice.

To run this recipe, do the following:
> python train.py hparams/<path-to-config>.yaml

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split, and many other possible variations.

Authors
 * Luca Della Libera 2022
"""

import json
import os
import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main


class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, tokens_bos_lens = batch.tokens_bos

        if stage == sb.Stage.TRAIN:
            # Add augmentation if specified
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        tokens_bos = torch.nn.functional.embedding(
            tokens_bos,
            self.modules.proj_lin.w.weight,
            padding_idx=self.hparams.pad_index,
        )
        encoder_out = self.modules.whisper.forward_encoder(wavs)
        decoder_out = self.modules.whisper.forward_decoder(
            encoder_out, tokens_bos
        )
        logits = self.modules.proj_lin(decoder_out)
        log_probs = self.hparams.log_softmax(logits)

        hyps = None
        if stage != sb.Stage.TRAIN:
            hyps = self._greedy_decode(encoder_out.detach())

        return log_probs, wav_lens, hyps

    @torch.no_grad()
    def _greedy_decode(self, encoder_out):
        batch_size = encoder_out.shape[0]
        startoftranscript_id = self.hparams.bos_index
        pad_id = self.hparams.pad_index
        endoftext_id = self.hparams.eos_index

        hyps = torch.full(
            (batch_size, self.hparams.max_gen_tokens + 1),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )

        # Prepare initial tokens in the right format
        hyps[:, 0] = startoftranscript_id

        # Autoregressive loop
        num_gen_tokens = 0
        unfinished_mask = torch.ones(
            len(hyps), dtype=torch.bool, device=self.device
        )
        while (
            hyps[unfinished_mask, num_gen_tokens + 3] != endoftext_id
        ).any() and (num_gen_tokens < self.hparams.max_gen_tokens):
            decoder_out = self.modules.whisper.forward_decoder(
                encoder_out[unfinished_mask],
                hyps[unfinished_mask, : num_gen_tokens + 4],
            )
            logits = (
                decoder_out
                @ self.modules.whisper.model.decoder.embed_tokens.weight.T
            )
            gen_tokens = logits.argmax(dim=-1)[:, -1]
            hyps[unfinished_mask, num_gen_tokens + 4] = gen_tokens
            unfinished_mask[unfinished_mask == True] = (
                gen_tokens != endoftext_id
            )
            num_gen_tokens += 1
        return hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (NLL) given predictions and targets."""
        log_probs, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens_eos, tokens_eos_lens = (
            tokens_eos.to(self.device),
            tokens_eos_lens.to(self.device),
        )

        # Loss computation: use left-shifted labels as reference
        loss = self.hparams.nll_loss(
            log_probs,
            tokens_eos,
            tokens_eos_lens,
            allowed_len_diff=float("inf"),
        )

        if stage != sb.Stage.TRAIN:
            tokens, tokens_lens = batch.tokens

            # Convert predicted tokens to words
            predicted_words = self.tokenizer.batch_decode(
                predicted_tokens, skip_special_tokens=True
            )

            # Convert target tokens to words
            tokens = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer.batch_decode(
                tokens, skip_special_tokens=True
            )

            predicted_words = [
                self.english_normalizer(w) for w in predicted_words
            ]
            target_words = [self.english_normalizer(w) for w in target_words]

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        if self.auto_mix_prec:
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            if self.check_gradients(loss):
                self.scaler.step(self.optimizer)

            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            loss.backward()

            if self.check_gradients(loss):
                self.optimizer.step()

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
            with open(
                os.path.join(os.path.dirname(__file__), "english.json")
            ) as f:
                english_spelling_mapping = json.load(f)
            self.english_normalizer = self.hparams.english_normalizer(
                english_spelling_mapping
            )

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
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            stats_meta_data = {
                "epoch": epoch,
                "lr": old_lr,
            }

            self.hparams.train_logger.log_stats(
                stats_meta=(stats_meta_data),
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
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "tokens", "tokens_bos", "tokens_eos"],
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

    # No tokens are forced as decoder outputs, no tokens are suppressed during generation
    hparams["modules"]["whisper"].model.config.forced_language_id = None
    hparams["modules"]["whisper"].model.config.suppress_tokens = []

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
