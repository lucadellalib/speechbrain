#!/usr/bin/env python3

"""Recipe for training a Bayesian phoneme recognizer on TIMIT via Stein variational gradient
descent (https://arxiv.org/abs/1608.04471).
The system relies on an encoder, a decoder, and attention mechanisms between them.
Linear layers are turned into Bayesian linear layers by placing a normal prior upon their
weights and biases. The Bayesian neural network is trained to minimize a trade-off between
the simplicity of the prior (complexity loss) and the complexity of the data (likelihood loss).
The likelihood loss is the standard loss function used in non-Bayesian ASR (CTC + negative log
likelihood), the complexity loss is the repulsive force based on the RBF kernel described in the
SVGD paper. Greedy search is using for validation, while beamsearch is used at test time to improve
the system performance.

To run this recipe, do the following:
> python train_with_wav2vec2_svgd.py hparams/train_with_wav2vec2_svgd.yaml --data_folder /path/to/TIMIT

Authors
 * Luca Della Libera 2023
"""

import os
import sys

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        phns_bos, _ = batch.phn_encoded_bos

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.modules.wav2vec2(wavs, wav_lens)
        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        e_in = self.modules.emb(phns_bos)
        h, _ = self.modules.dec(e_in, x, wav_lens)

        # output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        if stage == sb.Stage.VALID:
            hyps, scores = self.hparams.greedy_searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        elif stage == sb.Stage.TEST:
            hyps, scores = self.hparams.beam_searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        return p_ctc, p_seq, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."
        if stage == sb.Stage.TRAIN:
            p_ctc, p_seq, wav_lens = predictions
        else:
            p_ctc, p_seq, wav_lens, hyps = predictions

        ids = batch.id
        phns_eos, phn_lens_eos = batch.phn_encoded_eos
        phns, phn_lens = batch.phn_encoded

        loss_ctc = self.hparams.ctc_cost(p_ctc, phns, wav_lens, phn_lens)
        loss_seq = self.hparams.seq_cost(p_seq, phns_eos, phn_lens_eos)
        loss = self.hparams.ctc_weight * loss_ctc
        loss += (1 - self.hparams.ctc_weight) * loss_seq
        loss -= self.hparams.log_prior_weight * (
            self.modules.ctc_lin.log_prior + self.modules.seq_lin.log_prior
        )

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            self.ctc_metrics.append(ids, p_ctc, phns, wav_lens, phn_lens)
            self.seq_metrics.append(ids, p_seq, phns_eos, phn_lens_eos)
            self.per_metrics.append(
                ids, hyps, phns, None, phn_lens, self.label_encoder.decode_ndim,
            )

        return loss

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        self.seq_metrics = self.hparams.seq_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr_adam, new_lr_adam = self.hparams.lr_annealing_adam(per)
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                per
            )
            sb.nnet.schedulers.update_learning_rate(
                self.adam_optimizer, new_lr_adam
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_adam": old_lr_adam,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": stage_loss,
                    "ctc_loss": self.ctc_metrics.summarize("average"),
                    "seq_loss": self.seq_metrics.summarize("average"),
                    "PER": per,
                },
            )
            self.checkpointer.save_and_keep_only(
                meta={"PER": per}, min_keys=["PER"]
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            with open(self.hparams.wer_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nseq2seq loss stats:\n")
                self.seq_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "CTC, seq2seq, and PER stats written to file",
                    self.hparams.wer_file,
                )

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        # Managing automatic mixed precision
        if self.auto_mix_prec:

            self.wav2vec_optimizer.zero_grad()
            self.adam_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            for preconditioner in self.hparams.preconditioners:
                self.scaler.unscale_(preconditioner)
            self.scaler.unscale_(self.wav2vec_optimizer)
            self.scaler.unscale_(self.adam_optimizer)

            if self.check_gradients(loss):
                for preconditioner in self.hparams.preconditioners:
                    self.scaler.step(preconditioner)
                self.scaler.step(self.wav2vec_optimizer)
                self.scaler.step(self.adam_optimizer)

            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            loss.backward()

            if self.check_gradients(loss):
                for preconditioner in self.hparams.preconditioners:
                    preconditioner.step()
                self.wav2vec_optimizer.step()
                self.adam_optimizer.step()

            self.wav2vec_optimizer.zero_grad()
            self.adam_optimizer.zero_grad()

        return loss.detach().cpu()

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.adam_optimizer = self.hparams.adam_opt_class(
            [
                *self.hparams.model.parameters(),
                *self.hparams.seq_lin.parameters(),
                *self.hparams.ctc_lin.parameters(),
            ]
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)

    def zero_grad(self, set_to_none=False):
        self.wav2vec_optimizer.zero_grad(set_to_none)
        self.adam_optimizer.zero_grad(set_to_none)


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides(
        "phn_list",
        "phn_encoded_list",
        "phn_encoded",
        "phn_encoded_eos",
        "phn_encoded_bos",
    )
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        yield phn_encoded_list
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded
        phn_encoded_eos = torch.LongTensor(
            label_encoder.append_eos_index(phn_encoded_list)
        )
        yield phn_encoded_eos
        phn_encoded_bos = torch.LongTensor(
            label_encoder.prepend_bos_index(phn_encoded_list)
        )
        yield phn_encoded_bos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "bos_label": hparams["bos_index"],
        "eos_label": hparams["eos_index"],
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="phn_list",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "phn_encoded", "phn_encoded_eos", "phn_encoded_bos"],
    )

    return train_data, valid_data, test_data, label_encoder


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Dataset prep (parsing TIMIT and annotation into csv files)
    from timit_prepare import prepare_timit  # noqa

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_timit,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "skip_prep": hparams["skip_prep"],
            "uppercase": hparams["uppercase"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)

    # Subsampling
    if hparams["max_duration"] is not None:
        # Shuffle all data
        import random

        random.seed(hparams["seed"])
        all_keys = list(train_data.data.keys())
        random.shuffle(all_keys)
        train_data.data = {k: train_data.data[k] for k in all_keys}
        train_data.data_ids = list(train_data.data.keys())

        # Subsample
        subsampled_data = {}
        total_duration = 0
        for key, value in train_data.data.items():
            if total_duration > hparams["max_duration"]:
                break
            subsampled_data[key] = value
            total_duration += value["duration"]
        train_data.data = {k: v for k, v in subsampled_data.items()}
        train_data.data_ids = list(train_data.data.keys())

    # ###################################################################
    # Define Bayesian modules
    # ###################################################################
    import math
    try:
        from bayestorch.distributions import get_log_scale_normal
        from bayestorch.nn import ParticlePosteriorModule
        from bayestorch.optim import SVGD
    except ImportError:
        raise ImportError(
            "Please install BayesTorch to use BayesSpeech (e.g. `pip install bayestorch==0.0.3`)"
        )

    # Minimize number of modifications to existing training/evaluation loops
    class SVGDModule(ParticlePosteriorModule):
        def forward(self, *args, **kwargs):
            if self.training:
                output, self.log_prior = super().forward(
                    *args, return_log_prior=True, **kwargs
                )
                return output
            output, self.log_prior = (
                super().forward(*args, **kwargs),
                0.0,
            )
            return output

    def rbf_kernel(x1, x2):
        deltas = torch.cdist(x1, x2)
        squared_deltas = deltas ** 2
        bandwidth = (
            squared_deltas.detach().median()
            / math.log(min(x1.shape[0], x2.shape[0]))
        )
        log_kernels = -squared_deltas / bandwidth
        kernels = log_kernels.exp()
        return kernels

    for key in ["seq_lin", "ctc_lin"]:
        prior_builder, prior_kwargs = get_log_scale_normal(
            hparams["modules"][key].parameters(), log_scale=hparams["normal_prior_log_scale"],
        )
        hparams[key] = hparams["modules"][key] = SVGDModule(
            hparams["modules"][key],
            prior_builder,
            prior_kwargs,
            hparams["num_particles"],
        ).to(run_opts["device"])
    hparams["model"] = torch.nn.ModuleList(
        [hparams["enc"], hparams["emb"], hparams["dec"],]
    )
    hparams["greedy_searcher"].linear = hparams["seq_lin"]
    hparams["beam_searcher"].linear = hparams["seq_lin"]
    hparams["beam_searcher"].ctc_linear = hparams["ctc_lin"]
    hparams["preconditioners"] = [
        SVGD(
            hparams["seq_lin"].parameters(),
            rbf_kernel,
            hparams["num_particles"],
        ),
        SVGD(
            hparams["ctc_lin"].parameters(),
            rbf_kernel,
            hparams["num_particles"],
        )
    ]
    hparams["checkpointer"].recoverables["model"] = hparams["model"]
    hparams["checkpointer"].add_recoverable(
        "seq_lin", hparams["seq_lin"],
    )
    hparams["checkpointer"].add_recoverable(
        "ctc_lin", hparams["ctc_lin"],
    )
    for i, preconditioner in enumerate(hparams["preconditioners"]):
        hparams["checkpointer"].add_recoverable(
            f"preconditioner{i}", preconditioner,
        )
    # ###################################################################

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder

    # Training/validation loop
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    asr_brain.evaluate(
        test_data,
        min_key="PER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
