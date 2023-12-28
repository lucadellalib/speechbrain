#!/usr/bin/env/python3

"""Profile a separation model.

To run this script, do the following:
> python profile.py hparams/resepformer.yaml --data_folder .

Authors
 * Luca Della Libera 2023
"""

import os
import pickle
import sys
import time

from hyperpyyaml import load_hyperpyyaml
from matplotlib import pyplot as plt
import ptflops
import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

import speechbrain as sb


class ProfilableModel(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = hparams["Encoder"]
        self.masknet = hparams["MaskNet"]
        self.decoder = hparams["Decoder"]

    def forward(self, mix):
        mix_w = self.encoder(mix)
        est_mask = self.masknet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams["num_spks"])
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams["num_spks"])
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source


def profile(hparams):
    # True might improve performance sometimes: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/29
    # torch.backends.cudnn.benchmark = False

    # Default PyTorch value is usually OK
    # torch.set_num_threads(4)
    print(f"Number of threads: {torch.get_num_threads()}")

    torch.manual_seed(2)

    device = "cuda"
    model = ProfilableModel(hparams)
    # Move to device and set in eval mode
    model = model.to(device).eval()

    # Do some warmup iterations
    # NOTE: use no_grad to skip gradient recording
    # NOTE: use float16 for faster inference (more optimized for transformers)
    # NOTE: to further improve inference time for transformers,
    #       you should set return_attn_weights=False and enable
    #       FlashAttention in SpeechBrain's attention.py
    #       with torch.backends.cuda.sdp_kernel(
    #           enable_flash=True, enable_math=False, enable_mem_efficient=True
    #       ):
    #           ...
    n_warmups = 20
    for _ in range(n_warmups):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                model(torch.randn(1, hparams["sample_rate"] * 4, device=device))

    time_results = []
    memory_results = []
    macs_results = []

    n_runs = 10
    secs = [1, 2, 4, 8, 16, 32, 64, 128, 256]#, 512]
    for sec in secs:
        print(f"Trying {sec} seconds long input")

        total_time = 0.
        total_peak_memory = 0.

        # Average over n_runs
        for n_run in range(n_runs):
            print(f"Run {n_run}")
            # Reset memory stats to correctly record the peak memory usage
            # Comment out to make plot less irregular
            # torch.cuda.reset_peak_memory_stats()
            inputs = torch.randn(1, hparams["sample_rate"] * sec, device=device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    torch.cuda.synchronize()
                    time_start = time.time()
                    model(inputs)
                    # This is important for CUDA, if we do not include it we measure the scheduling time rather than the
                    # execution time (CUDA is asynchronous):
                    # https://discuss.pytorch.org/t/how-does-torch-cuda-synchronize-behave/147049
                    # https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
                    torch.cuda.synchronize()
                    time_end = time.time()
                    total_time += (time_end - time_start)
                    total_peak_memory += torch.cuda.max_memory_allocated(device) / 10 ** 9
        time_results.append(total_time / n_runs)
        memory_results.append(total_peak_memory / n_runs)

        # This is deterministic, no need to average
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                macs, _ = ptflops.get_model_complexity_info(
                    model, (inputs.shape[1],), as_strings=True, print_per_layer_stat=False, verbose=False,
                )
                macs_results.append(macs)

    results = {
        "time": time_results,
        "memory": memory_results,
        "macs": macs_results,
    }
    print(results)

    # Double check timings using PyTorch ad-hoc benchmarking tools
    # https://pytorch.org/tutorials/recipes/recipes/benchmark.html

    # Averaging is done automatically (using always the same input)
    time_results = []
    for sec in secs:
        print(f"Trying {sec} seconds long input")

        # Since the input is always the same across runs, let's keep it fixed to avoid discrepancies
        # Synchronization is handled by the timer
        inputs = torch.ones(1, hparams["sample_rate"] * sec, device=device)
        timer = benchmark.Timer(
            setup="import torch",
            stmt=(
                "with torch.no_grad():\n"
                "    with torch.cuda.amp.autocast():\n"
                "        model(inputs)"
            ),
            globals={"model": model, "inputs": inputs}
        )
        total_time = timer.timeit(n_runs).mean
        time_results.append(total_time)
    print(f"Double check times: {time_results}")

    return results


def plot(results):
    names = ["RE-SepFormer", "SkiM", "SepFormer-Light", "MambaFormer", "MambaNet"]
    marks = ["-ro", "-gx", "-mv", "-k^", "-bs", "-c>", "-y.", "-r<", "-go", "-ms"]
    fontsize = 14

    plt.figure(figsize=[5, 5], dpi=100)
    for result, name, mark in zip(results["time"], names, marks):
        plt.plot(result, mark, label=name)
    #plt.title("Inference Time", fontsize=fontsize)
    plt.ylabel("Inference time (s)", fontsize=fontsize)
    plt.xlabel("Input length (s)", fontsize=fontsize)
    plt.xticks([i for i in range(len(result))], [2 ** i for i in range(len(result))], fontsize=fontsize)
    plt.grid()
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("time_v10.pdf", bbox_inches="tight")
    plt.savefig("time_v10.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=[5, 5], dpi=100)
    for result, name, mark in zip(results["memory"], names, marks):
        plt.plot(result, mark, label=name)
    #plt.title("Memory", fontsize=fontsize)
    plt.ylabel("Memory (GB)", fontsize=fontsize)
    plt.xlabel("Input length (s)", fontsize=fontsize)
    plt.xticks([i for i in range(len(result))], [2 ** i for i in range(len(result))], fontsize=fontsize)
    plt.grid()
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("memory_v10.pdf", bbox_inches="tight")
    plt.savefig("memory_v10.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Profile
    results = profile(hparams)
    with open(os.path.basename(hparams_file).replace(".yaml", ".pkl"), "wb") as f:
        pickle.dump(results, f)

    # Plot
    results = []
    for filepath in ["resepformer.pkl", "skim_A100.pkl", "sepformer-light.pkl", "resepformer_conformer_small_mamba.pkl", "resepformer_conformer_small_mambanet.pkl"]:
        with open(filepath, "rb") as f:
            result = pickle.load(f)
        results.append(result)
    results = {
        "time": [x["time"] for x in results],
        "memory": [x["memory"] for x in results],
    }
    plot(results)
