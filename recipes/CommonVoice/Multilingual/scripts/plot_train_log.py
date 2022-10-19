#!/usr/bin/env python3

# ==============================================================================
# Copyright 2022 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt, rc
from numpy import ndarray


__all__ = [
    "parse_train_log",
    "plot_metrics",
]


_EXPECTED_METRICS = [
    "epoch",
    "train loss",
    "valid loss",
    "valid CER",
    "valid WER",
]


def parse_train_log(train_log_file: "str") -> "Dict[str, ndarray]":
    """Parse train log to extract metric names and values.

    Parameters
    ----------
    train_log_file:
        The path to the train log file.

    Returns
    -------
        The metrics, i.e. a dict that maps names of
        the metrics to the metric values themselves.

    Examples
    --------
    >>> metrics = parse_train_log("train_log.txt")

    """
    metrics = defaultdict(list)
    with open(train_log_file) as f:
        for line in f:
            line = line.strip().replace(" - ", ", ")
            if not line:
                continue
            tokens = line.split(", ")
            names, values = zip(*[token.split(": ") for token in tokens])
            names, values = list(names), list(values)
            for name in _EXPECTED_METRICS:
                if name not in names:
                    names.append(name)
                    values.append("nan")
            for name, value in zip(names, values):
                metrics[name].append(float(value))
    for name, values in metrics.items():
        metrics[name] = np.array(values)
    return metrics


def plot_metrics(
    metrics: "Dict[str, ndarray]",
    output_image: "str",
    title: "str" = "",
    figsize: "Tuple[float, float]" = (7.5, 6.0),
    usetex: "bool" = False,
    style_file_or_name: "str" = "classic",
) -> "None":
    """Plot metrics extracted from train log.

    Parameters
    ----------
    metrics:
        The metrics, i.e. a dict that maps names of
        the metrics to the metric values themselves.
    output_image:
        The path to the output image.
    title:
        The plot title.
    figsize:
        The figure size.
    usetex:
        True to render text with LaTeX, False otherwise.
    style_file_or_name:
        The path to a Matplotlib style file or the name of one
        of Matplotlib built-in styles
        (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).

    Examples
    --------
    >>> metrics = parse_train_log("train_log.txt")
    >>> plot_metrics(metrics, "train_log.png")

    """
    if os.path.isfile(style_file_or_name):
        style_file_or_name = os.path.realpath(style_file_or_name)
    with plt.style.context(style_file_or_name):
        rc("text", usetex=usetex)
        fig = plt.figure(figsize=figsize)
        plt.plot(
            metrics["epoch"],
            metrics["train loss"],
            marker=".",
            zorder=0,
            label="Train loss",
        )
        plt.plot(
            metrics["epoch"],
            metrics["valid loss"],
            marker=".",
            zorder=1,
            label="Validation loss",
        )
        min_valid_loss_epoch = int(
            metrics["epoch"][metrics["valid loss"].argmin()]
        )
        label = None
        if "valid CER" in metrics:
            label = f"CER = {metrics['valid CER'][min_valid_loss_epoch]}"
            if "valid WER" in metrics:
                label += f"\nWER = {metrics['valid WER'][min_valid_loss_epoch]}"
        plt.scatter(
            min_valid_loss_epoch,
            metrics["valid loss"][min_valid_loss_epoch],
            marker="d",
            s=60,
            color="red",
            zorder=2,
            label=label,
        )
        plt.grid()
        plt.xlim(0)
        plt.title(title)
        plt.legend(numpoints=1, scatterpoints=1)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        fig.tight_layout()
        plt.savefig(output_image, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot train log")
    parser.add_argument(
        "train_log", help="path to train log",
    )
    parser.add_argument(
        "-o", "--output_image", default=None, help="path to output image",
    )
    parser.add_argument(
        "-t", "--title", default=None, help="plot title",
    )
    parser.add_argument(
        "-f",
        "--figsize",
        nargs=2,
        default=(7.5, 6.0),
        type=float,
        help="figure size",
    )
    parser.add_argument(
        "-u", "--usetex", action="store_true", help="render text with LaTeX",
    )
    parser.add_argument(
        "-s",
        "--style",
        default="classic",
        help="path to a Matplotlib style file or name of one of Matplotlib built-in styles",
        dest="style_file_or_name",
    )
    args = parser.parse_args()
    metrics = parse_train_log(args.train_log)
    output_image = args.output_image or args.train_log.replace(".txt", ".png")
    title = args.title or output_image.capitalize().replace("_", " ").replace(
        ".png", ""
    )
    plot_metrics(
        metrics,
        output_image,
        title,
        args.figsize,
        args.usetex,
        args.style_file_or_name,
    )
