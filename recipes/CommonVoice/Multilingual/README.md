# Massively Multilingual Speech Recognition

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

First of all, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Clone the repository, navigate to `<path-to-repository>/recipes/CommonVoice/Multilingual`, open a terminal and run:

```
conda env create -f environment.yaml
```

Project dependencies (pinned to a specific version) will be installed in a
[Conda](https://www.anaconda.com/) virtual environment named `multilingual-env`.
To activate it, run:

```
conda activate multilingual-env
```

or alternatively:

```
source activate multilingual-env
```

To deactivate it, run:

```
conda deactivate
```

To permanently delete it, run:

```
conda remove --n multilingual-env --all
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

In this example we will train CTC on Common Voice 10.0 `small`.
Navigate to `<path-to-repository>/recipes/CommonVoice/Multilingual/ctc`, open a terminal and run:

```
conda activate multilingual-env
python train.py hparams/multilingual/train_small.yaml
```

**NOTE**: you can download Common Voice 10.0 beforehand, build the `small`, `medium` and `large` variants
(requires ~1.2 TB of free space) and store them for later use.
To do so, navigate to `<path-to-repository>/recipes/CommonVoice/Multilingual`, open a terminal and run:

```
conda activate multilingual-env
python common_voice_prepare.py small medium large
```

It is recommended to compress the downloaded datasets into `tar.gz` archives to store them more efficiently:

```
cd data
tar -czvf common_voice_10_0_<size>.tar.gz common_voice_10_0_<size>
rm -r common_voice_10_0_<size>
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
