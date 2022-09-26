# Digit recognition with ResNet

This folder contains a recipe for training a digit recognition system on audioMNIST
based on ResNet.

First of all, install the extra dependencies:

```
pip install -r ../extra-dependencies.txt
```

To train the digit recognition model, just execute the following on the command-line:

```
python train.py hparams.yaml
```

# Results

| Hyperparams file | Test Classification Error |       GPU        |
|:----------------:|:-------------------------:|:----------------:|
| hparams.yaml     | 2.33%                     | GeForce RTX 3070 |
