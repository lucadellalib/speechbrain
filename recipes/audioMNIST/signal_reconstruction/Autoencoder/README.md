# Signal reconstruction with convolutional autoencoder

This folder contains a recipe for training a signal reconstruction system on audioMNIST
based on a convolutional autoencoder.

First of all, install the extra dependencies:

```
pip install -r ../extra-dependencies.txt
```

To train the signal reconstruction model, just execute the following on the command-line:

```
python train.py hparams.yaml
```

# Results

| Hyperparams file | Test Mean Squared Error   |       GPU        |
|:----------------:|:-------------------------:|:----------------:|
| hparams.yaml     | 2.26e-03                  | GeForce RTX 3070 |
