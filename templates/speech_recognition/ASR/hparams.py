# ############################################################################
# Model: E2E ASR with attention-based ASR
# Encoder: CRDNN
# Decoder: GRU + beamsearch + RNNLM
# Tokens: 1000 BPE
# losses: CTC+ NLL
# Training: mini-librispeech
# Pre-Training: librispeech 960h
# Authors: Luca Della Libera 2022
# # ############################################################################

import sentencepiece
import torch

import speechbrain
from speechbrain.lobes import augment, features
from speechbrain.lobes.models import CRDNN, RNNLM


# Seed needs to be set at top of configuration file, before objects with parameters are instantiated
seed = 2602
torch.manual_seed(seed)

# If you plan to train a system on an HPC cluster with a big dataset,
# we strongly suggest doing the following:
# 1- Compress the dataset in a single tar or zip file.
# 2- Copy your dataset locally (i.e., the local disk of the computing node).
# 3- Uncompress the dataset in the local folder.
# 4- Set data_folder with the local path
# Reading data from the local disk of the compute node (e.g. $SLURM_TMPDIR with SLURM-based clusters) is very important.
# It allows you to read the data much faster without slowing down the shared filesystem.

data_folder = "../data" # In this case, data will be automatically downloaded here.
data_folder_rirs = data_folder # noise/ris dataset will automatically be downloaded here
output_folder = f"results/CRDNN_BPE_960h_LM/{seed}"
wer_file = f"{output_folder}/wer.txt"
save_folder = f"{output_folder}/save"
train_log = f"{output_folder}/train_log.txt"

# Language model (LM) pretraining
# NB: To avoid mismatch, the speech recognizer must be trained with the same
# tokenizer used for LM training. Here, we download everything from the
# speechbrain HuggingFace repository. However, a local path pointing to a
# directory containing the lm.ckpt and tokenizer.ckpt may also be specified
# instead. E.g if you want to use your own LM / tokenizer.
pretrained_path = "speechbrain/asr-crdnn-rnnlm-librispeech"


# Path where data manifest files will be stored. The data manifest files are created by the
# data preparation script
train_annotation = "../train.json"
valid_annotation = "../valid.json"
test_annotation = "../test.json"

# The train logger writes training statistics to a file, as well as stdout.
train_logger = speechbrain.utils.train_logger.FileTrainLogger(
    save_file=train_log,
)

# Training parameters
number_of_epochs = 15
number_of_ctc_epochs = 5
batch_size = 8
lr = 1.0
ctc_weight = 0.5
sorting = "ascending"
ckpt_interval_minutes = 15  # save checkpoint every N min
label_smoothing = 0.1

# Dataloader options
train_dataloader_opts = dict(
    batch_size=batch_size,
)

valid_dataloader_opts = dict(
    batch_size=batch_size,
)

test_dataloader_opts = dict(
    batch_size=batch_size,
)

# Feature parameters
sample_rate = 16000
n_fft = 400
n_mels = 40

# Model parameters
activation = torch.nn.LeakyReLU
dropout = 0.15
cnn_blocks = 2
cnn_channels = (128, 256)
inter_layer_pooling_size = (2, 2)
cnn_kernelsize = (3, 3)
time_pooling_size = 4
rnn_class = speechbrain.nnet.RNN.LSTM
rnn_layers = 4
rnn_neurons = 1024
rnn_bidirectional = True
dnn_blocks = 2
dnn_neurons = 512
emb_size = 128
dec_neurons = 1024
output_neurons = 1000  # Number of tokens (same as LM)
blank_index = 0
bos_index = 0
eos_index = 0

# Decoding parameters
min_decode_ratio = 0.0
max_decode_ratio = 1.0
valid_beam_size = 8
test_beam_size = 80
eos_threshold = 1.5
using_max_attn_shift = True
max_attn_shift = 240
lm_weight = 0.50
ctc_weight_decode = 0.0
coverage_penalty = 1.5
temperature = 1.25
temperature_lm = 1.25

# The first object passed to the Brain class is this "Epoch Counter"
# which is saved by the Checkpointer so that training can be resumed
# if it gets interrupted at any point.
epoch_counter = speechbrain.utils.epoch_loop.EpochCounter(
    limit=number_of_epochs,
)

# Feature extraction
compute_features = speechbrain.lobes.features.Fbank(
    sample_rate=sample_rate,
    n_fft=n_fft,
    n_mels=n_mels,
)

# Feature normalization (mean and std)
normalize = speechbrain.processing.features.InputNormalization(
    norm_type="global",
)

# Added noise and reverb come from OpenRIR dataset, automatically
# downloaded and prepared with this Environmental Corruption class.
env_corrupt = speechbrain.lobes.augment.EnvCorrupt(
    openrir_folder=data_folder_rirs,
    babble_prob=0.0,
    reverb_prob=0.0,
    noise_prob=1.0,
    noise_snr_low=0,
    noise_snr_high=15,
)

# Adds speech change + time and frequency dropouts (time-domain implementation).
augmentation = speechbrain.lobes.augment.TimeDomainSpecAugment(
    sample_rate=sample_rate,
    speeds=[95, 100, 105],
)

# The CRDNN model is an encoder that combines CNNs, RNNs, and DNNs.
encoder = speechbrain.lobes.models.CRDNN.CRDNN(
    input_shape=[None, None, n_mels],
    activation=activation,
    dropout=dropout,
    cnn_blocks=cnn_blocks,
    cnn_channels=cnn_channels,
    cnn_kernelsize=cnn_kernelsize,
    inter_layer_pooling_size=inter_layer_pooling_size,
    time_pooling=True,
    using_2d_pooling=False,
    time_pooling_size=time_pooling_size,
    rnn_class=rnn_class,
    rnn_layers=rnn_layers,
    rnn_neurons=rnn_neurons,
    rnn_bidirectional=rnn_bidirectional,
    rnn_re_init=True,
    dnn_blocks=dnn_blocks,
    dnn_neurons=dnn_neurons,
    use_rnnp=False,
)

# Embedding (from indexes to an embedding space of dimension emb_size).
embedding = speechbrain.nnet.embedding.Embedding(
    num_embeddings=output_neurons,
    embedding_dim=emb_size,
)

# Attention-based RNN decoder.
decoder = speechbrain.nnet.RNN.AttentionalRNNDecoder(
    enc_dim=dnn_neurons,
    input_size=emb_size,
    rnn_type="gru",
    attn_type="location",
    hidden_size=dec_neurons,
    attn_dim=1024,
    num_layers=1,
    scaling=1.0,
    channels=10,
    kernel_size=100,
    re_init=True,
    dropout=dropout,
)

# Linear transformation on the top of the encoder.
ctc_lin = speechbrain.nnet.linear.Linear(
    input_size=dnn_neurons,
    n_neurons=output_neurons,
)

# Linear transformation on the top of the decoder.
seq_lin = speechbrain.nnet.linear.Linear(
    input_size=dec_neurons,
    n_neurons=output_neurons,
)

# Final softmax (for log posteriors computation).
log_softmax = speechbrain.nnet.activations.Softmax(
    apply_log=True,
)

# Cost definition for the CTC part.
ctc_cost = lambda *args: speechbrain.nnet.losses.ctc_loss(
    *args,
    blank_index=blank_index,
)

# This is the RNNLM that is used according to the Huggingface repository
# NB: It has to match the pre-trained RNNLM!!
lm_model = speechbrain.lobes.models.RNNLM.RNNLM(
    output_neurons=output_neurons,
    embedding_dim=emb_size,
    activation=torch.nn.LeakyReLU,
    dropout=0.0,
    rnn_layers=2,
    rnn_neurons=2048,
    dnn_blocks=1,
    dnn_neurons=512,
    return_hidden=True,  # For inference
)


# Tokenizer initialization
tokenizer = sentencepiece.SentencePieceProcessor()

# Objects in "modules" dict will have their parameters moved to the correct
# device, as well as having train()/eval() called on them by the Brain class
modules=dict(
    encoder=encoder,
    embedding=embedding,
    decoder=decoder,
    ctc_lin=ctc_lin,
    seq_lin=seq_lin,
    normalize=normalize,
    env_corrupt=env_corrupt,
    lm_model=lm_model,
)

# Gathering all the submodels in a single model object.
model = torch.nn.ModuleList(
    [encoder, embedding, decoder, ctc_lin, seq_lin],
)

# Beamsearch is applied on the top of the decoder. If the language model is
# given, a language model is applied (with a weight specified in lm_weight).
# If ctc_weight is set, the decoder uses CTC + attention beamsearch. This
# improves the performance, but slows down decoding. For a description of
# the other parameters, please see the speechbrain.decoders.S2SRNNBeamSearchLM.

# It makes sense to have a lighter search during validation. In this case,
# we don't use the LM and CTC probabilities during decoding.
valid_search = speechbrain.decoders.S2SRNNBeamSearcher(
    embedding=embedding,
    decoder=decoder,
    linear=seq_lin,
    ctc_linear=ctc_lin,
    bos_index=bos_index,
    eos_index=eos_index,
    blank_index=blank_index,
    min_decode_ratio=min_decode_ratio,
    max_decode_ratio=max_decode_ratio,
    beam_size=valid_beam_size,
    eos_threshold=eos_threshold,
    using_max_attn_shift=using_max_attn_shift,
    max_attn_shift=max_attn_shift,
    coverage_penalty=coverage_penalty,
    temperature=temperature,
)

# The final decoding on the test set can be more computationally demanding.
# In this case, we use the LM + CTC probabilities during decoding as well.
# Please, remove this part if you need a faster decoder.
test_search = speechbrain.decoders.S2SRNNBeamSearchLM(
    embedding=embedding,
    decoder=decoder,
    linear=seq_lin,
    ctc_linear=ctc_lin,
    language_model=lm_model,
    bos_index=bos_index,
    eos_index=eos_index,
    blank_index=blank_index,
    min_decode_ratio=min_decode_ratio,
    max_decode_ratio=max_decode_ratio,
    beam_size=test_beam_size,
    eos_threshold=eos_threshold,
    using_max_attn_shift=using_max_attn_shift,
    max_attn_shift=max_attn_shift,
    coverage_penalty=coverage_penalty,
    lm_weight=lm_weight,
    ctc_weight=ctc_weight_decode,
    temperature=temperature,
    temperature_lm=temperature_lm,
)

# This function manages learning rate annealing over the epochs.
# We here use the NewBoB algorithm, that anneals the learning rate if
# the improvements over two consecutive epochs is less than the defined
# threshold.
lr_annealing = speechbrain.nnet.schedulers.NewBobScheduler(
    initial_value=lr,
    improvement_threshold=0.0025,
    annealing_factor=0.8,
    patient=0,
)

# This optimizer will be constructed by the Brain class after all parameters
# are moved to the correct device. Then it will be added to the checkpointer.
opt_class = lambda *args: torch.optim.Adadelta(
    *args,
    lr=lr,
    rho=0.95,
    eps=1.e-8,
)

# Functions that compute the statistics to track during the validation step.
error_rate_computer = speechbrain.utils.metric_stats.ErrorRateStats

cer_computer = lambda *args: speechbrain.utils.metric_stats.ErrorRateStats(
    *args,
    split_tokens=True,
)

# This object is used for saving the state of training both so that it
# can be resumed if it gets interrupted, and also so that the best checkpoint
# can be later loaded for evaluation or inference.
checkpointer = speechbrain.utils.checkpoints.Checkpointer(
    checkpoints_dir=save_folder,
    recoverables=dict(
        model=model,
        scheduler=lr_annealing,
        normalizer=normalize,
        counter=epoch_counter,
    )
)

# This object is used to pretrain the language model and the tokenizers
# (defined above). In this case, we also pretrain the ASR model (to make
# sure the model converges on a small amount of data)
pretrainer = speechbrain.utils.parameter_transfer.Pretrainer(
    collect_in=save_folder,
    loadables=dict(
        lm=lm_model,
        tokenizer=tokenizer,
        model=model,
    ),
    paths=dict(
        lm=f"{pretrained_path}/lm.ckpt",
        tokenizer=f"{pretrained_path}/tokenizer.ckpt",
        model=f"{pretrained_path}/asr.ckpt",
    ),
)
