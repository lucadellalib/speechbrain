"""LSTM-based variational autoencoder.

Authors
 * Luca Della Libera 2023
"""

import torch
from torch import nn
from transformers import AutoTokenizer


__all__ = ["LSTMVAE"]


class LSTMVAE(nn.Module):
    def __init__(self, source="bert-base-uncased", save_path="save", hidden_size=256, num_layers=6):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=save_path)
        self.embedding = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=hidden_size,
            padding_idx=0,
        )
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder = nn.LSTM(
            input_size=2 * hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.mean_head = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.log_stddev_head = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.projection = nn.Linear(2 * hidden_size, self.tokenizer.vocab_size)

    def forward_encoder(self, input_ids, attention_mask):
        embeds = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=-1).cpu()
        embeds = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False
        )
        embeds, _ = self.encoder(embeds)
        embeds, _ = nn.utils.rnn.pad_packed_sequence(embeds, batch_first=True, total_length=max(lengths))
        return embeds

    def forward_decoder(self, input_embeds, attention_mask):
        lengths = attention_mask.sum(dim=-1).cpu()
        input_embeds = nn.utils.rnn.pack_padded_sequence(
            input_embeds, lengths, batch_first=True, enforce_sorted=False
        )
        embeds, _ = self.decoder(input_embeds)
        embeds, _ = nn.utils.rnn.pad_packed_sequence(embeds, batch_first=True, total_length=max(lengths))
        logits = self.projection(embeds)
        return logits

    def forward(self, input_ids):
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        embeds = self.forward_encoder(input_ids, attention_mask)
        mean = self.mean_head(embeds)
        log_stddev = self.log_stddev_head(embeds)
        stddev = log_stddev.exp()
        latents = mean + stddev * torch.randn_like(mean)
        kl = (stddev ** 2 + mean ** 2 - log_stddev - 0.5).sum(dim=-1)
        kl = kl[attention_mask.bool()].mean()
        logits = self.forward_decoder(latents, attention_mask)
        return logits, latents, kl


if __name__ == "__main__":
    model = LSTMVAE()
    input_ids = model.tokenizer(
        ["hello", "world!"],
        return_attention_mask=False,
        return_tensors="pt",
        padding=True,
    ).input_ids
    print(model.tokenizer.batch_decode(model(input_ids)[0].argmax(dim=-1)))
