"""BERT-based variational autoencoder.

Authors
 * Luca Della Libera 2023
"""

import torch
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer


__all__ = ["BERTAsymmetricVAE", "BERTSymmetricVAE"]


class BERTAsymmetricVAE(nn.Module):
    def __init__(self, source="bert-base-uncased", save_path="save", freeze_encoder=True):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=save_path)
        model = AutoModelForMaskedLM.from_pretrained(source, cache_dir=save_path)
        self.config = model.config
        assert self.config.vocab_size == self.tokenizer.vocab_size
        self.encoder = model.bert
        self.decoder = model.cls
        self.mean_head = nn.Linear(
            self.config.hidden_size, self.config.hidden_size
        )
        self.log_stddev_head = nn.Linear(
            self.config.hidden_size, self.config.hidden_size
        )

    def forward_encoder(self, input_ids, attention_mask):
        embeds = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(input_ids),
        ).last_hidden_state
        return embeds

    def forward_decoder(self, input_embeds, attention_mask):
        logits = self.decoder(input_embeds)
        return logits

    def forward(self, input_ids):
        assert self.config.pad_token_id == 0
        attention_mask = (input_ids != self.config.pad_token_id).long()
        with torch.set_grad_enabled(not self.freeze_encoder):
            embeds = self.forward_encoder(input_ids, attention_mask)
        mean = self.mean_head(embeds)
        log_stddev = self.log_stddev_head(embeds)
        stddev = log_stddev.exp()
        latents = mean + stddev * torch.randn_like(mean)
        kl = (stddev ** 2 + mean ** 2 - log_stddev - 0.5).sum(dim=-1)
        kl = kl[attention_mask.bool()].mean()
        logits = self.forward_decoder(latents, attention_mask)
        return logits, latents, kl


class BERTSymmetricVAE(nn.Module):
    def __init__(self, source="bert-base-uncased", save_path="save", freeze_encoder=True):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=save_path)
        self.encoder = AutoModelForMaskedLM.from_pretrained(source, cache_dir=save_path)
        self.decoder = AutoModelForMaskedLM.from_pretrained(source, cache_dir=save_path)
        self.config = self.encoder.config
        assert self.config.vocab_size == self.tokenizer.vocab_size
        self.mean_head = nn.Linear(
            self.config.hidden_size, self.config.hidden_size
        )
        self.log_stddev_head = nn.Linear(
            self.config.hidden_size, self.config.hidden_size
        )

    def forward_encoder(self, input_ids, attention_mask):
        embeds = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(input_ids),
            output_hidden_states=True,
        ).hidden_states[-1]
        return embeds

    def forward_decoder(self, input_embeds, attention_mask):
        logits = self.decoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(input_ids),
        ).logits
        return logits

    def forward(self, input_ids):
        assert self.config.pad_token_id == 0
        attention_mask = (input_ids != self.config.pad_token_id).long()
        with torch.set_grad_enabled(not self.freeze_encoder):
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
    model = BERTAsymmetricVAE()
    input_ids = model.tokenizer(
        ["hello", "world!"],
        return_attention_mask=False,
        return_tensors="pt",
        padding=True,
    ).input_ids
    print(model.tokenizer.batch_decode(model(input_ids)[0].argmax(dim=-1)))

    model = BERTSymmetricVAE()
    input_ids = model.tokenizer(
        ["hello", "world!"],
        return_attention_mask=False,
        return_tensors="pt",
        padding=True,
    ).input_ids
    print(model.tokenizer.batch_decode(model(input_ids)[0].argmax(dim=-1)))
