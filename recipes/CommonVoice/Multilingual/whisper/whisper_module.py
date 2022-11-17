import torch
import whisper
import torch.nn as nn
from transformers import AdamW
from dataclasses import dataclass
from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup
from common_voice_dataset import WhisperDataCollatorWithPadding,CommonVoiceDataset


# # make offline training possible. huggingface downloads these files by default
# from wer import WER
# from cer import CER
from jiwer import wer,cer


@dataclass
class Config:
    learning_rate: float
    weight_decay: float
    adam_epsilon: float
    warmup_steps: int
    batch_size: int
    num_workers: int
    num_train_epochs: int
    gradient_accumulation_steps: int
    dev_batch_size: int


class WhisperModelModule(LightningModule):
    def __init__(
            self,
            *,
            cfg: Config,
            # train_dataset,
            # dev_dataset,
            model_name,
            dataset_size,
            dataset_dir,
            locales,
            manifests,


    ) -> None:
        super().__init__()
        lang=None
        if locales != None:
            lang=locales[0]
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name, in_memory=True)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language=lang, task=self.options.task)

        for p in self.model.encoder.parameters():
            p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = wer
        self.metrics_cer = cer

        self.cfg = cfg
        self.__train_dataset = manifests['train']
        self.__dev_dataset = manifests['dev']
        self.dataset_size=dataset_size
        self.dataset_dir=dataset_dir
        self.locales=locales
   

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch['input_ids']
        labels = batch['labels'].long()
        dec_input_ids = batch['dec_input_ids'].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        self.log('train loss', loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch['input_ids']
        labels = batch['labels'].long()
        dec_input_ids = batch['dec_input_ids'].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))
            l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))
        cer = self.metrics_cer(l_list, o_list) * 100
        wer = self.metrics_wer(l_list, o_list) * 100

        self.log('val/loss', loss, on_step=True, prog_bar=True, logger=True)
        self.log('val/cer', cer, on_step=True, prog_bar=True, logger=True)
        self.log('val/wer', wer, on_step=True, prog_bar=True, logger=True)

        return {
            'cer': cer,
            'wer': wer,
            'loss': loss
        }

    def configure_optimizers(self):
        model = self.model
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params':
                 [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay':
                 self.cfg.weight_decay
             },
            {'params':
                 [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay':
                 0.0
             }
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.cfg.learning_rate,
            eps=self.cfg.adam_epsilon
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.t_total
        )

        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step', 'frequency': 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = (
                    (len(self.__train_dataset) // self.cfg.batch_size)
                    // self.cfg.gradient_accumulation_steps
                    * float(self.cfg.num_train_epochs)
            )

    def train_dataloader(self):
        dataset = CommonVoiceDataset(
            dataset_size=self.dataset_size,
            dataset_dir=self.dataset_dir,
            manifests=self.__train_dataset,
            tokenizer=self.tokenizer,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=WhisperDataCollatorWithPadding()
        )

    def val_dataloader(self):
        dataset = CommonVoiceDataset(
            dataset_size=self.dataset_size,
            dataset_dir=self.dataset_dir,
            manifests=self.__dev_dataset,
            tokenizer=self.tokenizer,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.dev_batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=WhisperDataCollatorWithPadding()
        )




