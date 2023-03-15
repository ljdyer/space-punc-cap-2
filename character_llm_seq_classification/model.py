from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from character_llm_seq_classification.ner_data import get_dataset


class ByT5FineTuner(pl.LightningModule):
    """
    ByT5 model fine-tuned for punctuation restoration.
    """

    def __init__(self, hparam):
        super(ByT5FineTuner, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("google/byt5-base")
        self.tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
        self.save_hyperparameters()
        self.opt = None
        self.hparam = hparam

    def is_logger(self):
        return True

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :param batch_idx:
        :return:
        """
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        """
        Called at the end of the training epoch with the outputs of all train steps.
        :param outputs:
        :return:
        """
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :param batch_idx:
        :return:
        """
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of the validation epoch with the outputs of all validation steps.
        :param outputs:
        :return:
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}

    def configure_optimizers(self):
        """
        Prepare optimizer and schedule (linear warmup and decay)
        """

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparam["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparam["learning_rate"], eps=self.hparam['adam_epsilon'])
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self,
                       epoch=None,
                       batch_idx=None,
                       optimizer=None,
                       optimizer_idx=None,
                       optimizer_closure=None,
                       on_tpu=None,
                       using_native_amp=None,
                       using_lbfgs=None
                       ):
        """
        Called after each optimizer step.
        :param epoch:
        :param batch_idx:
        :param optimizer:
        :param optimizer_idx:
        :param optimizer_closure:
        :param on_tpu:
        :param using_native_amp:
        :param using_lbfgs:
        :return:
        """
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        """
        To log the loss of the model via tqdm
        :return:
        """
        tqdm_dict = {"loss": "{:.3f}".format(
            self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        """
        Called by PyTorch Lightning to create the DataLoader for training.
        :return:
        """
        train_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="train", args=self.hparam
        )
        dataloader = DataLoader(train_dataset, batch_size=self.hparam['train_batch_size'],
                                drop_last=True, shuffle=True, num_workers=0)

        t_total = ((len(dataloader.dataset) //
                    (self.hparam['train_batch_size'] * 1)) //
                   self.hparam['gradient_accumulation_steps'] * float(self.hparam['num_train_epochs'])
                   )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparam['warmup_steps'], num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="validation", args=self.hparam)
        return DataLoader(val_dataset, batch_size=self.hparam['eval_batch_size'], num_workers=0)
