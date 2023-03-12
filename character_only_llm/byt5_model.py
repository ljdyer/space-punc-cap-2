from transformers import T5ForConditionalGeneration, ByT5Tokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from character_only_llm.seq2seq_model import SeqToSeqModel
from character_only_llm.dataset_wrapper import LightningDataModule

import pandas as pd
import torch


class ByT5Model(object):
    def __init__(self, device=None):
        self.byt5_model = T5ForConditionalGeneration.from_pretrained('google/byt5-base', return_dict=True)
        self.byt5_tokenizer = ByT5Tokenizer.from_pretrained('google/byt5-base')
        self.device = device
        self.data_module = None
        self.model = None

    def train(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            eval_df: pd.DataFrame,
            source_max_token_len: int = 512,
            target_max_token_len: int = 512,
            batch_size: int = 8,
            max_epochs: int = 5,
            output_dir: str = "outputs",
            early_stopping_patience_epochs: int = 0,  # 0 to disable early stopping feature
            dataloader_num_workers: int = 0,
            save_only_last_epoch: bool = False,
    ):
        """
        trains T5/MT5 model on custom dataset
        Args:
            train_df (pd.DataFrame): training datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            eval_df ([type], optional): validation datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
            batch_size (int, optional): batch size. Defaults to 8.
            max_epochs (int, optional): max number of epochs. Defaults to 5.
            output_dir (str, optional):
                output directory to save model checkpoints. Defaults to "outputs".
            early_stopping_patience_epochs (int, optional): monitors val_loss on epoch end and stops training,
                if val_loss does not improve after the specified number of epochs. set 0 to disable early stopping.
                Defaults to 0 (disabled)
            precision (int, optional): sets precision training
                - Double precision (64), full precision (32) or half precision (16). Defaults to 32.
            dataloader_num_workers (int, optional):
                number of workers in train/test/val dataloader
            save_only_last_epoch (bool, optional):
                If True, saves only the last epoch else models are saved at every epoch
        """
        self.data_module = LightningDataModule(
            train_df=train_df,
            test_df=eval_df,
            val_df=val_df,
            tokenizer=self.byt5_tokenizer,
            batch_size=batch_size,
            source_max_token_len=source_max_token_len,
            target_max_token_len=target_max_token_len,
            num_workers=dataloader_num_workers,
        )

        self.model = SeqToSeqModel(
            tokenizer=self.byt5_tokenizer,
            model=self.byt5_model,
            outputdir=output_dir,
            save_only_last_epoch=save_only_last_epoch,
        )

        # add callbacks
        callbacks = [TQDMProgressBar(refresh_rate=1)]

        if early_stopping_patience_epochs > 0:
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=early_stopping_patience_epochs,
                verbose=True,
                mode="min",
            )
            callbacks.append(early_stop_callback)

        # prepare trainer
        trainer = pl.Trainer(
            callbacks=callbacks,
            max_epochs=max_epochs,
            log_every_n_steps=1,
            accelerator="gpu",
            devices=[2],
            num_sanity_val_steps=0
        )

        # fit trainer to our data
        trainer.fit(self.model, self.data_module)

    def load_model(self,
                   model_dir: str = '/experiments/ahughes/character_only_llm/outputs/simplet5-epoch-2-train-loss-0.7237-val-loss-0.6376'):
        """
        loads a checkpoint for inference/prediction
        """
        self.byt5_model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.byt5_tokenizer = ByT5Tokenizer.from_pretrained(model_dir)
        self.device = torch.device("cuda")
        self.byt5_model = self.byt5_model.to(self.device)

    def predict(
            self,
            source_text: str,
            max_length: int = 512,
            num_return_sequences: int = 1,
            num_beams: int = 2,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 2.5,
            length_penalty: float = 1.0,
            early_stopping: bool = True,
            skip_special_tokens: bool = True,
            clean_up_tokenization_spaces: bool = True,
    ):
        """
        generates prediction for T5/MT5 model
        Args:
            source_text (str): any text for generating predictions
            max_length (int, optional): max token length of prediction. Defaults to 512.
            num_return_sequences (int, optional): number of predictions to be returned. Defaults to 1.
            num_beams (int, optional): number of beams. Defaults to 2.
            top_k (int, optional): Defaults to 50.
            top_p (float, optional): Defaults to 0.95.
            do_sample (bool, optional): Defaults to True.
            repetition_penalty (float, optional): Defaults to 2.5.
            length_penalty (float, optional): Defaults to 1.0.
            early_stopping (bool, optional): Defaults to True.
            skip_special_tokens (bool, optional): Defaults to True.
            clean_up_tokenization_spaces (bool, optional): Defaults to True.
        Returns:
            list[str]: returns predictions
        """
        input_ids = self.byt5_tokenizer.encode(
            source_text, return_tensors="pt", add_special_tokens=True
        )
        input_ids = input_ids.to(self.device)
        generated_ids = self.byt5_model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
        )
        preds = [
            self.byt5_tokenizer.decode(
                g,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for g in generated_ids
        ]
        return preds
