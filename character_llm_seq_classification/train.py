from transformers import AutoTokenizer
from datasets import load_dataset

from character_llm_seq_classification.custom_pl_logger import LoggingCallback
from character_llm_seq_classification.model import ByT5FineTuner
from character_llm_seq_classification.ner_dataset import NERDataset
from character_llm_seq_classification.pl_parameters import args_dict as args

import pytorch_lightning as pl
import torch

pl.seed_everything(42)
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    dataset = load_dataset("wikiann", "en")
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
    input_dataset = NERDataset(tokenizer=tokenizer, dataset=dataset, type_path='train')
    model = ByT5FineTuner(args)
    train_params = dict(
        accumulate_grad_batches=args['gradient_accumulation_steps'],
        gpus=args['n_gpu'],
        max_epochs=args['num_train_epochs'],
        precision=16 if args["fp_16"] else 32,
        gradient_clip_val=args["max_grad_norm"],
        callbacks=[LoggingCallback()],
    )
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
