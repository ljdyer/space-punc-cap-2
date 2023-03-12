import torch
import pytorch_lightning as pl
from character_only_llm.byt5_model import ByT5Model
from character_only_llm.summarisation_data import fetch_summarisation_data, parse_summarisation_data, generate_model_splits

pl.seed_everything(42)
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')


if __name__ == '__main__':
    model = ByT5Model()
    train_df, val_df, eval_df = generate_model_splits(parse_summarisation_data(fetch_summarisation_data()))
    # train
    model.train(
        train_df=train_df[:5000],
        eval_df=eval_df[:100],
        val_df=val_df[:100],
        source_max_token_len=128,
        target_max_token_len=50,
        batch_size=8,
        max_epochs=3)
