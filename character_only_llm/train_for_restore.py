import torch
import pytorch_lightning as pl
from character_only_llm.byt5_model import ByT5Model
from character_only_llm.restoration_data import fetch_restoration_data, parse_restoration_data, generate_model_splits

pl.seed_everything(42)
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')


if __name__ == '__main__':
    train_df, val_df, eval_df = generate_model_splits(parse_restoration_data(fetch_restoration_data()))
    model = ByT5Model()
    model.train(
        train_df=train_df[:2000],
        eval_df=eval_df[:500],
        val_df=val_df[:100],
        source_max_token_len=128,
        target_max_token_len=50,
        batch_size=8,
        max_epochs=3)

