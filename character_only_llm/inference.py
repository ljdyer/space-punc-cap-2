import torch
import pytorch_lightning as pl
from character_only_llm.byt5_model import ByT5Model

pl.seed_everything(42)
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    model = ByT5Model()
    model.load_model(
        model_dir="/experiments/ahughes/character_only_llm/outputs/simplet5-epoch-1-train-loss-0.1662-val-loss-0.078",
    )
    # train
    #     result = model.predict("""restore: Rahul Gandhi has replied to Goa CM Manohar Parrikar's letter,
    # which accused the Congress President of using his "visit to an ailing man for political gains".
    # "He's under immense pressure from the PM after our meeting and needs to demonstrate his loyalty by attacking me,"
    # Gandhi wrote in his letter. Parrikar had clarified he didn't discuss Rafale deal with Rahul.
    # """)
    prediction = """restore: rahul gandhi has replied to goa cm manohar parrikars letter 
which accused the congress president of using his visit to an ailing man for political gains 
hes under immense pressure from the pm after our meeting and needs to demonstrate his loyalty by attacking me 
gandhi wrote in his letter parrikar had clarified he didn't discuss rafale deal with rahul
"""
    result = model.predict(prediction)
    print(result)

