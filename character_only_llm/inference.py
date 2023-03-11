import torch
import pytorch_lightning as pl
from character_only_llm.byt5_model import ByT5Model

pl.seed_everything(42)
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    model = ByT5Model()
    model.load_model()
    # train
    result = model.predict("""summarize: Rahul Gandhi has replied to Goa CM Manohar Parrikar's letter, 
which accused the Congress President of using his "visit to an ailing man for political gains". 
"He's under immense pressure from the PM after our meeting and needs to demonstrate his loyalty by attacking me," 
Gandhi wrote in his letter. Parrikar had clarified he didn't discuss Rafale deal with Rahul.
""")
    print(result)
