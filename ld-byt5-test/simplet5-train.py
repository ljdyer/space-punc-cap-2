import argparse
import pandas as pd
import more_itertools
import socket
from sklearn.model_selection import train_test_split
from my_simplet5 import SimpleT5
from typing import List
from pathlib import Path
from constants import *
from helper import *
from transformers import ByT5Tokenizer
import os

parser = argparse.ArgumentParser(
    description='Train a SimpletT5 model for feature restoration',
    allow_abbrev=False
)

parser.add_argument('--num_docs_to_use', '-n', type=str, help="The number of documents to use. Must be 'all' or an integer.")
parser.add_argument('--outputdir', '-o', type=str, help="The directory in which to store saved models.")
parser.add_argument('--max_epochs', '-e', type=int, help="The maximum number of epochs.")
parser.add_argument('--language', '-l', type=str, help="The language to train for.")
parser.add_argument('--model_name', '-m', type=str, help="Model name. 's' or 'b'.")
parser.add_argument('--gpus', '-g', type=str, help="GPUs. E.g. 1,2")
parser.add_argument('--strategy', '-s', type=str, help="Strategy. ddp or ddp_sharded.")
parser.add_argument('--check', '-c', type=int, default=1)



# ====================
def get_data():

    global train_df
    global val_df
    train_path = TRAIN_PATH[lang]
    if 'LOCAL_RANK' not in os.environ.keys() and 'NODE_RANK' not in os.environ.keys():
        print(f'Loading training data from {train_path}...')
    train_df = load_and_prep_df(train_path, num_docs_to_use, lang)
    if 'LOCAL_RANK' not in os.environ.keys() and 'NODE_RANK' not in os.environ.keys():
        print()
        print(train_df.head(10))
        print()
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    if 'LOCAL_RANK' not in os.environ.keys() and 'NODE_RANK' not in os.environ.keys():
        print(f'Num samples: {len(train_df)} train, {len(val_df)} validation')
        print()
        show_input_lengths(train_df)
        

# ====================
def train():

    model = SimpleT5()
    model.from_pretrained(
        model_type="byt5",
        model_name=model_name,
        dataloader_num_workers=20
    )
    model.train(train_df=train_df,
                eval_df=val_df,
                outputdir=outputdir,
                source_max_token_len=MAX_TOKEN_LEN_BYTES,
                target_max_token_len=MAX_TOKEN_LEN_BYTES,
                batch_size=8,
                max_epochs=max_epochs,
                gpus=gpus,
                strategy=strategy
            )


# ====================
def show_input_lengths(data):

    info = []
    for text_type, col_label in [('Inputs', 'source_text'), ('Outputs', 'target_text')]:
        texts = data[col_label].to_list()
        text_lengths = [len(s) for s in texts]
        info.append({'Text_type': text_type, 'Min': min(text_lengths), 'Max': max(text_lengths), 'Avg': f"{sum(text_lengths)/len(text_lengths):.2f}"})
    print('Expected character lengths of training examples:')
    print(tabulate_list_of_dicts(info))
    print()

    info = []
    for text_type, col_label in [('Inputs', 'source_text'), ('Outputs', 'target_text')]:
        texts = data[col_label].to_list()
        text_lengths = [utf8len(s)+1 for s in texts]
        info.append({'Text_type': text_type, 'Min': min(text_lengths), 'Max': max(text_lengths), 'Avg': f"{sum(text_lengths)/len(text_lengths):.2f}"})
    print('Expected tokenized lengths of training examples:')
    print(tabulate_list_of_dicts(info))
    print()


# ====================
if __name__ == "__main__":

    global outputdir, num_docs_to_use, max_epochs, lang, model_name, gpus, strategy, check
    args = parser.parse_args()
    num_docs_to_use = args.num_docs_to_use
    outputdir = args.outputdir
    max_epochs = args.max_epochs
    lang = args.language
    model_name = 'google/byt5-small' if args.model_name.lower() == 's' else 'google/byt5-base'
    gpus = [int(x) for x in re.findall(r'\d', args.gpus)]
    strategy = None if args.strategy == 'none' else args.strategy
    check = bool(args.check)

    get_data()
    if check is False:
        train()