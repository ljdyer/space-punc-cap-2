import argparse
import pandas as pd
import more_itertools
import socket
from sklearn.model_selection import train_test_split
from simplet5 import SimpleT5

if socket.gethostname() == 'Laurences-MacBook-Air.local':
    train_path, test_path = 'ted_train.csv', 'ted_test.csv'
else:
    train_path, test_path = '/data/ldyer/ted_train.csv', '/data/ldyer/ted_test.csv'

CHUNK_LENGTH_TARGET = 100
SOURCE_MAX_TOKEN_LEN = 100 
TARGET_MAX_TOKEN_LEN = 150
NUM_TEST_SAMPLES = 10

parser = argparse.ArgumentParser(description='Train or evaluate SimpletT5 model for feature restoration')
parser.add_argument('mode', type=str, choices=['train', 'evaluate'])
parser.add_argument(
    '--num_docs_to_use', '-n', type=str, default=None,
    help="The number of documents to use. Should be 'all' or an integer. Required in training mode only.")
parser.add_argument(
    '--outputdir', '-o', type=str, default=None,
    help="The directory in which to store saved models. Required in training model only.")
parser.add_argument(
    '--max_epochs', '-e', type=int, default=None,
    help="The maximum number of epochs. Required in training mode only.")
parser.add_argument(
    '--model_dir', '-m', type=str, default=None,
    help="The directory in which the model is stored. Required in evaluation mode only.")


# ====================
def train(num_docs_to_use, outputdir, max_epochs):

    print(f'Loading training data from {train_path}...')
    train_df = load_and_prep_df(train_path, num_docs_to_use)
    print(train_df.head(10))
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    print(f'Num samples: {len(train_df)} train, {len(val_df)} validation')
    model = SimpleT5()
    model.from_pretrained(model_type="byt5", model_name="google/byt5-small")
    model.train(train_df=train_df,
                eval_df=val_df,
                outputdir=f'outputs/{outputdir}',
                source_max_token_len=SOURCE_MAX_TOKEN_LEN,
                target_max_token_len=TARGET_MAX_TOKEN_LEN,
                batch_size=8,
                max_epochs=max_epochs,
                use_gpu=True
            )


# ====================
def evaluate(model_dir):

    print(f'Loading test data from {test_path}...')
    test_df = load_and_prep_df(test_path, 'all')
    test_data = test_df.sample(NUM_TEST_SAMPLES).to_dict(orient='records')
    model = SimpleT5()
    model.load_model("byt5", model_dir)
    for t in test_data:
        print(f"Input: {t['source_text']}")
        print(f"Reference: {t['target_text']}")
        print(f"Hypothesis: {model.predict(t['source_text'])[0]}")

    
# ====================
def load_and_prep_df(csv_path, num_docs_to_use):

    all_cleaned = pd.read_csv(csv_path)['all_cleaned'].to_list()
    if num_docs_to_use == 'all':
        text = ' '.join(all_cleaned)
    else:
        num_docs_to_use = int(num_docs_to_use)
        text = ' '.join(all_cleaned[:num_docs_to_use])
    target_text = chunked_text(text, CHUNK_LENGTH_TARGET)
    source_text = [remove_formatting(s) for s in target_text]
    return pd.DataFrame({
        'source_text': pd.Series(source_text),
        'target_text': pd.Series(target_text)
    })


# ====================
def chunked_text(text, n):

    return [''.join(chunk)
            for chunk
            in more_itertools.chunked(list(text), n)]


# ====================
def remove_formatting(string):

    string = string.lower().replace(' ', '').replace('.', '').replace(',', '')
    return string


# ====================
if __name__ == "__main__":

    args = parser.parse_args()

    if args.mode == 'train':
        if args.num_docs_to_use is None:
            raise ValueError("num_docs_to_use is required for training mode")
        if args.outputdir is None:
            raise ValueError("outputdir is required for training mode")
        if args.max_epochs is None:
            raise ValueError("epochs is required for training mode")
        train(args.num_docs_to_use, args.outputdir, args.max_epochs)

    elif args.mode == 'evaluate':
        if args.model_dir is None:
            raise ValueError("model_dir is required for evaluation mode")
        evaluate(args.model_dir)
