import argparse
from tabulate import tabulate
import pandas as pd
import more_itertools
import socket
from sklearn.model_selection import train_test_split
from simplet5 import SimpleT5
from typing import List
from pathlib import Path
import re

if socket.gethostname() == 'Laurences-MacBook-Air.local':
    test_path = 'ted_test.csv'
else:
    test_path = '/data/ldyer/ted_test.csv'

CHUNK_LENGTH_TARGET = 100
SOURCE_MAX_TOKEN_LEN = 100 
TARGET_MAX_TOKEN_LEN = 150
NUM_TEST_SAMPLES = 10

CHUNK_LENGTH_PREDICT = 100
CHUNKER_NUM_PREFIX_WORDS = 5

parser = argparse.ArgumentParser(
    description='Evaluate SimpletT5 models for feature restoration',
    allow_abbrev=False
)
parser.add_argument('outputsdir', metavar='O', type=str)

SOD = '▶'
EOD = '◀' 



# ====================
def single_match_single_group(regex: str, string: str):

    find_all = re.findall(regex, string)
    if len(find_all) != 1:
        raise RuntimeError(
            f"Expected a single match, but found {len(find_all)} matches. " +\
            f"RegEx: {regex}; String: {string}."
        )
    return find_all[0]


# ====================
def get_dict_by_value(list_of_dicts, key, value):

    matches = [d for d in list_of_dicts if d[key] == value]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected a single match, but found {len(matches)} matches. " +\
            f"Key: {key}; Value: {value}"
        )
    return matches[0]


# ====================
def get_model_info(outputsdir: str) -> dict:

    outputsdir = Path(outputsdir)
    models = [m.name for m in outputsdir.glob('*')]
    model_info = []
    for model_name in models:
        model_info.append({
            'name': model_name,
            'epoch': int(single_match_single_group(r'epoch-([\d\.]*)', model_name)),
            'train_loss': float(single_match_single_group(r'train-loss-([\d\.]*)', model_name)),
            'val_loss': float(single_match_single_group(r'val-loss-([\d\.]*)', model_name)),
        })
    return sorted(model_info, key=lambda x: x['epoch'])


# ====================
def select_model(outputsdir: str) -> str:

    model_info = get_model_info(outputsdir)
    model_info = [{'option': i, **m} for i, m in enumerate(model_info)]
    print(tabulate(model_info))
    choice = int(input('Which model do you wish to evaluate? '))
    chosen_model = get_dict_by_value(model_info, 'option', choice)
    return chosen_model['name']


# # ====================
# def evaluate_quick(model_dir):

#     print(f'Loading test data from {test_path}...')
#     test_df = load_and_prep_df(test_path, 'all')
#     test_data = test_df.sample(NUM_TEST_SAMPLES).to_dict(orient='records')
#     model = model_from_path(model_dir)
#     for t in test_data:
#         input = t['source_text']
#         reference = t['target_text']
#         hypothesis = model.predict(t['source_text'])[0]
#         print(f"Input:      {input}")
#         print(f"Reference:  {reference}")
#         print(f"Hypothesis: {hypothesis}")


# # ====================
# def evaluate_full(model_dir, num_docs_to_use):

#     print(f'Loading test data from {test_path}...')
#     test_docs = pd.read_csv(test_path)[['no_spaces', 'all_cleaned']].to_dict(orient='records')
#     model = model_from_path(model_dir)
#     if num_docs_to_use != 'all':
#         num_docs_to_use = int(num_docs_to_use)
#         test_docs = test_docs[:num_docs_to_use]
#     result = pd.DataFrame()
#     for doc in test_docs:
#         input = doc['no_spaces']
#         reference = doc['all_cleaned']
#         hypothesis = predict_doc(model, input)
#         print(f'Input:\n{input}\n\n')
#         print(f'Reference:\n{reference}\n\n')
#         print(f'Hypothesis:\n{hypothesis}\n\n')
#         print('====================')
#         result = result.append({'input': input, 'reference': reference, 'hypothesis': hypothesis}, ignore_index=True)
#         result.to_csv(Path(model_dir) / f'evaluate_full_{num_docs_to_use}.csv')


# # ====================
# def predict_doc(model, doc):

#     all_output: List[str] = []
#     prefix = ''
#     while doc:
#         restore_until = CHUNK_LENGTH_PREDICT - len(prefix)
#         text_to_restore = prefix + doc[:restore_until]
#         doc = doc[restore_until:]
#         print(f"Chars remaining to process: {len(doc)}")
#         chunk_restored: str = model.predict(text_to_restore)[0]
#         chunk_restored_split: List[str] = chunk_restored.split(' ')
#         prefix = remove_formatting(' '.join(chunk_restored_split[-CHUNKER_NUM_PREFIX_WORDS:]))
#         all_output.extend(chunk_restored_split[:-CHUNKER_NUM_PREFIX_WORDS])
#     output = ' '.join(all_output)
#     # Add any text remaining in 'prefix'
#     if prefix:
#         prefix_restored = model.predict(prefix)[0]
#         output = output + ' ' + prefix_restored.strip()
#     return output
    
    
# # ====================
# def predict(model_dir):

#     model = model_from_path(model_dir)
#     while True:
#         input_ = input("Enter text to restore formatting to (or 'x' to exit):\n")
#         if input_.lower() == 'x':
#             return
#         print('\nPrediction:')
#         print(model.predict(SOD + input_ + EOD)[0])
#         print()


# # ====================
# def load_and_prep_df(csv_path, num_docs_to_use):

#     all_cleaned = pd.read_csv(csv_path)['all_cleaned'].to_list()
#     all_cleaned = [SOD + doc + EOD for doc in all_cleaned]
#     if num_docs_to_use == 'all':
#         text = ' '.join(all_cleaned)
#     else:
#         num_docs_to_use = int(num_docs_to_use)
#         text = ' '.join(all_cleaned[:num_docs_to_use])
#     target_text = chunked_text(text, CHUNK_LENGTH_TARGET)
#     source_text = [remove_formatting(s) for s in target_text]
#     target_text = [s.replace(SOD, '').replace(EOD, '') for s in target_text]
#     return pd.DataFrame({
#         'source_text': pd.Series(source_text),
#         'target_text': pd.Series(target_text)
#     })


# # ====================
# def chunked_text(text, n):

#     return [''.join(chunk)
#             for chunk
#             in more_itertools.chunked(list(text), n)]


# # ====================
# def remove_formatting(string):

#     string = string.lower().replace(' ', '').replace('.', '').replace(',', '')
#     return string


# # ====================
# def model_from_path(path):

#     print(f'Loading model from {path}...')
#     model = SimpleT5()
#     model.load_model("byt5", path)
#     return model


# ====================
if __name__ == "__main__":

    args = parser.parse_args()
    outputsdir = args.outputsdir
    
    model_dir = select_model(outputsdir)
    print(model_dir)

    
