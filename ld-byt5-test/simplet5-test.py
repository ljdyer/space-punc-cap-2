import argparse
from tabulate import tabulate
import pandas as pd
import more_itertools
import socket
from sklearn.model_selection import train_test_split
from simplet5 import SimpleT5
from typing import List
from pathlib import Path
from datetime import datetime
from fre import FeatureRestorationEvaluator
from helper import tabulate_list_of_dicts, get_dict_by_value
import re

if socket.gethostname() == 'Laurences-MacBook-Air.local':
    test_path = 'ted_test.csv'
else:
    test_path = '/data/ldyer/ted_test.csv'

CHUNK_LENGTH_TARGET = 118
NUM_TEST_SAMPLES = 10
CHUNK_LENGTH_PREDICT = 80
CHUNKER_NUM_PREFIX_WORDS = 5
AVG_CHARS_PROCESSED = 60

parser = argparse.ArgumentParser(
    description='Evaluate SimpletT5 models for feature restoration',
    allow_abbrev=False
)
parser.add_argument(
    'outputsdir', type=str,
    help='The folder where the model outputs are stored'
)

SOD = '▶'
EOD = '◀' 


# ====================
def current_timestamp():

    return datetime.now().strftime('%d%b%Y_%H%M%S')


# ====================
def single_match_single_group(regex: str, string: str):

    find_all = re.findall(regex, string)
    if len(find_all) != 1:
        return None
    else:
        return find_all[0]


# ====================
def get_model_info(outputsdir: str) -> dict:

    outputsdir = Path(outputsdir)
    models = [m.name for m in outputsdir.glob('*')]
    model_info = []
    for model_name in models:
        try:
            epoch = int(single_match_single_group(r'epoch-([\d\.]*)', model_name)),
            train_loss = float(single_match_single_group(r'train-loss-([\d\.]*)', model_name))
            val_loss = float(single_match_single_group(r'val-loss-([\d\.]*)', model_name))
        except:
            continue
        model_info.append({
            'name': model_name,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
        })
    if len(model_info) < 1:
        raise ValueError('No models found in directory!')
    return sorted(model_info, key=lambda x: x['epoch'])


# ====================
def select_model(outputsdir: str) -> str:

    model_info = get_model_info(outputsdir)
    model_info = [{'option': i, **m} for i, m in enumerate(model_info)]
    print(tabulate_list_of_dicts(model_info))
    choice = int(input('Which model do you wish to evaluate? '))
    chosen_model = get_dict_by_value(model_info, 'option', choice)
    return Path(outputsdir) / chosen_model['name']


# ====================
def evaluate_quick(model_dir):

    print(f'Loading test data from {test_path}...')
    test_df = load_and_prep_df(test_path, 'all')
    test_data = test_df.sample(NUM_TEST_SAMPLES).to_dict(orient='records')
    model = model_from_path(model_dir)
    for t in test_data:
        input_ = t['source_text']
        reference = t['target_text']
        hypothesis = model.predict(input_)[0]
        print(f"Input:      {input_}")
        print(f"Reference:  {reference}")
        print(f"Hypothesis: {hypothesis}")


# ====================
def evaluate_full(model_dir):

    test_docs = pd.read_csv(test_path)[['no_spaces', 'all_cleaned']].to_dict(orient='records')
    num_docs_to_use = int(input(
        f"There are {len(test_docs)} test documents available. How many would you like to use? (Input an integer, or 'all'): "
    ))
    if num_docs_to_use != 'all':
        num_docs_to_use = int(num_docs_to_use)
        test_docs = test_docs[:num_docs_to_use]
    model = model_from_path(model_dir)
    results_path = Path(model_dir) / f"results_{current_timestamp()}.csv"
    metrics_path = Path(model_dir) / f"metrics_{current_timestamp()}.csv"
    results = pd.DataFrame()
    for doc in test_docs:
        input_ = SOD + doc['no_spaces'] + EOD
        reference = doc['all_cleaned']
        hypothesis = predict_doc(model, input_)
        print(f'Input:\n{input_}\n\n')
        print(f'Reference:\n{reference}\n\n')
        print(f'Hypothesis:\n{hypothesis}\n\n')
        print('====================')
        results = results.append({'input': input_, 'reference': reference, 'hypothesis': hypothesis}, ignore_index=True)
        results.to_csv(results_path, index=False)
    fre = FeatureRestorationEvaluator(
        results['reference'],
        results['hypothesis'],
        capitalization=True,
        feature_chars='., '
    )
    prfs = pd.DataFrame(fre.get_prfs()).transpose()
    print(prfs)
    prfs.to_csv(metrics_path, index=False)


# ====================
def predict_doc(model, doc):

    all_output: List[str] = []
    prefix = ''
    while doc:
        restore_until = CHUNK_LENGTH_PREDICT - len(prefix)
        text_to_restore = prefix + doc[:restore_until]
        doc = doc[restore_until:]
        print(f"Chars remaining to process: {len(doc)}")
        chunk_restored: str = model.predict(text_to_restore)[0].strip().lstrip('.,')
        with open('output.txt', 'a', encoding='utf-8') as f:
            f.write(text_to_restore + '\n')
            f.write(chunk_restored + '\n')
        chunk_restored = match_chars_2(text_to_restore.replace(SOD, '').replace(EOD, ''), chunk_restored)
        chunk_restored_split: List[str] = chunk_restored.split(' ')
        prefix = remove_formatting(' '.join(chunk_restored_split[-CHUNKER_NUM_PREFIX_WORDS:]))
        all_output.extend(chunk_restored_split[:-CHUNKER_NUM_PREFIX_WORDS])
    output = ' '.join(all_output)
    # Add any text remaining in 'prefix'
    if prefix:
        prefix_restored = model.predict(prefix)[0].strip().lstrip('.,')
        prefix_restored = match_chars_2(prefix.replace(SOD, '').replace(EOD, ''), prefix_restored)
        with open('output.txt', 'a', encoding='utf-8') as f:
            f.write(prefix + '\n')
            f.write(prefix_restored + '\n')
        output = output + ' ' + prefix_restored
    output = output.strip().lstrip('., ')
    output = normalize(output, '., ')
    return output


# ====================
def normalize(str_, chars):

    for char in list(chars):
        str_ = re.sub(rf'{re.escape(char)}+', char, str_)
    return str_


# ====================
def remove_formatting(str_):

    str_ = str_.lower()
    str_ = remove_chars(str_, '., ')
    return str_


# ====================
def remove_chars(str_, chars):

    return str_.translate({ord(c): None for c in chars})


# ====================
def match_chars(input_, hypothesis) -> str:

    orig = input_
    input_ = list(input_)
    hypothesis = list(hypothesis)
    hypothesis_output = []
    while input_:
        next_input_char = input_.pop(0)
        while hypothesis:
            next_hypothesis_char = hypothesis.pop(0)
            if next_hypothesis_char.lower() not in [' ', '.', ',', next_input_char.lower()]:
                hypothesis_output.append(next_input_char)
                with open('warning_log.txt', 'a', encoding='utf-8') as f:
                    f.write(f'WARNING: Unexpected character: {next_hypothesis_char}. Expected: {next_input_char}')
                break
            hypothesis_output.append(next_hypothesis_char)
            if next_hypothesis_char.lower() == next_input_char.lower():
                break
    while hypothesis:
        next_hypothesis_char = hypothesis.pop(0)
        if next_hypothesis_char not in [' ', ',', '.']:
            break
        hypothesis_output.append(next_hypothesis_char)
    input_chars = orig.lower().replace(' ', '').replace('.', '').replace(',', '')
    hypothesis_output = ''.join(hypothesis_output)
    output_chars = hypothesis_output.lower().replace(' ', '').replace('.', '').replace(',', '')
    assert input_chars == output_chars
    return hypothesis_output


# ====================
def match_chars_2(input_, hypothesis) -> str:

    input_ = remove_formatting(input_)
    chars_to_add = ''
    for hyp_matches_up_to in range(len(hypothesis) + 1):
        hyp_chars_so_far = remove_formatting(hypothesis[:hyp_matches_up_to])
        if not input_.startswith(hyp_chars_so_far):
            hyp_matches_up_to -= 1
            chars_to_add = input_[len(hyp_chars_so_far) - 1:]
            break
    if len(chars_to_add) > 1:
        warning_string = f'WARNING: adding the following characters unformatted as they do not appear in the hypothesis: {chars_to_add}'
        print(warning_string)
        with open('warning_log.txt', 'a', encoding='utf-8') as f:
            f.write(warning_string)
    output = hypothesis[:hyp_matches_up_to] + chars_to_add
    assert remove_formatting(output) == input_
    return output

    
# ====================
def predict(model_dir):

    model = model_from_path(model_dir)
    while True:
        input_ = input("Enter text to restore formatting to (or 'x' to exit):\n")
        if input_.lower() == 'x':
            return
        print('\nPrediction:')
        print(model.predict(input_)[0])
        print()


# ====================
def load_and_prep_df(csv_path, num_docs_to_use):

    all_cleaned = pd.read_csv(csv_path)['all_cleaned'].to_list()
    all_cleaned = [SOD + doc + EOD for doc in all_cleaned]
    if num_docs_to_use == 'all':
        text = ' '.join(all_cleaned)
    else:
        num_docs_to_use = int(num_docs_to_use)
        text = ' '.join(all_cleaned[:num_docs_to_use])
    target_text = chunked_text(text, CHUNK_LENGTH_TARGET)
    source_text = [remove_formatting(s) for s in target_text]
    target_text = [s.replace(SOD, '').replace(EOD, '') for s in target_text]
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
def model_from_path(path):

    print(f'Loading model from {path}...')
    model = SimpleT5()
    model.load_model("byt5", path)
    return model


# ====================
if __name__ == "__main__":

    args = parser.parse_args()
    outputsdir = args.outputsdir
    
    model_dir = select_model(outputsdir)
    print(model_dir)
    print()

    OPTIONS = {
        1: ('Quick evaluate', evaluate_quick),
        2: ('Full evaluate', evaluate_full),
        3: ('Free predict', predict),
    }
    for i, o in OPTIONS.items():
        print(f"{i}. {o[0]}")
    chosen_option = OPTIONS[int(input('What would you like to do? '))]
    chosen_option[1](model_dir)
    
