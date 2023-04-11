import argparse
from tabulate import tabulate
import pandas as pd
import more_itertools
import socket
from sklearn.model_selection import train_test_split
from my_simplet5 import SimpleT5
from typing import List
from pathlib import Path
from fre import FeatureRestorationEvaluator
import re
from tqdm import tqdm
import time
from helper import *
from constants import *

parser = argparse.ArgumentParser(
    description='Evaluate SimpletT5 models for feature restoration',
    allow_abbrev=False
)

parser.add_argument('outputsdir', type=str, help='The folder where the model outputs are stored.')
parser.add_argument('language', type=str, help='The language of the model.')


# ====================
def get_model_info(outputsdir: str) -> dict:

    outputsdir = Path(outputsdir)
    models = [m.name for m in outputsdir.glob('*')]
    model_info = []
    for model_name in models:
        try:
            epoch = int(single_match_single_group(r'epoch-([\d\.]*)', model_name))
            if isinstance(epoch, tuple):
                epoch = epoch[0]
            train_loss = float(single_match_single_group(r'train-loss-([\d\.]*)', model_name))
            val_loss = float(single_match_single_group(r'val-loss-([\d\.]*)', model_name))
        except:
            continue
        model_info.append({
            'name': model_name,
            'epoch': epoch,
            'inference_carried_out': 'YES' if Path.exists(outputsdir / model_name / 'inference_results.csv') else '',
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
def get_metrics(model_dir, lang):

    results_path = Path(model_dir) / f"inference_results.csv"
    metrics_path = Path(model_dir) / f"metrics.csv"
    wer_path = Path(model_dir) / f"wer.csv"
    results = pd.read_csv(results_path)
    get_metrics_(results, metrics_path, wer_path, lang)


# ====================
def get_metrics_(results, metrics_path, wer_path, lang):

    num_results = len(results)
    print(f'Number of results before removing errors: {num_results}')
    results = results[results['hypothesis'] != "ERROR"]
    num_results = len(results)
    print(f'Number of results after removing errors: {num_results}')
    fre = FeatureRestorationEvaluator(
        results['reference'],
        results['hypothesis'],
        capitalization=CAPITALIZATION[lang],
        feature_chars=FEATURE_CHARS[lang],
    )
    prfs = pd.DataFrame(fre.get_prfs()).transpose()
    lines = []
    for feature in PRF_COLUMNS[lang]:
        lines.append(
            '& ' + ' & '.join(
                [str(round(prfs.loc[feature, metric] * 100)) for metric in ['Precision', 'Recall', 'F-score']]
            )
        )
    print()
    print('\n'.join(lines))
    wer_info = pd.DataFrame([fre.wer_info['all']]).transpose()
    print(prfs)
    print(wer_info)
    prfs.to_csv(metrics_path)
    wer_info.to_csv(wer_path)


# ====================
def evaluate_quick(model_dir, lang):

    test_path = TEST_PATH[lang]
    print(f'Loading test data from {test_path}...')
    test_df = load_and_prep_df(test_path, 'all', lang)
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
def get_test_docs(lang):

    test_path = TEST_PATH[lang]
    test_df = pd.read_csv(test_path)
    ref_col_name = ALL_CLEANED_COL_NAME[lang]
    reference = test_df[ref_col_name].to_list()
    input_col_name = INPUT_COL_NAME[lang]
    if input_col_name is not None:
        input_ = test_df[input_col_name].to_list()
    else:
        input_ = [remove_formatting(doc, lang) for doc in reference]
    return [{'input': i, 'reference': r} for i, r in zip(input_, reference)]
    

# ====================
def evaluate_full(model_dir, lang):

    test_docs = get_test_docs(lang)
    num_docs_to_use = input(
        f"There are {len(test_docs)} test documents available. How many would you like to use? (Input an integer, a range, 'all', or 'patch'): "
    )
    if num_docs_to_use == 'patch':
        results_path = Path(model_dir) / f"inference_results.csv"
        results = pd.read_csv(results_path)
        errored = results[results['hypothesis'] == 'ERROR']
        print(errored)
        for doc_idx, row in errored.iterrows():
            input_ = row.input
            reference = test_docs[doc_idx]['reference']
            inference_log_path = Path('inference_logs') / f"{model_dir.name}" / f"{str(doc_idx)}_{current_timestamp()}"
            inference_log_path.mkdir(parents=True)
            model = model_from_path(model_dir)
            hypothesis = predict_doc(model, input_, inference_log_path)
            results.at[doc_idx, 'reference'] = reference
            results.at[doc_idx, 'hypothesis'] = hypothesis
            results.to_csv(results_path, index=False)
        return
    if num_docs_to_use != 'all':
        if num_docs_to_use.isnumeric():
            num_docs_to_use = int(num_docs_to_use)
            test_docs = test_docs[:num_docs_to_use]
        else:
            first, last = re.findall(r'(\d+)-(\d+)', num_docs_to_use)[0]
            test_docs = test_docs[int(first):int(last)+1]
    model = model_from_path(model_dir)
    results_path = Path(model_dir) / f"inference_results.csv"
    metrics_path = Path(model_dir) / f"metrics.csv"
    wer_path = Path(model_dir) / f"wer.csv"
    results = pd.DataFrame()
    predict_doc = predict_doc_spaces if SPACED[lang] is True else predict_doc_no_spaces
    for doc_idx, doc in tqdm(enumerate(test_docs)):
        try:
            inference_log_path = Path('inference_logs') / f"{model_dir.name}" / f"{str(doc_idx)}_{current_timestamp()}"
            inference_log_path.mkdir(parents=True)
            input_ = SOD + doc['input'] + EOD
            reference = doc['reference']
            hypothesis = predict_doc(model, input_, inference_log_path, lang)
            results = results.append({'doc_idx': doc_idx, 'input': input_, 'reference': reference, 'hypothesis': hypothesis}, ignore_index=True)
            results.to_csv(results_path, index=False)
        except:
            results = results.append({'doc_idx': doc_idx, 'input': input_, 'reference': 'ERROR', 'hypothesis': 'ERROR'}, ignore_index=True)
            results.to_csv(results_path, index=False)
    get_metrics_(results, metrics_path, wer_path, lang)


# ====================
def evaluate_lengths(model_dir, lang):

    for chunk_length in CHUNK_LENGTH_TEST[lang]:
        test_docs = get_test_docs(lang)[:100]
        model = model_from_path(model_dir)
        results_path = Path(model_dir) / f"check_results.csv"
        metrics_path = Path(model_dir) / f"check_metrics.csv"
        wer_path = Path(model_dir) / f"check_wer.csv"
        results = pd.DataFrame()
        predict_doc = predict_doc_spaces if SPACED[lang] is True else predict_doc_no_spaces
        for doc_idx, doc in tqdm(enumerate(test_docs)):
            try:
                inference_log_path = Path('inference_logs') / f"{model_dir.name}" / f"{str(doc_idx)}_{current_timestamp()}"
                inference_log_path.mkdir(parents=True)
                input_ = SOD + doc['input'] + EOD
                reference = doc['reference']
                hypothesis = predict_doc(model, input_, inference_log_path, lang, chunk_length)
                results = results.append({'doc_idx': doc_idx, 'input': input_, 'reference': reference, 'hypothesis': hypothesis}, ignore_index=True)
            except:
                results = results.append({'doc_idx': doc_idx, 'input': input_, 'reference': 'ERROR', 'hypothesis': 'ERROR'}, ignore_index=True)
        get_metrics_(results, metrics_path, wer_path, lang)
        fre = FeatureRestorationEvaluator(
            results['reference'],
            results['hypothesis'],
            capitalization=CAPITALIZATION[lang],
            feature_chars=FEATURE_CHARS[lang],
        )
        prfs = pd.DataFrame(fre.get_prfs()).transpose()
        print(chunk_length)
        print(prfs)
        print()
        with open(results_path, 'a') as f:
            f.write(str(chunk_length))
            f.write('\n')
            f.write(str(prfs))
            f.write('\n')
            f.write('\n')
        print()


# ====================
def predict_doc_spaces(model, doc, inference_log_path, lang, chunk_length_predict=None):

    reached_eod = False
    start_time = time.time()
    warning_log_path = inference_log_path / 'warnings.txt'
    output_log_path = inference_log_path / 'outputs.txt'
    num_prefix_words = CHUNKER_NUM_PREFIX_WORDS[lang]
    if chunk_length_predict is None:
        chunk_length_predict = CHUNK_LENGTH_PREDICT[lang]
    all_output: List[str] = []
    prefix = ''
    total_chars = len(doc)
    expected_rounds = total_chars // AVG_CHARS_PROCESSED
    with tqdm(total=expected_rounds) as pbar:
        while doc:
            current_time = time.time()
            if current_time - start_time > 60 * TIMEOUT_MINS:
                write_to_log(warning_log_path, 'ABORTED DUE TO TIMEOUT')
                raise RuntimeError()
            restore_until = chunk_length_predict - len(prefix)
            text_to_restore = prefix + doc[:restore_until]
            if reached_eod:
                text_to_restore = text_to_restore + EOD
            if EOD in text_to_restore:
                reached_eod = True
            doc = doc[restore_until:]
            chunk_restored: str = model.predict(text_to_restore)[0].strip().lstrip('., ')
            chunk_restored = match_chars(text_to_restore.replace(SOD, '').replace(EOD, ''), chunk_restored, warning_log_path, lang)
            write_to_log(output_log_path, [text_to_restore, chunk_restored])
            chunk_restored_split: List[str] = chunk_restored.split(' ')
            prefix, new_output =\
                (chunk_restored_split[-num_prefix_words:],
                chunk_restored_split[:-num_prefix_words])
            if len(new_output) < 1:
                new_output = prefix[:1]
                prefix = prefix[1:]
            all_output.extend(new_output)
            prefix = remove_formatting(' '.join(prefix), lang)
            pbar.update()
            pbar.set_description(f'Remaining: {len(doc)}/{total_chars}')
    output = ' '.join(all_output)
    # Add any text remaining in 'prefix'
    if prefix:
        prefix = prefix + EOD
        prefix_restored = model.predict(prefix)[0].strip().lstrip('., ')
        prefix_restored = match_chars(prefix.replace(SOD, '').replace(EOD, ''), prefix_restored, warning_log_path, lang)
        write_to_log(output_log_path, [prefix, prefix_restored])
        output = output + ' ' + prefix_restored
    output = output.strip().lstrip('., ')
    output = normalize(output, '., ')
    return output


# ====================
def predict_doc_no_spaces(model, doc, inference_log_path, lang):

    feature_chars = FEATURE_CHARS[lang]
    chunker_num_prefix_chars = CHUNKER_NUM_PREFIX_CHARS[lang]
    reached_eod = False
    start_time = time.time()
    warning_log_path = inference_log_path / 'warnings.txt'
    output_log_path = inference_log_path / 'outputs.txt'
    all_output: List[str] = []
    prefix = ''
    total_chars = len(doc)
    expected_rounds = total_chars // AVG_CHARS_PROCESSED
    with tqdm(total=expected_rounds) as pbar:
        while doc:
            current_time = time.time()
            if current_time - start_time > 60 * TIMEOUT_MINS:
                write_to_log(warning_log_path, 'ABORTED DUE TO TIMEOUT')
                raise RuntimeError()
            restore_until = CHUNK_LENGTH_PREDICT['ja'] - len(prefix)
            text_to_restore = prefix + doc[:restore_until]
            if reached_eod:
                text_to_restore = text_to_restore + EOD
            if EOD in text_to_restore:
                reached_eod = True
            doc = doc[restore_until:]
            chunk_restored: str = model.predict(text_to_restore)[0].strip().lstrip(feature_chars)
            chunk_restored = match_chars(text_to_restore.replace(SOD, '').replace(EOD, ''), chunk_restored, warning_log_path, lang)
            write_to_log(output_log_path, [text_to_restore, chunk_restored])
            prefix, new_output =\
                (chunk_restored[-chunker_num_prefix_chars:],
                chunk_restored[:-chunker_num_prefix_chars])
            if len(new_output) < 1:
                new_output = prefix[:1]
                prefix = prefix[1:]
            all_output.extend(new_output)
            prefix = remove_formatting(''.join(prefix), lang)
            pbar.update()
            pbar.set_description(f'Remaining: {len(doc)}/{total_chars}')
    output = ''.join(all_output)
    # Add any text remaining in 'prefix'
    if prefix:
        prefix = prefix + EOD
        prefix_restored = model.predict(prefix)[0].strip().lstrip(feature_chars)
        prefix_restored = match_chars(prefix.replace(SOD, '').replace(EOD, ''), prefix_restored, warning_log_path, lang)
        write_to_log(output_log_path, [prefix, prefix_restored])
        output = output + prefix_restored
    output = output.strip().lstrip(feature_chars)
    output = normalize(output, feature_chars)
    return output


# ====================
def match_chars(input_, hypothesis, warning_log_path, lang) -> str:

    capitalization = CAPITALIZATION[lang]
    feature_chars = FEATURE_CHARS[lang]
    output = []
    hypothesis_orig = hypothesis
    for i in input_:
        if capitalization is True:
            char_match_re = rf'(?:{i}|{i.upper()})[{feature_chars}]*'
        else:
            char_match_re = rf'{re.escape(i)}[{feature_chars}]*'
        ch_match = re.search(char_match_re, hypothesis)
        if ch_match:
            output.append(ch_match.group())
            if ch_match.start() != 0:
                skipped_chars = hypothesis[0: ch_match.start()]
                write_to_log(warning_log_path,
                    [f'WARNING: Skipped chars: "{skipped_chars}" as they do not appear in the input.', input_, hypothesis_orig])
            hypothesis = hypothesis[ch_match.end():]
        else:
            output.append(i)
            write_to_log(warning_log_path, 
                [f'WARNING: Added char: "{i}" as it does not appear in the output.', input_, hypothesis_orig])
    output = ''.join(output)
    if remove_formatting(output, lang) != input_:
        chars_to_add = input_[len(remove_formatting(output, lang)):]
        output = output + chars_to_add
        write_to_log(warning_log_path,
            [f'WARNING: Added chars: "{chars_to_add}" to end of output as they were not present.', input_, hypothesis_orig])
    assert remove_formatting(output, lang) == input_
    return output
    

# ====================
def predict(model_dir, lang):

    model = model_from_path(model_dir)
    while True:
        input_ = input("Enter text to restore formatting to (or 'x' to exit):\n")
        input_ = SOD + input_ + EOD
        if input_.lower() == 'x':
            return
        print('\nPrediction:')
        print(model.predict(input_)[0])
        print()


# ====================
def model_from_path(path):

    print(f'Loading model from {path}...')
    model = SimpleT5()
    model.load_model("byt5", path, use_gpu=True)
    return model


# ====================
if __name__ == "__main__":

    args = parser.parse_args()
    outputsdir = args.outputsdir
    lang = args.language
    
    model_dir = select_model(outputsdir)
    print(model_dir)
    print()

    OPTIONS = {
        1: ('Quick evaluate', evaluate_quick),
        2: ('Full evaluate', evaluate_full),
        3: ('Free predict', predict),
        4: ('Get metrics', get_metrics),
        5: ('Check lengths', evaluate_lengths)
    }
    for i, o in OPTIONS.items():
        print(f"{i}. {o[0]}")
    chosen_option = OPTIONS[int(input('What would you like to do? '))]
    chosen_option[1](model_dir, lang)
