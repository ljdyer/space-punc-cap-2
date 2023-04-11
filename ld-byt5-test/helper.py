from tabulate import tabulate
from datetime import datetime
import pandas as pd
import re
import more_itertools
from constants import *


# ====================
def tabulate_list_of_dicts(l):

    headers = {k: k for k in l[0].keys()}
    return tabulate(l, headers=headers)


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
def remove_chars(str_, chars):

    return str_.translate({ord(c): None for c in chars})


# ====================
def normalize(str_, chars):

    for char in list(chars):
        str_ = re.sub(rf'{re.escape(char)}+', char, str_)
    return str_


# ====================
def load_and_prep_df(csv_path, num_docs_to_use, lang):

    col_name = ALL_CLEANED_COL_NAME[lang]
    all_cleaned = pd.read_csv(csv_path)[col_name].to_list()
    print(f"Total number of documents available: {len(all_cleaned)}")
    all_cleaned = [SOD + doc + EOD for doc in all_cleaned]
    if SPACED[lang]:
        joiner = ' '
    else:
        joiner = ''
    if num_docs_to_use == 'all':
        text = joiner.join(all_cleaned)
    else:
        num_docs_to_use = int(num_docs_to_use)
        text = joiner.join(all_cleaned[:num_docs_to_use])
    target_text = chunked_text(text, MAX_TOKEN_LEN_CHARS[lang])
    source_text = [remove_formatting(s, lang) for s in target_text]
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
def remove_formatting(str_, lang):

    feature_chars = FEATURE_CHARS[lang]
    for c in feature_chars:
        str_ = str_.replace(c, '')
    if CAPITALIZATION[lang]:
        str_ = str_.lower()
    return str_


# ====================
def write_to_log(log_path, lines):

    if isinstance(lines, str):
        lines = [lines]
    with open(log_path, 'a', encoding='utf-8') as f:
        for l in lines:
            f.write(l)
            f.write('\n')


# ====================
def utf8len(str_):

    return len(str_.encode('utf-8'))