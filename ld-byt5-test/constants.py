# Values used for test_all_10:
# CHUNK_LENGTH_TARGET = 118
# NUM_TEST_SAMPLES = 10
# CHUNK_LENGTH_PREDICT = 80
# CHUNKER_NUM_PREFIX_WORDS = 5
# AVG_CHARS_PROCESSED = 60
# TIMEOUT_MINS = 30

NUM_TEST_SAMPLES = 10

CHUNKER_NUM_PREFIX_WORDS = {
    'en': 5,
    'gu': 5
}

CHUNKER_NUM_PREFIX_CHARS = {
    'ja': 10
}

AVG_CHARS_PROCESSED = 60
TIMEOUT_MINS = 30


SOD = '▶'
EOD = '◀'

TOKEN_LEN_BUFFER = 0

MAX_TOKEN_LEN_BYTES = 256

MAX_TOKEN_LEN_CHARS = {
    # Consider number of bytes in full-width/half-width characters and
    # allow for 1 end of sequence token </s>
    'en': 255,  # 118 used for test_all_10, 127 used later
    'ja': 84,   # 80 used for ja_all_10_2.
                # 84 * 3 = 252
    'gu': 84
}

CHUNK_LENGTH_PREDICT = {
    'en': 215,
    'ja': 70,   # Previously 60
    'gu': 70    # Previously 60
}

CHUNK_LENGTH_TEST = {
    'en': [180, 190, 200, 210, 220],
    'gu': [100, 110]
}

TRAIN_PATH = {
    'en': '/data/ldyer/ted_train.csv',
    'ja': 'data/oshiete_train.csv',
    'gu': 'data/guj_train.csv'
}

TEST_PATH = {
    'en': '/data/ldyer/ted_test.csv',
    'ja': 'data/oshiete_test.csv',
    'gu': 'data/guj_test.csv'
}

ALL_CLEANED_COL_NAME = {
    'en': 'all_cleaned',
    'ja': 'text',
    'gu': 'text'
}

INPUT_COL_NAME = {
    'en': 'no_spaces',
    'ja': None,
    'gu': None
}

FEATURE_CHARS = {
    'en': '., ',
    'ja': '。、？',
    'gu': '., '
}

CAPITALIZATION = {
    'en': True,
    'ja': False,
    'gu': False
}

SPACED = {
    'en': True,
    'ja': False,
    'gu': True
}

PRF_COLUMNS = {
    'en': [' ', 'CAPS', '.', ',', 'all'],
    'ja': ['。', '、', '？', 'all'],
    'gu': [' ', '.', ',', 'all']
}