import pandas as pd
import more_itertools
import socket
from simplet5 import SimpleT5

if socket.gethostname() == 'Laurences-MacBook-Air.local':
    train_path, test_path = 'ted_train.csv', 'ted_test.csv'
else:
    train_path, test_path = '/data/ldyer/ted_train.csv', '/data/ldyer/ted_test.csv'


# ====================
def load_and_prep_df(csv_path, num_docs_to_use):

    all_cleaned = pd.read_csv(csv_path)['all_cleaned'].to_list()
    text = ' '.join(all_cleaned[:num_docs_to_use])
    target_text = chunked_text(text, 100)
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

    train_df = load_and_prep_df(train_path, 10)
    print(train_df)
    test_df = load_and_prep_df(test_path, 1)
    print(test_df)
    model = SimpleT5()
    model.from_pretrained(model_type="byt5", model_name="google/byt5-small")
    model.train(train_df=train_df,
                eval_df=test_df, 
                source_max_token_len=100, 
                target_max_token_len=150, 
                batch_size=8, 
                max_epochs=10,
                use_gpu=True
            )


