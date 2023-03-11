import pandas as pd
import more_itertools
from simplet5 import SimpleT5
train_path, test_path = 'ted_train.csv', 'ted_test.csv'


# ====================
def load_and_prep_df(csv_path):

    all_cleaned = pd.read_csv(csv_path)['all_cleaned'].to_list()
    first_doc = all_cleaned[0]
    chunked = chunked_text(first_doc, 100)
    target_text = [remove_formatting(s) for s in chunked]
    return pd.DataFrame({
        'source_text': pd.Series(chunked),
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

    train_df = load_and_prep_df(train_path)
    print(train_df)
    test_df = load_and_prep_df(test_path)[:10]
    print(test_df)
    model = SimpleT5()
    model.from_pretrained(model_type="byt5", model_name="google/byt5-small")
    model.train(train_df=train_df,
                eval_df=test_df, 
                source_max_token_len=128, 
                target_max_token_len=50, 
                batch_size=8, 
                max_epochs=3,
                use_gpu=True
            )


