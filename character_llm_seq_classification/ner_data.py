from character_llm_seq_classification.ner_dataset import NERDataset
from datasets import load_dataset


def get_dataset(tokenizer, type_path, args):
    tokenizer.max_length = args['max_seq_length']
    tokenizer.model_max_length = args['max_seq_length']
    dataset = load_dataset(args['data_dir'], "en")
    return NERDataset(tokenizer=tokenizer, dataset=dataset, type_path=type_path)
