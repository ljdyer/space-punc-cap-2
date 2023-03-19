from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, tokenizer, dataset, type_path, max_len=512):
        self.data = dataset[type_path]
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.tokenizer.max_length = max_len
        self.tokenizer.model_max_length = max_len
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        print("Building dataset...")
        for idx in range(len(self.data)):
            # {
            # 'tokens': ['Karl', 'Ove', 'Knausgård', '(', 'born', '1968', ')'],
            # 'ner_tags': [1, 2, 2, 0, 0, 0, 0],
            # 'langs': ['en', 'en', 'en', 'en', 'en', 'en', 'en'],
            # 'spans': ['PER: Karl Ove Knausgård']
            # }
            current_utterance = self.data[idx]
            tokens = current_utterance["tokens"]
            spans = current_utterance["spans"]
            input_, target = " ".join(tokens), "; ".join(spans)

            input_ = input_.lower() + ' </s>'
            target = target.lower() + " </s>"

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
