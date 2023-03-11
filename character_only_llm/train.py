from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
import pytorch_lightning as pl
from character_only_llm.byt5_model import ByT5Model
from character_only_llm.data import fetch_summarisation_data, parse_summarisation_data, generate_model_splits

pl.seed_everything(42)
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')


def test_simple_forward_pass(model) -> None:
    """

    :return:
    """

    input_ids = torch.tensor(
        [list("Life is like a box of chocolates.".encode("utf-8"))]) + 3  # add 3 for special tokens
    labels = torch.tensor(
        [list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + 3  # add 3 for special tokens

    loss = model(input_ids, labels=labels).loss  # forward pass
    print(loss)


def test_forward_pass_on_a_task(task: str, model) -> None:
    """
    This function will take a task and run a forward pass on it.
    :return:
    """

    input_ids = torch.tensor(
        [list(f"{task}: This is a machine learning example.".encode("utf-8"))]) + 3  # add 3 for special tokens
    labels = torch.tensor(
        [list("Dis na ɛgzampul fɔ lan bɔt mashin.".encode("utf-8"))]) + 3  # add 3 for special tokens

    loss = model(input_ids, labels=labels).loss  # forward pass
    print(loss)


def test_basic_inference_on_a_task(model: PreTrainedModel,
                                   tokenizer: PreTrainedTokenizer,
                                   task: str,
                                   text: str,
                                   max_length: int = 512,
                                   num_return_sequences: int = 1,
                                   num_beams: int = 2,
                                   top_k: int = 50,
                                   top_p: float = 0.95,
                                   repetition_penalty: float = 2.5,
                                   length_penalty: float = 1.0,
                                   early_stopping: bool = True,
                                   skip_special_tokens: bool = True,
                                   clean_up_tokenization_spaces: bool = True
                                   ) -> list:
    """
    Generate the result of a task.
    :return:
    """
    input_ids = tokenizer.encode(
        f"{task}: {text}", return_tensors="pt", add_special_tokens=True
    )
    input_ids = input_ids
    generated_ids = model.generate(
        input_ids=input_ids,
        num_beams=num_beams,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        early_stopping=early_stopping,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
    )
    predictions = [
        tokenizer.decode(
            g,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        for g in generated_ids
    ]
    return predictions


if __name__ == '__main__':
    model = ByT5Model()
    train_df, val_df, eval_df = generate_model_splits(parse_summarisation_data(fetch_summarisation_data()))
    # train
    model.train(
        train_df=train_df[:5000],
        eval_df=eval_df[:100],
        val_df=val_df[:100],
        source_max_token_len=128,
        target_max_token_len=50,
        batch_size=8,
        max_epochs=3)
