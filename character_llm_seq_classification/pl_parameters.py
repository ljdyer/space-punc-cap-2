args_dict = dict(
    data_dir="wikiann",  # path for data files
    output_dir="./outputs",  # path to save the checkpoints
    max_seq_length=128,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=1,
    gradient_accumulation_steps=16,
    n_gpu=[1],
    early_stop_callback=False,
    fp_16=True,  # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1',
    # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)
