import glob
import os
import json
import time
import random
import re
from itertools import chain
from string import punctuation

import nltk
from transformers import AutoTokenizer
from datasets import load_dataset
from character_llm_seq_classification.ner_dataset import NERDataset

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
pl.seed_everything(42)

if __name__ == "__main__":
    dataset = load_dataset("wikiann", "en")
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
    input_dataset = NERDataset(tokenizer=tokenizer, dataset=dataset, type_path='train')
