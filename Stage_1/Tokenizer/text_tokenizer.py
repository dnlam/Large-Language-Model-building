
import numpy as np
import pandas as pd
import re

######## Reading The_Verdict story
with open("The_Verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character is :", len(raw_text))

print("First 100 characters are :", raw_text[96:196])

############ Step 1. Text Tokenization
######## Let's tokenize this text into individual words

results = re.split(r'(\s)', raw_text)
print(results)

######## ... However, some words are still connected to punctuation character
######## ... Thus, we can split the text, to gether with the punctuation and whitespace

preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
print(preprocessed)

######## A small remaining issue that this list includes whitespace characters, Let's remove it

preprocessed =[item.strip() for item in preprocessed if item.strip()]
print(preprocessed)


############ Step 2. Converting tokens into tokens ID, which is an intermediate steps before converting the token IDs
############ embedding vectors

####### Let's first create a vocabulary which contains a list of all unique tokens and is alphabetically sorted.
all_words = sorted(list(set(preprocessed)))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)
print("vocab_size", vocab_size)

# Vocabulary creation
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i > 50:
        break

####### For more convenient, we can use Byte Pair Encoding method for tokenizer
import importlib
import importlib.metadata
import tiktoken
print("tiktoken version:", importlib.metadata.version("tiktoken"))

# Initialize BPE tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
# Encode text with special token
enc_text = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
# print(enc_sample)
# Decode token ids to word
dec_text = tokenizer.decode(enc_text)
# print(dec_sample)

############ Step 3. Create embedding from tokens
####### Data sampling
enc_sample = enc_text[500:]
context_size = 4
# Create input-target pairs from encoded text
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y: {y}")
# Create next word prediction task
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# In order to make the input-targets more efficient, let's do with tensor
import torch
from torch.utils.data import Dataset, DataLoader

# Building input-targets using tiktoken and torch for efficient data loading
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# Build a dataloader from input-target data set
def create_dataloader(txt, batch_size=4,
        max_length=256, stride=128, shuffle=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader