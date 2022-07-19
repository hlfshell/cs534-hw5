import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import string
from math import floor
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

VOCAB_PATH = "./vocab.pth"

data = pd.read_csv('./reviews.csv')
print(data.head(10))

tokenizer = get_tokenizer("basic_english")
class Dataset(object):

    def __getitem__(self, index):
        if index >= len(data):
            raise IndexError
        row = data.loc[index]
        summary = str(row["Summary"])
        score = row["Score"]

        summary = summary.lower()
        summary = summary.translate(str.maketrans('','',string.punctuation))

        return tokenizer(summary), score

    def __len__(self):
        return len(data)


base_dataset = Dataset()

# Load or build our vocab object
build = False
try: 
    vocab = torch.load(VOCAB_PATH)
    print("Vocab loaded")
except:
    print("Vocab could not be loaded. Generating now")
    build = True

if build:
    def yield_tokens(data_iter):
        for summary, _ in data_iter:
            yield summary
        
    vocab = build_vocab_from_iterator(yield_tokens(base_dataset), specials=["<unk>"])
    print("Completed building")
    vocab.set_default_index(vocab["<unk>"])
    print("Finished generationg - saving now")
    torch.save(vocab, VOCAB_PATH)

# Build our dataset that uses the vocab now
class VocabDataset(object):

    def __getitem__(self, index):
        if index >= len(data):
            raise IndexError
        row = data.loc[index]
        summary = str(row["Summary"])
        score = row["Score"]

        summary = summary.lower()
        summary = summary.translate(str.maketrans('','',string.punctuation))

        return vocab(tokenizer(summary)), score

    def __len__(self):
        return len(data)

# Calculate the largest max words!
max_words = 0
for summary, _ in VocabDataset():
    if len(summary) > max_words:
        max_words = len(summary)

print("Max words - ", max_words)