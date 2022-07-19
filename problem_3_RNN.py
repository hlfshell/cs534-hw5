import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import string
from math import floor
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

VOCAB_PATH = "./vocab.pth"
MAX_WORDS = 24

data = pd.read_csv('./reviews.csv')

tokenizer = get_tokenizer("basic_english")

vocab = torch.load(VOCAB_PATH)
print("Vocab loaded")

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

# Build our training and test datasets
train_size = floor(len(data)*0.7)
test_size = len(data) - train_size

train_dataset, test_dataset = random_split(VocabDataset(), [train_size, test_size])

print(f"size of train_dataset {len(train_dataset)} : {len(test_dataset)}")

train_dataset = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size=10, shuffle=False)

def vectorize_batch(batch):
    Y, X = list(zip(*batch))
    X = [vocab(tokenizer(text)) for text in X]
    # Pad the tokens with 0s if below the MAX_WORDS length
    X = [tokens + ([0]* (MAX_WORDS-len(tokens))) if len(tokens) < MAX_WORDS else tokens[:MAX_WORDS] for tokens in X]

    return torch.tensor(X, dtype=torch.int32), torch.tensor(Y) - 1 ## We have deducted 1 from target names to get them in range [0,1,2,3] from [1,2,3,4]


train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=vectorize_batch, shuffle=True)
test_loader  = DataLoader(test_dataset , batch_size=32, collate_fn=vectorize_batch)


for X, Y in train_loader:
    print(X, Y)
    raise "nope"