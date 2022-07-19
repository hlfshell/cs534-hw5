from time import perf_counter
from tkinter import HIDDEN
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import string
from math import floor
from torchtext.data import get_tokenizer
from torch.optim import Adam
import torch.nn.functional as F


VOCAB_PATH = "./vocab.pth"
BATCH_SIZE = 4096
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

# train_dataset = DataLoader(train_dataset, batch_size=10, shuffle=True)
# test_dataset = DataLoader(test_dataset, batch_size=10, shuffle=False)

def vectorize_batch(batch):
    X, Y= list(zip(*batch))

    # Pad the tokens with 0s if below the MAX_WORDS length
    X = [tokens + ([0]* (MAX_WORDS-len(tokens))) if len(tokens) < MAX_WORDS else tokens[:MAX_WORDS] for tokens in X]

    # One hot encode the score
    Y = F.one_hot(torch.sub(torch.tensor(Y), 1), num_classes=5).float()

    return torch.tensor(X, dtype=torch.int32), Y


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=vectorize_batch, shuffle=True)
test_loader  = DataLoader(test_dataset , batch_size=BATCH_SIZE, collate_fn=vectorize_batch)


# Now finally we create our RNN class
HIDDEN_DIMENSION = 50
NUMBER_OF_LAYERS = 3


class RNNReviewNetwork(nn.Module):
    def __init__(self):
        super(RNNReviewNetwork, self).__init__()

        # Create the embedding input - up to MAX_WORDS in size
        # (with padding if not used), with embeddings at size of
        # vocabulary
        self.embedding_layer = nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=MAX_WORDS
        )

        # Create the RNN network - HIDDEN_DIMENSION neurons wide
        # for each hidden layer, for NUMBER_OF_LAYERS long. Input
        # size is the max word size of the dataset
        self.rnn = nn.RNN(
            input_size=MAX_WORDS,
            hidden_size=HIDDEN_DIMENSION,
            num_layers=NUMBER_OF_LAYERS,
            batch_first=True
        )
        # Final output layer is just a score 1-5, so we treat them as categories
        self.linear = nn.Linear(HIDDEN_DIMENSION, 5)

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn(
            embeddings,
            torch.randn(NUMBER_OF_LAYERS, len(X_batch), HIDDEN_DIMENSION)
        )
        return self.linear(output[:,-1])

review_model = RNNReviewNetwork()

# Printing out our model
for layer in review_model.children():
    print("Layer : {}".format(layer))
    print("Parameters : ")
    for param in layer.parameters():
        print(param.shape)
    print()


# Now we train!
epochs = 15
learning_rate = 1e-3

loss_fn = nn.CrossEntropyLoss()
rnn_classifier = RNNReviewNetwork()
optimizer = Adam(rnn_classifier.parameters(), lr=learning_rate)

# Loop
for i in range(0, epochs):
    losses = []
    batch = 0
    batch_durations = []
    epoch_start = perf_counter()

    for X, Y in train_loader:
        batch_start = perf_counter()

        Y_preds = review_model(X)

        loss = loss_fn(Y_preds, Y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_complete = perf_counter()

        batch_durations.append(batch_complete - batch_start)

        batch += 1

        if batch % 25 == 0:
            print(f"Batch {batch} - Loss at {torch.tensor(losses).mean()} - took {sum(batch_durations)}s at an average of {sum(batch_durations)/len(batch_durations)}s per batch")
            batch_durations = []

    print(f"Epoch {i + 1} complete. Took {perf_counter() - epoch_start}s")
    print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))