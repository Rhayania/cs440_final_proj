# Author: Kira Vasquez-Kapit
# Colorado State Univeristy
# This file uses the code from Lab 5 in CS 440 as a starting point

# Imports
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Change each base in both sequences to an array, then add them so they can be input to a neural network
def encode_and_combine_sequences(seq1, seq2):
    encoding = {'a': [1, 0, 0, 0], 't': [0, 1, 0, 0], 'c': [0, 0, 1, 0], 'g': [0, 0, 0, 1]}
    first = np.array([encoding[base] for base in seq1])
    second = np.array([encoding[base] for base in seq2])

    # Handle sequences with different lengths
    if len(first) > len(second):
        padding = np.zeros((len(first) - len(second), 4))
        second = np.vstack([second, padding])
    else:
        padding = np.zeros((len(second) - len(first), 4))
        first = np.vstack([first, padding])

    return first + second

# Encode variant class labels into numbers
def encode_label(label):
    ans = 0 # Default, "no_" for "no variation"
    if label == "snp":
        ans = 1
    elif label == "inv":
        ans = 2
    elif label == "ind":
        ans = 3
    return ans

# Ensures all sequqnces are length 180.
def homogenize_seq_lengths(inputs):
    return np.array([np.vstack([seq, np.zeros((180 - len(seq), 4))]) for seq in inputs])

# Go through all files in a given folder (assumed to be all .txt files).
# Pull out the first and second lines for the sequences, and get the label for the sequences from the file name.
def get_dataset(folderpath):

    inputs = []
    labels = []

    for filename in os.listdir(folderpath):
        filepath = os.path.join(folderpath, filename)

        # get variant label from filename
        label = filename[:3]

        if os.path.isfile(filepath):
            with open(filepath, 'r') as file:
                # Read the first two lines
                seq1 = file.readline().strip()  # Read the first line
                seq2 = file.readline().strip()  # Read the second line

        inputs.append(encode_and_combine_sequences(seq1, seq2))
        labels.append(encode_label(label))

    # Pad each sequence so that they are all the same length (180). I don't know how to get around this yet. Hopefully doesn't affect results.
    inputs = homogenize_seq_lengths(inputs)

    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Calculate model accuracy for a given dataset
# Function taken from CS440 lab 5
def compute_accuracy(model, data_set):

    model = model.eval()

    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(data_set):

        with torch.no_grad():
            output = model(features)

        predictions = torch.argmax(output, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples

# Used guide to construct this Dataset class
class SequenceDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# PyTorch neural network class for this specific data
# Code modified from CS440 lab 5
class GenomeVariantRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):

        super().__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)
        # *2 for bidirectional model
        self.layers = torch.nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.layers(out)
        return out

# Prepare the input sequences for the model
training_inputs, training_labels = get_dataset("../data/basic_training")
validation_inputs, validation_labels = get_dataset("../data/basic_validation")
testing_inputs, testing_labels = get_dataset("../data/basic_testing")

# Create datasets and DataLoaders
training_dataset = SequenceDataset(training_inputs, training_labels)
val_dataset = SequenceDataset(validation_inputs, validation_labels)
testing_dataset = SequenceDataset(testing_inputs, testing_labels)

training_loader = DataLoader(
    dataset = training_dataset,
    batch_size = 4,
    shuffle = True,
)

validation_loader = DataLoader(
    dataset = val_dataset,
    batch_size = 4,
    shuffle = False,
)

test_loader = DataLoader(
    dataset = testing_dataset,
    batch_size = 4,
    shuffle = False,
)

# Initialize and train the RNN
model = GenomeVariantRNN(input_size = 4, hidden_size = 16, num_classes = 4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_list = []
train_acc_list = []
val_acc_list = []

# Training loop
# Code modified from CS440 lab 5
for epoch in range(100):

    model = model.train()
    for batch_idx, (features, labels) in enumerate(training_loader):

        outputs = model(features)

        #print(outputs.shape) # Cause of issues with loss function
        #print(labels.shape) # Cause of issues with loss function

        #loss = F.cross_entropy(outputs, labels) # Details in report
        optimizer.zero_grad()
        #loss.backward()
        optimizer.step()

        # Keep track of loss each iteration
        #loss_list.append(loss.item())

    # Logging per epoch
    train_acc = compute_accuracy(model, training_loader)
    val_acc = compute_accuracy(model, validation_loader)
    print(f"Train Acc {train_acc*100:.2f}% | Val Acc {val_acc*100:.2f}%")
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

# Calculate and output final results
# Code section taken from CS440 lab 5
train_acc = compute_accuracy(model, training_loader) # Performance on training dataset
val_acc = compute_accuracy(model, validation_loader) # Performance on validation dataset
test_acc = compute_accuracy(model, test_loader) # Performance on test dataset

print(f"Train Acc {train_acc*100:.2f}%")
print(f"Val Acc {val_acc*100:.2f}%")
print(f"Test Acc {test_acc*100:.2f}%")
