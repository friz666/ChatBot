import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import Neural_Network

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = [] #patterns + tags
# petla przez kazde zdanie / rozbicie
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenizacja
        tokenized_pattern = tokenize(pattern)
        # dodanie do listy wszystkich slow
        all_words.extend(tokenized_pattern)
        # para xy do wyznaczenia tagu i paternu
        xy.append((tokenized_pattern, tag))

# kazde litera na mala i rodzielenie do samego tematu
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# usuniecie powtorzen i sortowanie slow
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# stworzenie danych treningowych
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: lista slow na kazde zdanie danego paternu
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: crossentropia z pytorch
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Parametry sieci neurnowej
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

#wywolanie konstruktora
dataset = ChatDataset()

#zaladowanie treningu
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Neural_Network(input_size, hidden_size, output_size).to(device)

# optymalizacja danych
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# trening stworzonego modelu
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # przepuszczenei danych treningowych
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # ponowna optymalizacja
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data_trained.pth"
torch.save(data, FILE)

print(f'Data has been trained succesfully! Your data file: {FILE}')

