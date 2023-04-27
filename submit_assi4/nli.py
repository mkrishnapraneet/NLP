# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
import torchtext

# get dataloader
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe
from nltk.stem import PorterStemmer

# %%
nli_dataset = load_dataset("multi_nli")

# %%
print(nli_dataset.shape)
train_dataset = nli_dataset["train"]
# validation_dataset = nli_dataset["validation"]
test_dataset = nli_dataset["validation_matched"]
print(train_dataset.shape)
# print(validation_dataset.shape)
print(test_dataset.shape)

# filter out samples with label -1 
train_dataset = train_dataset.filter(lambda example: example["label"] != -1)
# validation_dataset = validation_dataset.filter(lambda example: example["label"] != -1)
test_dataset = test_dataset.filter(lambda example: example["label"] != -1)

# remove all columns except premise, hypothesis and label
train_dataset = train_dataset.remove_columns(["promptID", "pairID", "premise_binary_parse", "premise_parse", "hypothesis_binary_parse", "hypothesis_parse", "genre"])
# validation_dataset = validation_dataset.remove_columns(["promptID", "pairID", "premise_binary_parse", "premise_parse", "hypothesis_binary_parse", "hypothesis_parse", "genre"])
test_dataset = test_dataset.remove_columns(["promptID", "pairID", "premise_binary_parse", "premise_parse", "hypothesis_binary_parse", "hypothesis_parse", "genre"])

print(train_dataset.shape)
# print(validation_dataset.shape)
print(test_dataset.shape)

# %%
# convert premise and hypothesis to lower
train_dataset = train_dataset.map(lambda example: {"premise": example["premise"].lower(), "hypothesis": example["hypothesis"].lower()})
# validation_dataset = validation_dataset.map(lambda example: {"premise": example["premise"].lower(), "hypothesis": example["hypothesis"].lower()})
test_dataset = test_dataset.map(lambda example: {"premise": example["premise"].lower(), "hypothesis": example["hypothesis"].lower()})
print(train_dataset.shape)

# %%
# tokenize premise and hypothesis and concatenate them using [SEP] token
train_dataset = train_dataset.map(lambda example: {"premise": word_tokenize(example["premise"]), "hypothesis": word_tokenize(example["hypothesis"])})
# validation_dataset = validation_dataset.map(lambda example: {"premise": word_tokenize(example["premise"]), "hypothesis": word_tokenize(example["hypothesis"])})
test_dataset = test_dataset.map(lambda example: {"premise": word_tokenize(example["premise"]), "hypothesis": word_tokenize(example["hypothesis"])})
print(train_dataset.shape)

# # perform stemming
# stemmer = PorterStemmer()
# train_dataset = train_dataset.map(lambda example: {"premise": [stemmer.stem(token) for token in example["premise"]], "hypothesis": [stemmer.stem(token) for token in example["hypothesis"]]})
# # validation_dataset = validation_dataset.map(lambda example: {"premise": [stemmer.stem(token) for token in example["premise"]], "hypothesis": [stemmer.stem(token) for token in example["hypothesis"]]})
# test_dataset = test_dataset.map(lambda example: {"premise": [stemmer.stem(token) for token in example["premise"]], "hypothesis": [stemmer.stem(token) for token in example["hypothesis"]]})
# print(train_dataset.shape)

# concatenate premise and hypothesis using [SEP] token
train_dataset = train_dataset.map(lambda example: {"premise": example["premise"] + ["[SEP]"] + example["hypothesis"]})
# validation_dataset = validation_dataset.map(lambda example: {"premise": example["premise"] + ["[SEP]"] + example["hypothesis"]})
test_dataset = test_dataset.map(lambda example: {"premise": example["premise"] + ["[SEP]"] + example["hypothesis"]})
print(train_dataset.shape)


# %%
# remove the hypothesis column
train_dataset = train_dataset.remove_columns(["hypothesis"])
# validation_dataset = validation_dataset.remove_columns(["hypothesis"])
test_dataset = test_dataset.remove_columns(["hypothesis"])
print(train_dataset.shape)
print(train_dataset[0])

# %%
# find the maximum length of the premise + hypothesis
max_length = max([len(example["premise"]) for example in train_dataset])
print(f'Maximum length of premise + hypothesis: {max_length}')
avg_length = np.mean([len(example["premise"]) for example in train_dataset])
print(f'Average length of premise + hypothesis: {avg_length}')

# %%
# percentage of samples with length less than 100
print(f'Percentage of samples with length less than 100: {np.mean([len(example["premise"]) < 100 for example in train_dataset])}')

# %%
# truncate the premise + hypothesis to 128 tokens and add padding to the end
train_dataset = train_dataset.map(lambda example: {"premise": example["premise"][:128] + ["[PAD]"] * (128 - len(example["premise"][:128]))})
# validation_dataset = validation_dataset.map(lambda example: {"premise": example["premise"][:128] + ["[PAD]"] * (128 - len(example["premise"][:128]))})
test_dataset = test_dataset.map(lambda example: {"premise": example["premise"][:128] + ["[PAD]"] * (128 - len(example["premise"][:128]))})

# %%
# build vocabulary
vocabulary = torchtext.vocab.build_vocab_from_iterator([example["premise"] for example in train_dataset], specials=["[UNK]", "[PAD]", "[SEP]"], min_freq=5)
vocabulary.set_default_index(vocabulary["[UNK]"])

print(f'Vocabulary size: {len(vocabulary)}')
print(vocabulary.get_itos()[:10])

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
# convert the tokens to indices
train_indices = train_dataset.map(lambda example: {"premise": torch.tensor([vocabulary[token] for token in example["premise"]], dtype=torch.long)})
# validation_dataset = validation_dataset.map(lambda example: {"premise": torch.tensor([vocabulary[token] for token in example["premise"]], dtype=torch.long)})
test_indices = test_dataset.map(lambda example: {"premise": torch.tensor([vocabulary[token] for token in example["premise"]], dtype=torch.long)})
print(train_indices.shape)
print(train_indices[0])

train_indices_only = np.array(train_indices["premise"])
print(train_indices_only.shape)
# validation_indices_only = np.array(validation_dataset["premise"])
# print(validation_indices_only.shape)
test_indices_only = np.array(test_indices["premise"])
print(test_indices_only.shape)

print(train_indices_only[0])

train_labels_only = np.array(train_dataset["label"])
print(train_labels_only.shape)
# validation_labels_only = np.array(validation_dataset["label"])
# print(validation_labels_only.shape)
test_labels_only = np.array(test_dataset["label"])
print(test_labels_only.shape)


# %%
print(train_indices_only.shape)
# print(validation_indices.shape)
print(test_indices_only.shape)

print(train_indices_only[0])
print(test_indices_only[0])

print(train_labels_only.shape)
# print(validation_labels.shape)
print(test_labels_only.shape)

print(train_labels_only[0])
print(test_labels_only[0])

# %%
# batch the dataset
batch_size = 32
train_dataloader = DataLoader(list(zip(train_indices_only, train_labels_only)), batch_size=batch_size, shuffle=True)
# validation_dataloader = DataLoader(list(zip(validation_indices_only, validation_labels_only)), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(list(zip(test_indices_only, test_labels_only)), batch_size=batch_size, shuffle=True)


# %%
# define the model as a sentence classifier using ELMO embeddings using two biLSTM layers and use glove embeddings for the words
class SentenceClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_classes=3, num_layers=2, dropout=0.5):
        super(SentenceClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm1f = nn.LSTM(self.embedding_dim, self.hidden_dim, 1, bidirectional=False, batch_first=True, dropout=self.dropout)
        self.lstm1b = nn.LSTM(self.embedding_dim, self.hidden_dim, 1, bidirectional=False, batch_first=True, dropout=self.dropout)
        self.lstm2f = nn.LSTM(self.hidden_dim, self.hidden_dim, 1, bidirectional=False, batch_first=True, dropout=self.dropout)
        self.lstm2b = nn.LSTM(self.hidden_dim, self.hidden_dim, 1, bidirectional=False, batch_first=True, dropout=self.dropout)
        # self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, bidirectional=True, dropout=self.dropout)
        self.linear = nn.Linear(self.hidden_dim * 4 + self.embedding_dim, self.num_classes)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        # x.shape = (batch_size, max_length)
        embedded = self.embedding(x)
        # embedded.shape = (batch_size, max_length, embedding_dim)
        # print(embedded.shape)
        output1f, (hidden1f, cell1f) = self.lstm1f(embedded)
        output1b, (hidden1b, cell1b) = self.lstm1b(embedded.flip(dims = [1]))
        # output1b, (hidden1b, cell1b) = self.lstm1b(embedded)
        output2f, (hidden2f, cell2f) = self.lstm2f(output1f)
        output2b, (hidden2b, cell2b) = self.lstm2b(output1b)
        # output.shape = (batch_size, max_length, hidden_dim)
        
        # concatenate the forward and backward outputs for both layers
        output1 = torch.cat((output1f, output1b), dim=2)
        output2 = torch.cat((output2f, output2b), dim=2)
        # output.shape = (batch_size, max_length, hidden_dim * 2)

        # pass these outputs through a linear layer along with the initial embedding
        output = torch.cat((output1, output2, embedded), dim=2)
        # output.shape = (batch_size, max_length, hidden_dim * 4 + embedding_dim)
        output = self.linear(output)
        # output.shape = (batch_size, max_length, num_classes)
        output = output[:, -1, :]
        # output.shape = (batch_size, num_classes)
        output = self.dropout(output)
        return output    

# %%
# define the hyperparameters
embedding_dim = 300
hidden_dim = 256
vocab_size = len(vocabulary)
num_classes = 3
num_layers = 2
dropout = 0.5
learning_rate = 0.001
num_epochs = 10

# initialize the model
model = SentenceClassifier(embedding_dim, hidden_dim, vocab_size, num_classes, num_layers, dropout)
model.to(device)

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%
# train the model
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for batch in train_dataloader:
        # get the inputs and labels
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # calculate the statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # print the statistics
    print("Epoch: %d, Train Loss: %.3f, Train Accuracy: %.3f" % (epoch + 1, train_loss / len(train_dataloader), train_correct / train_total))

# %%
# evaluate the model
model.eval()
test_loss = 0
test_correct = 0
test_total = 0
for batch in test_dataloader:
    # get the inputs and labels
    inputs, labels = batch
    inputs = inputs.to(device)
    labels = labels.to(device)

    # forward
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # calculate the statistics
    test_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    test_total += labels.size(0)
    test_correct += (predicted == labels).sum().item()

# print the statistics
print("Test Loss: %.3f, Test Accuracy: %.3f" % (test_loss / len(test_dataloader), test_correct / test_total))

# save the model
torch.save(model.state_dict(), "nli_model.pt")


# %%
# get classificaiton report, confusion matrix, and ROC curve using sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

model.eval()
y_true = []
y_pred = []
for batch in test_dataloader:
    # get the inputs and labels
    inputs, labels = batch
    inputs = inputs.to(device)
    labels = labels.to(device)

    # forward
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)

    # append the labels
    y_true.extend(labels.tolist())
    y_pred.extend(predicted.tolist())

# print the classification report
print(classification_report(y_true, y_pred, target_names=["entailment", "neutral", "contradiction"]))

# plot the confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks([0, 1, 2], ["entailment", "neutral", "contradiction"])
plt.yticks([0, 1, 2], ["entailment", "neutral", "contradiction"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# plot the ROC curve
y_true = np.array(y_true)
y_pred = np.array(y_pred)
fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=2)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()



# %%
# load the model
model = SentenceClassifier(embedding_dim, hidden_dim, vocab_size, num_classes, num_layers, dropout)
model.load_state_dict(torch.load("nli_model.pt"))
model.to(device)

# %%



