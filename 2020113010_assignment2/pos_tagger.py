# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import gc

import torch
import torch.nn as nn
from torchtext.datasets import UDPOS
from torch.utils.data import DataLoader
import torchtext
from sklearn.metrics import classification_report

# %%
print('Starting...')
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Currently, it is running on {device}')

train_iter = UDPOS(split='train')
val_iter = UDPOS(split='valid')
test_iter = UDPOS(split='test')

# print(type(train_iter))

# for i in train_iter:
#     # print(len(i))
#     print(i[0])   # i[0] is the sentence
#     print(i[1])   # i[1] is the label
#     break

train_labels = []
train_tokens = []
val_tokens = []
val_labels = []
test_tokens = []
test_labels = []

# store the data in a list
def store_data(data_iter, data, labels):
    for i in data_iter:
        # store the tokens in lower case
        data.append([j.lower() for j in i[0]])
        # append '<eos>' to the end of each sentence
        data[-1].append('<eos>')
        # data.append(i[0])
        labels.append(i[1])
        labels[-1].append('<eos>')
    return data, labels

train_tokens, train_labels = store_data(train_iter, train_tokens, train_labels)
val_tokens, val_labels = store_data(val_iter, val_tokens, val_labels)
test_tokens, test_labels = store_data(test_iter, test_tokens, test_labels)

# print(len(train_tokens))
# print(len(val_tokens))
# print(len(test_tokens))

# print(train_tokens[0])
# print(train_labels[0])

# build the vocabulary for the tokens and labels
# from torchtext.vocab import Vocab
# flatten the list
# train_tokens_flat = [item.lower() for sublist in train_tokens for item in sublist]
# train_labels_flat = [item.lower() for sublist in train_labels for item in sublist]

train_vocab = torchtext.vocab.build_vocab_from_iterator(train_tokens, specials=['<unk>'], min_freq=2)
# vocabulary = torchtext.vocab.build_vocab_from_iterator(corpus_tokens, specials=['<unk>'], min_freq=2)
train_vocab.set_default_index(train_vocab['<unk>']) 
# print(len(train_vocab))
# print(train_vocab.get_itos()[:10])

# build the vocabulary for the labels
train_labels_vocab = torchtext.vocab.build_vocab_from_iterator(train_labels, specials=['<unk>'], min_freq=1)
train_labels_vocab.set_default_index(train_labels_vocab['<unk>'])
# print(len(train_labels_vocab))
# print(train_labels_vocab.get_itos()[:18])

# store the tags as a list using the labels_vocab
pos_tags_list = list(train_labels_vocab.get_itos()[:18])

# %%
# use custom function to batch the data 
def batchify(data, labels, data_vocab, label_vocab, batch_size):
    # add a '<eos>' token to the end of each sentence
    # data = [i + ['<eos>'] for i in data]
    # labels = [i + ['PUNCT'] for i in labels]
    # print(data[0])
    # print(labels[0])

    # flatten all the sentences into a single list
    data_flat = [item for sublist in data for item in sublist]
    labels_flat = [item for sublist in labels for item in sublist]
    # print(data_flat[0:50])
    # print(labels_flat[0:50])

    # convert the tokens and labels into indices using the vocab
    data_indices = [data_vocab[token] for token in data_flat]
    labels_indices = [label_vocab[label] for label in labels_flat]
    # print(data_indices[0:50])
    # print(labels_indices[0:50])

    # convert the list into a tensor
    data_tensor = torch.tensor(data_indices, dtype=torch.long)
    labels_tensor = torch.tensor(labels_indices, dtype=torch.long)
    # print(data_tensor.shape)
    # print(labels_tensor.shape)

    # reshape the tensor into (batch_size, num_batches)
    num_batches = data_tensor.shape[0] // batch_size
    data_tensor = data_tensor[:num_batches * batch_size]
    labels_tensor = labels_tensor[:num_batches * batch_size]
    data_tensor = data_tensor.view(batch_size, -1)
    labels_tensor = labels_tensor.view(batch_size, -1)
    # print(data_tensor.shape)
    # print(labels_tensor.shape)

    return data_tensor, labels_tensor


train_data_tensor, train_label_tensor = batchify(train_tokens, train_labels, train_vocab, train_labels_vocab, batch_size)
valid_data_tensor, valid_label_tensor = batchify(val_tokens, val_labels, train_vocab, train_labels_vocab, batch_size)
test_data_tensor, test_label_tensor = batchify(test_tokens, test_labels, train_vocab, train_labels_vocab, batch_size)


# %%
class bi_directional_LSTM_tagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_tags, dropout):
        super(bi_directional_LSTM_tagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # first, pass the input through the embedding layer (dropout is applied here)
        embedded = self.dropout(self.embedding(x))
        # print(embedded.shape)        
        
        # then, pass the embedded input through the LSTM layer
        output, (hidden, cell) = self.lstm(embedded)
        # print(output.shape, hidden.shape, cell.shape)

        # finally, pass the output through the fully connected layer
        pred = self.fc(self.dropout(output))
        # print(output.shape)
        
        return pred

# define the hyperparameters
vocab_size = len(train_vocab)
embedding_dim = 100
hidden_dim = 128
num_layers = 2
num_tags = len(train_labels_vocab)
dropout = 0.2
batch_size = batch_size
num_epochs = 50
learning_rate = 0.001

# initialize the model
model = bi_directional_LSTM_tagger(vocab_size, embedding_dim, hidden_dim, num_layers, num_tags, dropout).to(device)
# print(model)

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=train_labels_vocab['<unk>'])
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%
# function to evaluate the model
def evaluate(model, data_tensor, label_tensor):
    
    torch.cuda.reset_peak_memory_stats() 
    gc.collect()
    torch.cuda.empty_cache()
    # torch.cuda.clear_memory_allocated()

    # set the model to evaluation mode
    model.eval()
    # initialize the hidden and cell states
    hidden = torch.zeros(num_layers * 2, batch_size, hidden_dim).to(device)
    cell = torch.zeros(num_layers * 2, batch_size, hidden_dim).to(device)

    with torch.no_grad():
        # get the input and labels
        data = data_tensor.to(device)
        targets = label_tensor.to(device)

        # forward pass
        outputs = model(data)
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))
        print(f'Loss: {loss.item():.4f}')

        # get the predictions
        predictions = outputs.argmax(2)
        # print(predictions.shape)
        # print(predictions[0])

        # calculate the accuracy
        correct = (predictions == targets).float()
        # print(correct.shape)
        # print(correct[0])
        accuracy = correct.sum() / correct.numel()
        print(f'Accuracy: {accuracy:.4f}')

        # calculate the precision, recall and f1 score using classification_report
        predictions = predictions.reshape(-1).cpu().numpy()
        targets = targets.reshape(-1).cpu().numpy()
        report = classification_report(targets, predictions, target_names=pos_tags_list, zero_division=0)
        print(f'Precision, Recall and F1 Score:')
        print(report)       

        return accuracy

curr_accuracy = 0

# ask the user if they want to train the model or load the saved model
choice = input('Do you want to train a new model or load the saved model? (train/load)\n--> ')
if choice.lower() == 'load':
    loaded_model = bi_directional_LSTM_tagger(vocab_size, embedding_dim, hidden_dim, num_layers, num_tags, dropout).to(device)
    # check for saved model
    if os.path.exists('model.pt'):
        print('Loading the saved model...')
        loaded_model.load_state_dict(torch.load('model.pt'))
        # store the validation loss
        print('The saved model has been loaded. Its performance on the validation set is:')
        curr_accuracy = evaluate(loaded_model, valid_data_tensor, valid_label_tensor)
        # print the hyperparameters of the saved model
    else:
        print('No saved model found.')


elif choice.lower() == 'train':

    # train the model
    print('Training the model...')
    for epoch in range(num_epochs):
        # set the model to training mode
        model.train()
        # initialize the hidden and cell states
        hidden = torch.zeros(num_layers * 2, batch_size, hidden_dim).to(device)
        cell = torch.zeros(num_layers * 2, batch_size, hidden_dim).to(device)

        for i in range(0, train_data_tensor.size(1) - 1, 32):
            # get the input and labels
            data = train_data_tensor[:, i:i+32].to(device)
            targets = train_label_tensor[:, i:i+32].to(device)

            # forward pass
            outputs = model(data)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))

            # backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print the loss
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')


    print('\nEvaluating on the training set:')
    acc = evaluate(model, train_data_tensor, train_label_tensor)
    # test the model
    print('\nEvaluating on the validation set:')
    val_accuracy = evaluate(model, valid_data_tensor, valid_label_tensor)
    print('\nEvaluating on the test set:')
    acc = evaluate(model, test_data_tensor, test_label_tensor)

    if val_accuracy > curr_accuracy:
        curr_accuracy = val_accuracy
        print('Saving the model...')
        torch.save(model.state_dict(), 'model.pt')
        print('The model has been saved.\n')
        
else :
    print('Invalid choice.')
    exit()
# %%
import nltk
# input_sentence = 'The quick brown fox jumps over the lazy dog .'
# input_sentence = 'My name is John'
# input_sentence = 'I bank at Chase Bank.'

# take the input sentence from the user
input_sentence = input('Enter a sentence: \n--> ')

def predict(model, sentence, vocab, labels_vocab):
    
    # print(pos_tags)
    
    # set the model to evaluation mode
    model.eval()
    # initialize the hidden and cell states
    hidden = torch.zeros(num_layers * 2, 1, hidden_dim).to(device)
    cell = torch.zeros(num_layers * 2, 1, hidden_dim).to(device)

    # tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    tokens.append('<eos>')
    tokens = [token.lower() for token in tokens]
    # print(tokens)

    # convert the tokens to indices
    token_indices = [vocab[token] for token in tokens]
    # print(token_indices)

    # convert the token indices to a tensor
    token_tensor = torch.LongTensor(token_indices).unsqueeze(1).to(device)
    # print(token_tensor.shape)

    with torch.no_grad():
        # forward pass
        outputs = model(token_tensor)
        # print(outputs.shape)

        # get the predictions
        predictions = outputs.argmax(2)
        # print(predictions.shape)
        # print(predictions)

        # convert the predictions to a list
        predicted_indices = [p.item() for p in predictions]
        # print(predicted_indices)

        # convert the indices to tags
        tags = [pos_tags_list[index] for index in predicted_indices]
        # print(tags)

        # print the predictions
        for token, tag in zip(tokens, tags):
            print(f'{token}\t\t{tag}')

if (choice == 'train'):
    predict(model, input_sentence, train_vocab, train_labels_vocab)

elif (choice == 'load'):
    predict(loaded_model, input_sentence, train_vocab, train_labels_vocab)

# %%



