# %%
# tokenization
import numpy as np
import re
from nltk.tokenize import sent_tokenize
import random


def tokenize(text, n):

    # convert to lower case
    text = text.lower()

    sentences = sent_tokenize(text)
    final_tokens = []
    # one_word_hist_table = {}
    # one_hist_word_table = {}

    # two_word_hist_table = {}
    # two_hist_word_table = {}

    # three_word_hist_table = {}
    # three_hist_word_table = {}

    # four_word_hist_table = {}
    # four_hist_word_table = {}

    # his = tuple()
    # one_word_hist_table[his] = {}
    # one_word_hist_table[his]['<unk>'] = 1
    # one_hist_word_table['<unk>'] = {}
    # one_hist_word_table['<unk>'][his] = 1
    
    for sentence in sentences:
        # print(sentence)
        text = sentence
        # split into tokens by white space
        tokens = text.split()

        # if it is a url, replace it with <url>
        tokens = ['<url>' if re.match(
            r'^https?:\/\/.*[\r\n]*$', word) else word for word in tokens]
        # if it is a number, replace it with <number>
        tokens = ['<number>' if re.match(
            r'^\d+$', word) else word for word in tokens]
        # if it is a word with only digits, replace it with <number>
        # tokens = ['<number>' if re.match(r'^\d+\w+$', word) else word for word in tokens]
        # if it is a mention, replace it with <mention>
        tokens = ['<mention>' if re.match(
            r'^@\w+$', word) else word for word in tokens]
        # if it is a hashtag, replace it with <hashtag>
        tokens = ['<hashtag>' if re.match(
            r'^#\w+$', word) else word for word in tokens]

        # make separate tokens for punctuations and keep for special tokens like <url>, <number>, <mention>, <hashtag>
        # tokens = [re.split('(\W+)', word) for word in tokens]
        tokens = [re.split('(\W+)', word) if (word != '<url>' and word != '<number>' and word !=
                                            '<mention>' and word != '<hashtag>') else [word] for word in tokens]
        # tokens = [tok for word in tokens for tok in re.split('(\W+)', word) if (word != '<url>' and word != '<number>' and word != '<mention>' and word != '<hashtag>')]

        # flatten the elements
        tokens = [tok for word in tokens for tok in word]

        # remove empty tokens
        tokens = [tok for tok in tokens if len(tok) > 0]

        # unigram model
        # make a dict for each word
        # for i in range(1-1, len(tokens)):
        #     # store the previous n-1 words as history
        #     history = tuple(tokens[i-1+1:i])
        #     word = tokens[i]
        #     # print(history, word)
        #     # if the history is not in the dict, add it
        #     if history not in one_word_hist_table:
        #         one_word_hist_table[history] = {}
        #     # if the word is not in the dict, add it
        #     if word not in one_word_hist_table[history]:
        #         one_word_hist_table[history][word] = 0
        #     # increment the count
        #     one_word_hist_table[history][word] += 1

        # # make a dict for each history
        # for i in range(1-1, len(tokens)):
        #     # store the previous n-1 words as history
        #     history = tuple(tokens[i-1+1:i])
        #     word = tokens[i]
        #     # print(history, word)
        #     # if the history is not in the dict, add it
        #     if word not in one_hist_word_table:
        #         one_hist_word_table[word] = {}
        #     # if the word is not in the dict, add it
        #     if history not in one_hist_word_table[word]:
        #         one_hist_word_table[word][history] = 0
        #     # increment the count
        #     one_hist_word_table[word][history] += 1

        # for i in range(n-1):
        #     # add start tokens
        #     tokens.insert(0, '<start>')
        #     # add end tokens
        #     tokens.append('<end>')

        # add start tokens
        tokens.insert(0, '<start>')
        # add end tokens
        tokens.append('<end>')

        final_tokens.append(tokens)
        # print(tokens)

    # print(word_hist_table)
    # print(hist_word_table)              

    print(len(final_tokens))

    # print(final_tokens[:100])
    return final_tokens
    # return final_tokens, one_word_hist_table, one_hist_word_table, two_word_hist_table, two_hist_word_table, three_word_hist_table, three_hist_word_table, four_word_hist_table, four_hist_word_table


# issues : continuous punctuations are not separated

n = 4

# input_sentence = "One can not decipher a man's feelings"
# input_sentence = "This was invitation enough."
# input_sentence = "This was not invitation enough."
# input_sentence = "That is an evening gamer."
# input_sentence = "The boy bought a chocolate."
# input_sentence = "I sneezed loudly."
input_sentence = "He slid it into the left slot for them."

# f_tokens, word_hist_table, hist_word_table = tokenize(input_sentence, n)
# input_tokens, ig, fig, dig, hig, gig, hig, rig, wig = tokenize(input_sentence, n)
input_tokens = tokenize(input_sentence, n)
print(input_tokens[0])

# corpus
# path_to_corpus = 'test.txt'
path_to_corpus = 'Pride and Prejudice - Jane Austen.txt'
# path_to_corpus = 'Ulysses - James Joyce.txt'

with open(path_to_corpus, 'r') as f:
    text = f.read()
# text = "The boy ate a chocolate. The girl bought a chocolate. The girl ate a chocolate. The boy bought a horse."
# corpus_tokens, one_word_hist_table, one_hist_word_table, two_word_hist_table, two_hist_word_table, three_word_hist_table, three_hist_word_table, four_word_hist_table, four_hist_word_table = tokenize(text, n)
corpus_tokens = tokenize(text, n)

# select random 70% of the sentences as training data
train_tokens = random.sample(corpus_tokens, int(0.7*len(corpus_tokens)))
# remove these from corpus_tokens
corpus_tokens = [sentence for sentence in corpus_tokens if sentence not in train_tokens]
# select random 50% of the sentences as validation data
val_tokens = random.sample(corpus_tokens, int(0.5*len(corpus_tokens)))
# remove these from corpus_tokens
corpus_tokens = [sentence for sentence in corpus_tokens if sentence not in val_tokens]
# the remaining sentences are test data
test_tokens = corpus_tokens

corpus_tokens = train_tokens
print(len(corpus_tokens))




# %%
# print(corpus_tokens)
# write corpus tokens to file
with open('corpus_tokens.txt', 'w') as f:
    for tokens in corpus_tokens:
        # f.write(' '.join(tokens))
        # f.write('\n')
        f.write(str(tokens))
        f.write('\n')

# vocabulary = {}
# for tokens in corpus_tokens:
#     for token in tokens:
#         if token not in vocabulary:
#             vocabulary[token] = 0
#         vocabulary[token] += 1
# vocabulary['<unk>'] = 1
# vocabulary = set()
# for tokens in corpus_tokens:
#     for token in tokens:
#         vocabulary.add(token)
# vocabulary.add('<unk>')
# vocabulary = list(vocabulary)
# print(len(vocabulary))
# print(vocabulary)
        

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.utils.data import DataLoader
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.manual_seed(0)

# %%
# build a pytorch vocabulary from the vocabulary already built
vocabulary = torchtext.vocab.build_vocab_from_iterator(corpus_tokens, specials=['<unk>'], min_freq=2)
vocabulary.set_default_index(vocabulary['<unk>']) 
print(len(vocabulary))                         
print(vocabulary.get_itos()[:10])

# %%
# load data from corpus using torchtext data loader
def form_data_for_lstm(corpus_tokens, vocabulary, batch_size):
    # encode each token in the corpus using the vocabulary as a list of indices
    corpus_indices = []
    for sentence in corpus_tokens:
        indices = [vocabulary[token] for token in sentence]
        corpus_indices.append(indices)
    # print(corpus_indices[:10])
    # print(len(corpus_indices))
    # flatten the list of lists into a list
    corpus_indices = [index for sentence in corpus_indices for index in sentence]
    # print(len(corpus_indices))
    corpus_indices = torch.LongTensor(corpus_indices)
    num_batches = corpus_indices.shape[0] // batch_size 
    corpus_indices = corpus_indices[:num_batches * batch_size]                       
    corpus_indices = corpus_indices.view(batch_size, num_batches)          
    return corpus_indices

batch_size = 64
train_data = form_data_for_lstm(corpus_tokens, vocabulary, batch_size)
valid_data = form_data_for_lstm(val_tokens, vocabulary, batch_size)
test_data = form_data_for_lstm(test_tokens, vocabulary, batch_size)

# %%
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate):
                
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # if tie_weights:
        self.embedding.weight = self.fc.weight

    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)          
        output = self.dropout(output) 
        prediction = self.fc(output)
        return prediction, hidden
    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell
    
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell
        

# %%
vocab_size = len(vocabulary)
embedding_dim = 1024             
hidden_dim = 1024                
num_layers = 2                   
dropout_rate = 0.65              
# tie_weights = True                  
lr = 0.001

model = LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# %%
# break the data into batches
def make_batch(data, i, seq_len, num_batches):
    seq_len = min(seq_len, num_batches - 1 - i)
    x = data[:, i:i+seq_len]
    y = data[:, i+1:i+1+seq_len]
    return x, y

def train(model, data, optimizer, criterion, batch_size, seq_len, clip, device):
    
    epoch_loss = 0
    model.train()
    # drop all batches that are not a multiple of seq_len
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)
    
    for idx in range(0, num_batches - 1, seq_len):
        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)

        src, target = make_batch(data, seq_len, num_batches, idx)
        src, target = src.to(device), target.to(device)
        batch_size = src.shape[0]
        prediction, hidden = model(src, hidden)               

        prediction = prediction.reshape(batch_size * seq_len, -1)   
        target = target.reshape(-1)
        loss = criterion(prediction, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches

def evaluate(model, data, criterion, batch_size, seq_len, device):

    epoch_loss = 0
    model.eval()
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for idx in range(0, num_batches - 1, seq_len):
            hidden = model.detach_hidden(hidden)
            src, target = make_batch(data, seq_len, num_batches, idx)
            src, target = src.to(device), target.to(device)
            batch_size= src.shape[0]

            prediction, hidden = model(src, hidden)
            prediction = prediction.reshape(batch_size * seq_len, -1)
            target = target.reshape(-1)

            loss = criterion(prediction, target)
            epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches

# %%
n_epochs = 10
seq_len = 35
clip = 0.25
saved = False

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)

if saved:
    model.load_state_dict(torch.load('model.pt',  map_location=device))
    test_loss = evaluate(model, test_data, criterion, batch_size, seq_len, device)
    print(f'Test Perplexity: {math.exp(test_loss):.3f}')
else:
    best_valid_loss = float('inf')

    for epoch in range(n_epochs):
        train_loss = train(model, train_data, optimizer, criterion, 
                    batch_size, seq_len, clip, device)
        valid_loss = evaluate(model, valid_data, criterion, batch_size, 
                    seq_len, device)
        
        lr_scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model.pt')

        print(f'\tTrain Perplexity: {math.exp(train_loss):.3f}')
        print(f'\tValid Perplexity: {math.exp(valid_loss):.3f}')

# %%


# %%




    






# %%



