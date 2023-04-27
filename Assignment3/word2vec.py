# %%
import numpy as np
import matplotlib.pyplot as plt
import json
from nltk.tokenize import sent_tokenize, word_tokenize
import collections
import itertools
from sklearn.manifold import TSNE
import random
import torch
import torch.nn as nn
import pickle

# %%
input_file = '../../reviews_Movies_and_TV.json'
# input_file = 'try.json'

# Load the data
sentences = []
counter = 0
with open(input_file, 'r') as f:
    for line in f:
        if counter > 50000:
            break
        # add each sentence as a list of words to the sentences list, but each line of the json object is a document containing multiple sentences
        # sentences.append(word_tokenize(json.loads(line)['reviewText']))
        doc_sentences = sent_tokenize(json.loads(line)['reviewText'])
        # sentences.append([word_tokenize(sentence) for sentence in doc_sentences])
        for sentence in doc_sentences:
            sentences.append([word.lower() for word in word_tokenize(sentence)])
        counter += 1
        

print('Number of sentences: {}'.format(len(sentences)))
print(sentences[0])

# for sentence in sentences:
#     print(sentence)

# %%
# form the vocabulary
# Flatten the list of sentences into a single list of words
words = itertools.chain.from_iterable(sentences)

# Create a Counter object to count the frequency of each word
word_counter = collections.Counter(words)

# Extract the unique words from the Counter object to form the vocabulary
min_freq = 5
# vocabulary = set(word_counter.keys())
# vocabulary = set(word for word, count in word_counter.items() if count >= min_freq)
# add the word if it occurs more than min_freq times, else add <unk> token
vocabulary = set(word if count >= min_freq else '<unk>' for word, count in word_counter.items())

# add the <pad> token
vocabulary.add('<pad>')

# Print the size of the vocabulary
print('Vocabulary size: {}'.format(len(vocabulary)))

# Create a dictionary to map each word to an index
word2idx = {word: idx for idx, word in enumerate(vocabulary)}

# Create a dictionary to map each index to a word
idx2word = {idx: word for idx, word in enumerate(vocabulary)}

# print the 10 most common words
print('The 10 most common words are: ')
print(word_counter.most_common(10))

# %%
# prepare the data for training
window_size = 2
sliding_window_size = window_size*2 + 1
num_neg_samples_per_context = 3

vocab_indices = list(word2idx.values())
vocab_size = len(vocab_indices)

# create data with X being indices of the context words and the target word, and y being 0 or 1 based on whether the target word is correct for the context words
# also add negative samples
def create_data_with_negative_sampling(sentences, word2idx, window_size, num_neg_samples_per_context):
    X = []
    y = []
    # counter = 0
    for sentence in sentences:
        for i in range(len(sentence)):
            # a list of indices of context words and the target word
            # if it goes out of bounds, add <pad> tokens            
            context_words = sentence[max(0, i-window_size):i] + sentence[i+1:min(len(sentence), i+window_size+1)]
            target_word = sentence[i]
            # if the any of the words are not in the vocabulary, replace it with <unk>
            context_words = [word if word in word2idx else '<unk>' for word in context_words]
            target_word = target_word if target_word in word2idx else '<unk>'
            
            data_point = [word2idx[context_word] for context_word in context_words]
            # if the size of the data point is less than the sliding window size, add <pad> tokens
            # if len(data_point) < sliding_window_size:
            data_point += [word2idx['<pad>']]*(sliding_window_size-len(data_point)-1)
            data_point.append(word2idx[target_word])

            # add this to X and y
            X.append(data_point)
            y.append(1)

            # add negative samples
            for _ in range(num_neg_samples_per_context):
                # generate a random index between 0 and vocab_size
                negative_word = random.randint(0, vocab_size-1)
                X.append(data_point[:-1] + [negative_word])                
                y.append(0)
        # counter += 1
        # print(counter)
    return X, y 
            

    #         # convert the words to indices and add to X as [target_index, context_index1]
    #         for context_word in context_words:
    #             data_point = [word2idx[target_word], word2idx[context_word]]
    #             X.append(data_point)
    #             y.append(1)
    #             # add negative samples
    #             for _ in range(num_neg_samples_per_context):
    #                 # generate a random index between 0 and vocab_size
    #                 negative_word = random.randint(0, vocab_size-1)
    #                 X.append([word2idx[target_word], negative_word])                
    #                 y.append(0)
    # return X, y


   

    
X, y = create_data_with_negative_sampling(sentences, word2idx, window_size, num_neg_samples_per_context)



# %%

X = np.array(X)
y = np.array(y)

# shuffle the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# split the data into train and test

# save the data to a file so that it can be loaded later
# np.savez('data.npz', X=X, y=y)

# load the data from the file
def load_data(filename):
    data = np.load(filename)
    X = data['X']
    y = data['y']
    return X, y  

# %%
print('Number of data points: {}'.format(len(X)))
print('Number of labels: {}'.format(len(y)))

# print(vocab_indices)
print('index of <unk> is: {}'.format(word2idx['<unk>']))
print('index of <pad> is: {}'.format(word2idx['<pad>']))

for i in range (50):
    print('{}   {}'.format(X[i], y[i]))

# %%
# cbow with negative sampling
# hyperparameters
embedding_size = 100
epochs = 100
learning_rate = 0.001
batch_size = 64

# initialize the weights
# embedding matrix
# embeddings = np.random.uniform(-1, 1, (len(vocabulary), embedding_size))

# use the same embedding matrix for both context and target

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# softmax function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# create batches
def create_batches(X, y, batch_size):
    batches = []
    num_batches = len(X) // batch_size
    for i in range(num_batches):
        batch = (X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
        batches.append(batch)
    return batches       
    


# %%
# create the model
'''
first, we get the embeddings of the context words and the target word
we then average the embeddings of the context words to get the context embedding
we then take the cosine similarity between the context embedding and the target embedding
we then use the sigmoid function to get the probability of the target word being the correct word for the context words
we then calculate the loss by subtracting the probability from the actual label
we then backpropagate the loss to update the weights
'''
# we can use tensors to perform the operations

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initialize the weights
# embedding matrix
# embeddings = torch.randn(len(vocabulary), embedding_size, requires_grad=True)
embeddings = torch.randn(len(vocabulary), embedding_size, requires_grad=True, device=device)
# embeddings = torch.zeros(len(vocabulary), embedding_size, requires_grad=True, device=device)

# embeddings = embeddings.to(device)

# # write a manual loss function
# def loss_fn(y_pred, y):
#     return torch.sum(y - y_pred)

# write a function to train the model using gpu

def train(X, y, embeddings, learning_rate, epochs, batch_size):
    # convert X and y to torch tensors
    X = torch.LongTensor(X)
    X = X.to(device)
    y = torch.FloatTensor(y)
    y = y.to(device)
    # create batches
    batches = create_batches(X, y, batch_size)
    # create an optimizer
    optimizer = torch.optim.Adam([embeddings], lr=learning_rate)
    # create a loss function for regression
    loss_fn = torch.nn.BCELoss()
    prev_loss = 1000
    # train the model
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in batches:
            # get the embeddings of the context words and the target word
            # print(X_batch.shape)
            context_embeddings = embeddings[X_batch[:, :-1]]
            target_embeddings = embeddings[X_batch[:, -1]]
            # context_embeddings = embeddings[X_batch[:, 1]]
            # target_embeddings = embeddings[X_batch[:, 0]]
            # print(context_embeddings.shape)
            # print(target_embeddings.shape)

            # average the context embeddings
            context_embeddings = torch.mean(context_embeddings, dim=1)

            # calculate the dot product between the context embedding and the target embedding
            logits = torch.sum(context_embeddings * target_embeddings, dim=1)

            
            # # normalize the logits
            # logits = logits / (torch.norm(context_embeddings, dim=1) * torch.norm(target_embeddings, dim=1))
            # use the sigmoid function to get the probability of the target word being the correct word for the context words
            probs = torch.sigmoid(logits)
            
            # calculate the loss
            loss = loss_fn(probs, y_batch)
            epoch_loss += loss.item()
            # backpropagate the loss to update the weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch: {}, Loss: {}'.format(epoch, epoch_loss/len(batches)))
        if epoch_loss/len(batches) > prev_loss:
            break
        prev_loss = epoch_loss/len(batches)


# train the model
train(X, y, embeddings, learning_rate, epochs, batch_size)

# %%
# print 10 most similar words to a given word
def most_similar(word, embeddings, k):
    # get the embedding of the word
    word_embedding = embeddings[word2idx[word]]
    # calculate the cosine similarity between the word embedding and the embeddings of all the words
    similarities = torch.matmul(word_embedding, embeddings.T)
    # get the k most similar words
    top_k = torch.topk(similarities, k+1)[1].tolist()
    most_similar = []
    for idx in top_k:
        if idx != word2idx[word]:
            most_similar.append([idx2word[idx], similarities[idx].item()])
    return most_similar


sim_words = most_similar('movie', embeddings, 10)
for word, similarity in sim_words:
    print('{}\t\t{}'.format(word, similarity))

# %%
# save the embeddings
def save_embeddings(embeddings, filename):
    embeddings = embeddings.cpu().detach().numpy()
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)

# load the embeddings
def load_embeddings(filename):
    with open(filename, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

# %%
# save_embeddings(embeddings, 'embeddings.pkl')
# embeddings = load_embeddings('embeddings.pkl')

# %%



