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
import torch.nn.functional as F
import torch.optim as optim

# %%
input_file = '../../reviews_Movies_and_TV.json'
# input_file = 'try.json'

# Load the data
sentences = []
counter = 0
with open(input_file, 'r') as f:
    for line in f:
        if counter > 10000:
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# implement a model to learn the word embeddings from the data generated above

# define the model
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        # self.linear1 = nn.Linear(embedding_size, vocab_size)
        # self.linear2 = nn.Linear(vocab_size, embedding_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x is a list of indices of context words and the target word
        
        # get the embeddings of the context words
        context_embeddings = self.embeddings(x[:, :-1])
        # get the embedding of the target word
        target_embedding = self.embeddings(x[:, -1])

        # get the average of the context embeddings
        context_embeddings = torch.mean(context_embeddings, dim=1)

        # get the dot product of the context embeddings and the target embedding
        dot_product = torch.sum(context_embeddings*target_embedding, dim=1)
        # get the sigmoid of the dot product
        sigmoid = torch.sigmoid(dot_product)

        # return the sigmoid as the prediction
        return sigmoid
    
    def get_embedding(self, x):
        # get the embedding of the word with index x
        return self.embeddings(x)
    
    def get_embeddings(self):
        # get the embeddings of all the words
        return self.embeddings.weight
    
    def get_embedding_size(self):
        # get the size of the embeddings
        return self.embedding_size
    

# define the hyperparameters
vocab_size = len(vocab_indices)
embedding_size = 100
learning_rate = 0.001
batch_size = 64
num_epochs = 25

# create the model
model = Word2Vec(vocab_size, embedding_size).to(device)

# define the loss function
criterion = nn.BCELoss()

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# create the dataloader
dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y).float())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
       

# %%
# train the model
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(dataloader):
        # move the data to the device
        data = data.to(device)
        labels = labels.to(device)
        
        # forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10000 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))
    # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# save the model
# torch.save(model.state_dict(), 'word2vec.ckpt')

# load the model
# model.load_state_dict(torch.load('word2vec.ckpt'))




# %%
# get the embeddings of all the words
embeddings = model.get_embeddings().cpu().detach().numpy()

# print the shape of the embeddings
print(embeddings.shape)

def cosine_similarity(x, y):
    return np.dot(x, y)/(np.sqrt(np.sum(x**2))*np.sqrt(np.sum(y**2)))

# %%
# print the 10 most similar words to the word 'king'
def get_similar_words(word, embeddings, word2idx, idx2word, k=10):
    # get the index of the word
    word_idx = word2idx[word]
    # get the embedding of the word
    word_embedding = embeddings[word_idx]
    # get the cosine similarity between the word embedding and all the other embeddings
    similarities = []
    for i in range(embeddings.shape[0]):
        similarity = cosine_similarity(word_embedding, embeddings[i])
        similarities.append(similarity)
    # get the indices of the k most similar words
    most_similar_indices = np.argsort(similarities)[-k:]
    # get the words corresponding to the indices
    most_similar_words = [idx2word[idx] for idx in most_similar_indices]
    # print the most similar words in decreasing order of similarity
    for i in range(k):
        print('{}. {}'.format(i+1, most_similar_words[-(i+1)]))
    


# %%
# print the 10 most similar words to the word 'king'
get_similar_words('titanic', embeddings, word2idx, idx2word)


# %%
# use tnse to visualize the embeddings in 2D
tsne = TSNE(n_components=2, random_state=0)
word_vectors_2d = tsne.fit_transform(embeddings)

# %%
# # display the 10 closest words to each word in the words_to_visualise list in a 2D plot using word_vectors_2d using cosine similarity
# def plot_words(words_to_visualise, word_vectors_2d, word2idx, idx2word):
#     # get the indices of the words to visualize
#     indices = [word2idx[word] for word in words_to_visualise]
#     # get the embeddings of the words to visualize
#     embs = [embeddings[idx] for idx in indices]
#     # get the 10 most similar words to each word
#     similar_words = []
#     for i in range(len(words_to_visualise)):
#         similar_words.append(get_similar_words(words_to_visualise[i], embs, word2idx, idx2word, k=10))
#     # plot the words and their 10 most similar words
#     plt.figure(figsize=(10, 10))
#     for i in range(len(words_to_visualise)):
#         # plot the word
#         plt.scatter(word_vectors_2d[indices[i], 0], word_vectors_2d[indices[i], 1], marker='x', color='red')
#         plt.annotate(words_to_visualise[i], (word_vectors_2d[indices[i], 0], word_vectors_2d[indices[i], 1]))
#         # plot the 10 most similar words
#         for j in range(10):
#             idx = word2idx[similar_words[i][j]]
#             plt.scatter(word_vectors_2d[idx, 0], word_vectors_2d[idx, 1], marker='o', color='blue')
#             plt.annotate(similar_words[i][j], (word_vectors_2d[idx, 0], word_vectors_2d[idx, 1]))
#     plt.show()




# %%
# words_to_visualise = ['woman', 'interesting', 'enjoy', 'john', 'movie']

# plot_words(words_to_visualise, word_vectors_2d, word2idx, idx2word)

# %%



