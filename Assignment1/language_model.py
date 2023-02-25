# %%
# import numpy as np
# import torch

# %%
# tokenization
import numpy as np
import re
from nltk.tokenize import sent_tokenize
import sys

print("Making LM from corpus...")

def tokenize(text, n):

    # convert to lower case
    text = text.lower()

    sentences = sent_tokenize(text)
    final_tokens = []
    one_word_hist_table = {}
    one_hist_word_table = {}

    two_word_hist_table = {}
    two_hist_word_table = {}

    three_word_hist_table = {}
    three_hist_word_table = {}

    four_word_hist_table = {}
    four_hist_word_table = {}

    his = tuple()
    one_word_hist_table[his] = {}
    one_word_hist_table[his]['<unk>'] = 1
    one_hist_word_table['<unk>'] = {}
    one_hist_word_table['<unk>'][his] = 1
    
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
        for i in range(1-1, len(tokens)):
            # store the previous n-1 words as history
            history = tuple(tokens[i-1+1:i])
            word = tokens[i]
            # print(history, word)
            # if the history is not in the dict, add it
            if history not in one_word_hist_table:
                one_word_hist_table[history] = {}
            # if the word is not in the dict, add it
            if word not in one_word_hist_table[history]:
                one_word_hist_table[history][word] = 0
            # increment the count
            one_word_hist_table[history][word] += 1

        # make a dict for each history
        for i in range(1-1, len(tokens)):
            # store the previous n-1 words as history
            history = tuple(tokens[i-1+1:i])
            word = tokens[i]
            # print(history, word)
            # if the history is not in the dict, add it
            if word not in one_hist_word_table:
                one_hist_word_table[word] = {}
            # if the word is not in the dict, add it
            if history not in one_hist_word_table[word]:
                one_hist_word_table[word][history] = 0
            # increment the count
            one_hist_word_table[word][history] += 1

        # for i in range(n-1):
        #     # add start tokens
        #     tokens.insert(0, '<start>')
        #     # add end tokens
        #     tokens.append('<end>')

        # add start tokens
        tokens.insert(0, '<start>')
        # add end tokens
        tokens.append('<end>')

        # bigram model
        # make a dict for each word
        for i in range(2-1, len(tokens)):
            # store the previous n-1 words as history
            history = tuple(tokens[i-2+1:i])
            word = tokens[i]
            # print(history, word)
            # if the history is not in the dict, add it
            if history not in two_word_hist_table:
                two_word_hist_table[history] = {}
            # if the word is not in the dict, add it
            if word not in two_word_hist_table[history]:
                two_word_hist_table[history][word] = 0
            # increment the count
            two_word_hist_table[history][word] += 1

        # make a dict for each history
        for i in range(2-1, len(tokens)):
            # store the previous n-1 words as history
            history = tuple(tokens[i-2+1:i])
            word = tokens[i]
            # print(history, word)
            # if the history is not in the dict, add it
            if word not in two_hist_word_table:
                two_hist_word_table[word] = {}
            # if the word is not in the dict, add it
            if history not in two_hist_word_table[word]:
                two_hist_word_table[word][history] = 0
            # increment the count
            two_hist_word_table[word][history] += 1
        
        # add start tokens
        tokens.insert(0, '<start>')
        # add end tokens
        tokens.append('<end>')

        # trigram model
        # make a dict for each word
        for i in range(3-1, len(tokens)):
            # store the previous n-1 words as history
            history = tuple(tokens[i-3+1:i])
            word = tokens[i]
            # print(history, word)
            # if the history is not in the dict, add it
            if history not in three_word_hist_table:
                three_word_hist_table[history] = {}
            # if the word is not in the dict, add it
            if word not in three_word_hist_table[history]:
                three_word_hist_table[history][word] = 0
            # increment the count
            three_word_hist_table[history][word] += 1

        # make a dict for each history
        for i in range(3-1, len(tokens)):
            # store the previous n-1 words as history
            history = tuple(tokens[i-3+1:i])
            word = tokens[i]
            # print(history, word)
            # if the history is not in the dict, add it
            if word not in three_hist_word_table:
                three_hist_word_table[word] = {}
            # if the word is not in the dict, add it
            if history not in three_hist_word_table[word]:
                three_hist_word_table[word][history] = 0
            # increment the count
            three_hist_word_table[word][history] += 1
        
        # add start tokens
        tokens.insert(0, '<start>')
        # add end tokens
        tokens.append('<end>')

        # 4gram model
        # make a dict for each word
        for i in range(4-1, len(tokens)):
            # store the previous n-1 words as history
            history = tuple(tokens[i-4+1:i])
            word = tokens[i]
            # print(history, word)
            # if the history is not in the dict, add it
            if history not in four_word_hist_table:
                four_word_hist_table[history] = {}
            # if the word is not in the dict, add it
            if word not in four_word_hist_table[history]:
                four_word_hist_table[history][word] = 0
            # increment the count
            four_word_hist_table[history][word] += 1

        # make a dict for each history
        for i in range(4-1, len(tokens)):
            # store the previous n-1 words as history
            history = tuple(tokens[i-4+1:i])
            word = tokens[i]
            # print(history, word)
            # if the history is not in the dict, add it
            if word not in four_hist_word_table:
                four_hist_word_table[word] = {}
            # if the word is not in the dict, add it
            if history not in four_hist_word_table[word]:
                four_hist_word_table[word][history] = 0
            # increment the count
            four_hist_word_table[word][history] += 1

        final_tokens.append(tokens)
        # print(tokens)

    # print(word_hist_table)
    # print(hist_word_table)
            
                

    # print(len(final_tokens))

    # print(final_tokens[:100])

    return final_tokens, one_word_hist_table, one_hist_word_table, two_word_hist_table, two_hist_word_table, three_word_hist_table, three_hist_word_table, four_word_hist_table, four_hist_word_table


# issues : continuous punctuations are not separated
# path_to_corpus = sys.argv[2]
# path_to_corpus = 'test.txt'
# path_to_corpus = 'Pride and Prejudice - Jane Austen.txt'

n = 4
# with open(path_to_corpus, 'r') as f:
#     text = f.read()

# t,t,t,t,t,t,t,t,t = tokenize(text, n)


# %%
# witten bell smoothing

import numpy as np

def n_gram_probabilities(n, word_hist_table, hist_word_table, input_tokens, i):
    # prob = count(history, word) / count(history)
    hist_to_compare = tuple(input_tokens[i-n+1:i])
    word_to_compare = input_tokens[i]
    # print(hist_to_compare, word_to_compare)

    # if the history is not in the dict, return 0
    if hist_to_compare not in word_hist_table:
        return 0
    # if the word is not in the dict, return 0
    if word_to_compare not in word_hist_table[hist_to_compare]:
        if(n == 1):
            word_to_compare = '<unk>'
        else:
            return 0
    
    # count(history, word) / count(history)
    prob = word_hist_table[hist_to_compare][word_to_compare] / sum(word_hist_table[hist_to_compare].values())
    # print(prob)
    return prob    

def send_correct_model(n):
    if n == 1:
        return one_word_hist_table, one_hist_word_table
    elif n == 2:
        return two_word_hist_table, two_hist_word_table
    elif n == 3:
        return three_word_hist_table, three_hist_word_table
    elif n == 4:
        return four_word_hist_table, four_hist_word_table
    

    
n = 4



# corpus
# path_to_corpus = 'test.txt'
# path_to_corpus = 'Pride and Prejudice - Jane Austen.txt'
# path_to_corpus = 'Ulysses - James Joyce.txt'
path_to_corpus = sys.argv[2]
with open(path_to_corpus, 'r') as f:
    text = f.read()
# text = "The boy ate a chocolate. The girl bought a chocolate. The girl ate a chocolate. The boy bought a horse."
corpus_tokens, one_word_hist_table, one_hist_word_table, two_word_hist_table, two_hist_word_table, three_word_hist_table, three_hist_word_table, four_word_hist_table, four_hist_word_table = tokenize(text, n)

# n = 2
# total_prob = 1
# for i in range(n-1, len(input_tokens[0])):
#     cur = (n_gram_probabilities(n, *send_correct_model(n), input_tokens[0], i))
#     print(cur)
#     total_prob *= cur
# print("total prob = " + str(total_prob))


# %%
def calc_lambda(n, word_hist_table, hist_word_table, input_tokens, i):
    # calculate the distinct nth words for the history
    hist_to_compare = tuple(input_tokens[i-n+1:i])
    word_to_compare = input_tokens[i]
    # print(hist_to_compare)

    # if the history is not in the dict, return 0
    if hist_to_compare not in word_hist_table:
        return 0
    # if the word is not in the dict, return 0
    # if word_to_compare not in word_hist_table[hist_to_compare]:
    #     return 0
    distinct_nth_words = len(word_hist_table[hist_to_compare])
    # print(distinct_nth_words)
    count_hist = sum(word_hist_table[hist_to_compare].values())
    # print(count_hist)

    _lambda = 1 - (distinct_nth_words / (distinct_nth_words + count_hist))
    # print(_lambda)
    return _lambda
    
   

def witten_bell(n, word_hist_table, hist_word_table, input_tokens, i):
    if n == 1:
        return n_gram_probabilities(n, *send_correct_model(n), input_tokens, i)
    _lambda = calc_lambda(n, *send_correct_model(n), input_tokens, i)
    return _lambda * n_gram_probabilities(n, *send_correct_model(n), input_tokens, i) + (1 - _lambda) * witten_bell(n-1, *send_correct_model(n-1), input_tokens, i)
    




# %%
def calc_kn_lambda(n, word_hist_table, hist_word_table, input_tokens, i, d):
    # calculate the distinct nth words for the history
    hist_to_compare = tuple(input_tokens[i-n+1:i])
    word_to_compare = input_tokens[i]
    # print(hist_to_compare)

    # if the history is not in the dict, return 0
    if hist_to_compare not in word_hist_table:
        return 1
    # if the word is not in the dict, return 0
    if word_to_compare not in word_hist_table[hist_to_compare]:
        if (n == 1):
            word_to_compare = '<unk>'
        else :
            return 1
    distinct_nth_words = len(word_hist_table[hist_to_compare])
    # print(distinct_nth_words)
    count_hist = sum(word_hist_table[hist_to_compare].values())
    # print(count_hist)
    return d * distinct_nth_words / count_hist

    

def kneser_ney(n, word_hist_table, hist_word_table, input_tokens, i, d=0.250):
    if n == 1:
        # check this later############################################################################
        # return n_gram_probabilities(n, *send_correct_model(n), input_tokens, i) 
        # numerator is the number of distinct bigram histories the word appears in as the last word (use the hist_word_table)
        numerator = 0
        if input_tokens[i] in hist_word_table:
            numerator = len(hist_word_table[input_tokens[i]])
        else:
            numerator = len(hist_word_table['<unk>'])
        # denominator is the number of distinct bigrams
        denominator = len(two_word_hist_table)
        
        return numerator / denominator


    _lambda = calc_kn_lambda(n, *send_correct_model(n), input_tokens, i, d)
    # word_hist_table, hist_word_table = send_correct_model(n)
    first_term = 0
    if tuple(input_tokens[i-n+1:i]) in word_hist_table:
        if input_tokens[i] in word_hist_table[tuple(input_tokens[i-n+1:i])]:
            # first term is max of 0 and the count of the ngram minus d divided by the count of the history
            first_term = max(0, (word_hist_table[tuple(input_tokens[i-n+1:i])][input_tokens[i]] - d) / sum(word_hist_table[tuple(input_tokens[i-n+1:i])].values()))
    return first_term + _lambda * kneser_ney(n-1, *send_correct_model(n-1), input_tokens, i, d)

type = str(sys.argv[1])

# input_sentence = "One can not decipher a man's feelings"
# input_sentence = "This was invitation enough."
# input_sentence = "This was not invitation enough."
# input_sentence = "That is an evening gamer."
# input_sentence = "The boy bought a chocolate."
# input_sentence = "I sneezed loudly."
# input_sentence = "He slid it into the left slot for them."
input_sentence = input("LM created. Enter a sentence: ")

# f_tokens, word_hist_table, hist_word_table = tokenize(input_sentence, n)
input_tokens, ig, fig, dig, hig, gig, hig, rig, wig = tokenize(input_sentence, n)
# print(input_tokens[0])

# %%
n = 4
if(type == 'w'):
    total_prob = 1
    for i in range(n-1, len(input_tokens[0])):
        cur = (witten_bell(n,*send_correct_model(n), input_tokens[0], i))
        # print(cur)
        total_prob *= cur
    print("total prob of sentence (witten bell) = " + str(total_prob))
    if total_prob == 0:
        print("perplexity of sentence (witten bell) = " + str("inf"))
    else:
        print("perplexity of sentence (witten bell) = " + str((1/total_prob)**(1/len(input_tokens[0]))))


# %%

elif(type == 'k'):
    total_prob = 1
    for i in range(n-1, len(input_tokens[0])):
        cur = (kneser_ney(n,*send_correct_model(n), input_tokens[0], i))
        # print(cur)
        total_prob *= cur
    print("total prob of sentence (kneser ney) = " + str(total_prob))
    if total_prob == 0:
        print("perplexity of sentence (kneser ney) = inf")
    else:
        print("perplexity of sentence (kneser ney) = " + str((1/total_prob)**(1/len(input_tokens[0]))))