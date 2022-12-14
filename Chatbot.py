#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import string
import warnings
warnings.filterwarnings('ignore')


# In[2]:


with open('chatbot.txt', 'r') as file:
    raw_chat = file.read().lower()


# In[3]:


sentence_tokens = nltk.sent_tokenize(raw_chat)


# In[4]:


word_tokens = nltk.word_tokenize(raw_chat)


# In[5]:


lemmatizer = nltk.stem.WordNetLemmatizer()


# In[6]:


def lemTokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

remove_punctuation = dict((ord(p), None) for p in string.punctuation)

def lemNormalize(text):
    return lemTokens(nltk.word_tokenize(text.lower().translate(remove_punctuation)))


# In[7]:


greeting_input = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
greeting_output = ['hi', 'hey', 'hi there', 'hello', 'hi, how are you?']

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greeting_input:
            return random.choice(greeting_output)


# In[8]:


def response(request):
    bot_response = ''
    sentence_tokens.append(request)
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lemNormalize, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf==0):
        bot_response += "I don't understand, I'm sorry!"
    else:
        bot_response += sentence_tokens[idx]        
    return bot_response


# In[9]:


print('Chatbot Online\n')
while True:
    user = str(input('> ')).lower()
    if user == 'bye' or user == 'exit' or user == 'thanks' or user == 'thank you':
        print('Bot: You are welcome!')
        break
    else:
        if greeting(user) != None:
            print('Bot: ', greeting(user))
        else:
            print('Bot: ', end='')
            print(response(user))
            sentence_tokens.remove(user)

