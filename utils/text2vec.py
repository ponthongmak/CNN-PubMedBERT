#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 20:22:32 2021

@author: Wanchana
"""
import numpy as np

# # load GloVe embeddings
def load_GloVe(file_path):
    vocab,embeddings = [],[]
    with open(file_path,'rt') as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)
    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)
    #insert '<pad>' and '<unk>' tokens at start of vocab_npa.
    vocab_npa = np.insert(vocab_npa, 0, '<pad>')
    vocab_npa = np.insert(vocab_npa, 1, '<unk>')
    
    pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
    unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token. using mean of all token
    embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
    return vocab_npa, embs_npa
# =============================================================================
# 
# =============================================================================
def load_Fasttext(file_path):
    vocab,embeddings = [],[]
    with open(file_path,'rt') as fi:
        full_content = fi.read().strip().split('\n')
        del full_content[0]
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)
    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)
    #insert '<pad>' and '<unk>' tokens at start of vocab_npa.
    vocab_npa = np.insert(vocab_npa, 0, '<pad>')
    vocab_npa = np.insert(vocab_npa, 1, '<unk>')
    
    pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
    unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token. using mean of all token
    embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
    return vocab_npa, embs_npa
# =============================================================================
# 
# =============================================================================
# convert strings to GloVe familiar tokens
def text2ids(data, vocab_npa, col):
    no_matches, glove_ids = [], []
    # create lookup for token ids
    word_map = dict(zip(vocab_npa, range(len(vocab_npa))))
    for doc in data[col]:
        doc = doc.split()
        tokens = []
        for word in doc:
            if word in word_map: # if word is a GloVE word
                known_idx = word_map.get(word)
                tokens.append(known_idx)
            else:
                unk_idx = 1  # unknown word idx
                no_matches.append(word)
                tokens.append(unk_idx)
        # combine the tokens
        glove_ids.append(tokens)
    no_matches = list(set(no_matches))
    return no_matches, glove_ids
# =============================================================================
# 
# =============================================================================
# post pad GloVe
def pad_ids(tokenized_data, max_len):
    padded_tokens = []
    # for each tokenized document
    for tokenized_sent in tokenized_data:
        if len(tokenized_sent) == max_len:
            padded_tokens.append(tokenized_sent)
        if len(tokenized_sent) < max_len:
            # find the difference in length
            extension = max_len - len(tokenized_sent)
            # pad sentences to max_len
            tokenized_sent.extend(np.repeat(0, extension))
            # append new padded token
            padded_tokens.append(tokenized_sent)
    return np.array(padded_tokens, dtype=np.int64)
