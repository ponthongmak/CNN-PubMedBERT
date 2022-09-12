#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 12:31:58 2021

@author: wanchana
"""

import os
import random
import pandas as pd
import numpy as np
import re
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from keras.preprocessing.sequence import pad_sequences
import transformers
import logging
import pickle
# =============================================================================
# set seed
# =============================================================================
def set_seed(seed):
    """ Set all seeds to make results reproducible (deterministic mode).
        When seed is a false-y value or not supplied, disables deterministic mode. """

    if seed:
        logging.info(f"Running in deterministic mode with seed {seed}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    else:
        logging.info(f"Running in non-deterministic mode")
      
set_seed(2021)
# =============================================================================
# ## check GPU
# =============================================================================
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# =============================================================================
# specify path
# =============================================================================
path = ''
os.chdir(path)
# =============================================================================
# import BERT
# =============================================================================
# Pretrained BERT
model_class = transformers.BertModel
tokenizer_class = transformers.BertTokenizer
pretrained_weights='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
print('Loading BERT tokenizer...')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
# =============================================================================
# Tokenize all of the sentences and map the tokens to thier word IDs
# =============================================================================
def token_index(text, chunk_len, overlap):
    ids = []
    tokens = []
    origin_ids = {}
    chunk_len_used = chunk_len - 2
    print('Tokenizing text samples...')
    for i, sent in enumerate(text):
        token_sent = tokenizer.tokenize(sent)
        encoded_sent = tokenizer.encode(sent, add_special_tokens = False)

        origin_ids[i] = []
        
        # If the sentence is too long, chunk it.
        if len(encoded_sent) > chunk_len_used:
            n = np.ceil(len(encoded_sent) / chunk_len_used).astype(int)
            added_len = (n-1) * overlap
            n = np.ceil((len(encoded_sent) + added_len) / chunk_len_used).astype(int)
        else: 
            n = 1
            
        # Make chunks...
        for j in range(n):
            if j == 0:
                token_part = token_sent[:chunk_len_used]
                tokens.append(token_part)
                
                encoded_part = encoded_sent[:chunk_len_used]
                
                # add special token to all encoding part
                encoded_part = [tokenizer.cls_token_id] + encoded_part + [tokenizer.sep_token_id]
                
                ids.append(encoded_part)
                

                origin_ids[i].append(len(ids) - 1)
            else:
                token_part = token_sent[(j * chunk_len_used) - (j * overlap) : (j * chunk_len_used) - (j* overlap) + chunk_len_used]
                tokens.append(token_part)
                
                encoded_part = encoded_sent[(j * chunk_len_used) - (j * overlap) : (j * chunk_len_used) - (j* overlap) + chunk_len_used]
                                
                # add special token to all encoding part
                encoded_part = [tokenizer.cls_token_id] + encoded_part + [tokenizer.sep_token_id]
                
                ids.append(encoded_part)
    
                origin_ids[i].append(len(ids) - 1)
                
            if ((len(ids) % 2000) == 0):
                print('  Read {:,} text samples.'.format(len(ids)))
    print('')        
    print('DONE.')
    print('{:>10,} text samples before chunking'.format(len(text)))
    print('{:>10,} text samples after chunking'.format(len(ids)))
    
    return ids, tokens, origin_ids

# =============================================================================
# Create attention masks
# =============================================================================
def gen_att_mask(ids):
    attention_masks = []
    
    print('Creating attention masks...')
    # For each sentence...
    for sent in ids:
        # If a token ID is 0, the padding set to 0, else 1
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
        
    # conver to torch int64
    attention_masks = torch.tensor(attention_masks).to(torch.int64)
    print('')        
    print('DONE.')
    return attention_masks

# =============================================================================
# 
# =============================================================================
def extract_input(sheet):
    # raw dataset
    df_raw = pd.read_csv("Dataset_final.csv", sep = '|',dtype = {'an': str, 'hn': str, 'gender' : int, 'age':int})
    # subset used
    top_code = pd.read_excel("top_freq_ICD.xlsx", dtype = str, sheet_name = '{}'.format(sheet)) # import set of code for freq 25
    top_code = set(top_code.icd10)
    
    df_raw['dx']  = df_raw['dx'].apply(literal_eval) # text to list
    
    regex = [
            (r'(^)A|(^)B', '1.infection'),
            (r'(^)C|(^)D1|(^)D2|(^)D3|(^)D4', '2.neoplasms'),
            (r'(^)D5|(^)D6|(^)D7|(^)D8|(^)D9', '3.blood diseases'),
            (r'(^)E', '4.endocrine'),
            (r'(^)F', '5.mental disorder'),
            (r'(^)G', '6.nervous system'),
            (r'(^)H0|(^)H1|(^)H2|(^)H3|(^)H4|(^)H5', '7.eye and adnexa'),
            (r'(^)H6|(^)H7|(^)H8|(^)H9', '8.ear and mastoid'),
            (r'(^)I', '9.circulatory system'),
            (r'(^)J', '10.respiratory system'),
            (r'(^)K', '11.digestive system'),
            (r'(^)L', '12.skin'),
            (r'(^)M', '13.musculoskeletal system'),
            (r'(^)N', '14.genitourinary system'),
            (r'(^)O', '15.pregnancy'),
            (r'(^)P', '16.perinatal'),
            (r'(^)Q', '17.congenital mal'),
            (r'(^)R', '18.symptoms and sign'),
            (r'(^)S|(^)T', '19.injury'),
            (r'(^)V|(^)Y', '20.external causes'),
            (r'(^)Z', '21.health services'),
            (r'(^)U', '22.special purposes'),
             ]
    
    def f(x):
        return bool(re.search(reg, x)) # regex from global parameter
    for reg, disease in regex:
        df_raw[disease] = df_raw['pdx'].apply(lambda x: 1*bool(re.search(reg, x)))
     
    ## select top 5 Pdx
    df = df_raw[(df_raw['1.infection'] == 1) | (df_raw['2.neoplasms'] == 1) | 
                (df_raw['9.circulatory system'] == 1) | (df_raw['10.respiratory system'] == 1) |
                (df_raw['14.genitourinary system'] == 1)]
    
    ## select class 
    df = df[df.columns[0:22]]
    for reg, disease in regex:
        df[disease] = df['dx'].apply(lambda x: list(map(f, x)))
        df[disease] = df[disease].apply(lambda x: 1 if sum(x) >= 1 else 0)
        
    ## select class of top 10 except health service and sign and symptom
    df = df[(df['1.infection'] == 1) | (df['2.neoplasms'] == 1) | 
                (df['3.blood diseases'] == 1) | (df['4.endocrine'] == 1) | 
                (df['9.circulatory system'] == 1) | (df['10.respiratory system'] == 1) |
                (df['11.digestive system'] == 1) | (df['14.genitourinary system'] == 1)]
    
    df = df.drop(columns = {'5.mental disorder', '6.nervous system', '7.eye and adnexa',
                            '8.ear and mastoid','12.skin', '13.musculoskeletal system',
                            '15.pregnancy', '16.perinatal','17.congenital mal', 
                            '18.symptoms and sign', '19.injury','20.external causes', 
                            '21.health services', '22.special purposes'})
    
    
    ## text to set
    df['dx'] = df['dx'].apply(set) # for each sample make icd codes as set
    df['dx'] = df.dx.apply(lambda x: list(x & top_code))
    ## only includes row contain ICD codes from top50 
    df = df[df.dx.astype(str) != '[]']
    del regex, reg, disease, top_code
    
    # =============================================================================
    # # dummy dx drop columns
    # =============================================================================
    mlb = MultiLabelBinarizer(sparse_output=True)
    df = df.join(pd.DataFrame.sparse.from_spmatrix(
        mlb.fit_transform(df['dx']),index=df.index,
        columns=mlb.classes_))
    
    infection_list     = ['A', 'B']
    neoplasms_list     = ['C', 'D0', 'D1', 'D2', 'D3', 'D4']
    blooddis_list      = ['D5', 'D6', 'D7', 'D8', 'D9']
    endocrine_list     = ['E']
    circulatory_list   = ['I']
    respiratory_list   = ['J']
    digestive_list     = ['K']
    genitourinary_list = ['N']
    
    cols_intersect = df[df['set'] == 'train'][df[df['set'] == 'train'].columns[30:]].sum().index.tolist()
    
    a1 = [word for subs in infection_list      for word in cols_intersect if subs in word]
    a2 = [word for subs in neoplasms_list      for word in cols_intersect if subs in word]
    a3 = [word for subs in blooddis_list       for word in cols_intersect if subs in word]
    a4 = [word for subs in endocrine_list      for word in cols_intersect if subs in word]
    a5 = [word for subs in circulatory_list    for word in cols_intersect if subs in word]
    a6 = [word for subs in respiratory_list    for word in cols_intersect if subs in word]
    a7 = [word for subs in digestive_list      for word in cols_intersect if subs in word]
    a8 = [word for subs in genitourinary_list  for word in cols_intersect if subs in word]
    
    cols_intersect = sorted(list(set(a1) | set(a2) | set(a3) | set(a4) | 
                                 set(a5) | set(a6) | set(a7) | set(a8)))
    
    df = df[df.columns[:30].tolist() + cols_intersect]
    del mlb,a1,a2,a3,a4,a5,a6,a7,a8,infection_list,neoplasms_list,blooddis_list,endocrine_list
    del circulatory_list,respiratory_list,digestive_list,genitourinary_list,cols_intersect
    # =============================================================================
    # 
    # =============================================================================
    # X of english translation
    X_train = pd.DataFrame(df[df['set'] == 'train'][['an','text_en_pun']])
    X_train = X_train.reset_index(drop = True)
    
    X_val = pd.DataFrame(df[df['set'] == 'val'][['an','text_en_pun']])
    X_val = X_val.reset_index(drop = True)
    
    X_test = pd.DataFrame(df[df['set'] == 'test'][['an','text_en_pun']])
    X_test = X_test.reset_index(drop = True)
    
    # y of english translation
    y_train = df[df['set'] == 'train'][df.columns[30:]].values.astype('int')
    y_val = df[df['set'] == 'val'][df.columns[30:]].values.astype('int')
    y_test = df[df['set'] == 'test'][df.columns[30:]].values.astype('int')
    y_list = df.columns[30:].tolist()
    
    y_chapter_train = df[df['set'] == 'train'][df.columns[22:30]].values.astype('int')
    y_chapter_val = df[df['set'] == 'val'][df.columns[22:30]].values.astype('int')
    y_chapter_test = df[df['set'] == 'test'][df.columns[22:30]].values.astype('int')
    y_chapter_list = df.columns[22:30].tolist()
    
    chunk_len, overlap = 256, 50
    train_ids, train_tokens, train_origin_ids = token_index(X_train['text_en_pun'], chunk_len, overlap)
    
    val_ids, val_tokens, val_origin_ids = token_index(X_val['text_en_pun'], chunk_len, overlap)
    
    test_ids, test_tokens, test_origin_ids = token_index(X_test['text_en_pun'], chunk_len, overlap)
    # =============================================================================
    # # padding zero for each sample ids
    # =============================================================================
    print('\nPadding/truncating all sentences to %d values...' % chunk_len)
    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))
    # Pad our input tokens with value 0.
    train_ids = pad_sequences(train_ids, maxlen=chunk_len, dtype="long", 
                              value=0, truncating="post", padding="post")
    train_ids = torch.tensor(train_ids).to(torch.int64)
    
    val_ids = pad_sequences(val_ids, maxlen=chunk_len, dtype="long", 
                              value=0, truncating="post", padding="post")
    val_ids = torch.tensor(val_ids).to(torch.int64)
    
    test_ids = pad_sequences(test_ids, maxlen=chunk_len, dtype="long", 
                              value=0, truncating="post", padding="post")
    test_ids = torch.tensor(test_ids).to(torch.int64)
    print('\nDone.')
    
    train_masks = gen_att_mask(train_ids)
    
    val_masks = gen_att_mask(val_ids)
    
    test_masks = gen_att_mask(test_ids)
    # =============================================================================
    # export as pickle
    # =============================================================================
     save_path = r'/home/wanchana/Data/thesis_data/data/final_dataset/bert_input/{}'.format(sheet)
     if not os.path.exists(save_path):
         os.makedirs(save_path)

     with open(save_path + r'/train_origin_ids.pickle', 'wb') as f:
         pickle.dump(train_origin_ids, f)
     with open(save_path + r'/test_origin_ids.pickle', 'wb') as f:
         pickle.dump(test_origin_ids, f)
     with open(save_path + r'/val_origin_ids.pickle', 'wb') as f:
         pickle.dump(val_origin_ids, f)
        
     with open(save_path + r'/train_ids.pickle', 'wb') as f:
         pickle.dump(train_ids, f)
     with open(save_path + r'/test_ids.pickle', 'wb') as f:
         pickle.dump(test_ids, f)
     with open(save_path + r'/val_ids.pickle', 'wb') as f:
         pickle.dump(val_ids, f)
        
     with open(save_path + r'/train_masks.pickle', 'wb') as f:
         pickle.dump(train_masks, f)
     with open(save_path + r'/test_masks.pickle', 'wb') as f:
         pickle.dump(test_masks, f)
     with open(save_path + r'/val_masks.pickle', 'wb') as f:
         pickle.dump(val_masks, f)
        
     with open(save_path + r'/X_train.pickle', 'wb') as f:
         pickle.dump(X_train, f)
     with open(save_path + r'/X_test.pickle', 'wb') as f:
         pickle.dump(X_test, f)
     with open(save_path + r'/X_val.pickle', 'wb') as f:
         pickle.dump(X_val, f)
        
     with open(save_path + r'/y_train.pickle', 'wb') as f:
         pickle.dump(y_train, f)
     with open(save_path + r'/y_test.pickle', 'wb') as f:
         pickle.dump(y_test, f)
     with open(save_path + r'/y_val.pickle', 'wb') as f:
         pickle.dump(y_val, f)
     with open(save_path + r'/y_list.pickle', 'wb') as f:
         pickle.dump(y_list, f)
        

# execute
extract_input('top_50')
