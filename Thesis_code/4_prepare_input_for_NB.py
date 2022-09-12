#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 15:38:37 2021

@author: wanchana
"""
import os
import random
import pandas as pd
import numpy as np
import re
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
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
# specify path
# =============================================================================
path = ''
os.chdir(path)
# =============================================================================
# 
# =============================================================================
def extract_input(sheet):
    # sheet = 'top_50'
    # raw dataset
    df_raw = pd.read_csv("Dataset_final.csv", 
                         sep = '|',dtype = {'an': str, 'hn': str, 'gender' : int, 'age':int})
    # subset used
    top_code = pd.read_excel("top_freq_ICD.xlsx", 
                           dtype = str, sheet_name = '{}'.format(sheet)) # import set of code for freq 25
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
    
    # # text to set
    df['dx'] = df['dx'].apply(set) # for each sample make icd codes as set
    df['dx'] = df.dx.apply(lambda x: list(x & top_code))
    # # only includes row contain ICD codes from freq25 
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
    X_train = pd.DataFrame(df[(df['set'] == 'train') | (df['set'] == 'val')][['an','text_en_pun']])
    X_train = X_train.reset_index(drop = True)
    
    X_test = pd.DataFrame(df[df['set'] == 'test'][['an','text_en_pun']])
    X_test = X_test.reset_index(drop = True)
    
    # y of english translation
    y_train = df[(df['set'] == 'train') | (df['set'] == 'val')][df.columns[30:]].values.astype('int')
    y_test = df[df['set'] == 'test'][df.columns[30:]].values.astype('int')
    y_list = df.columns[30:].tolist()
    
    # =============================================================================
    # pre-processing
    # =============================================================================
    contractions = {"ain't": "are not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
                    "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have",
                    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                    "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                    "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is",
                    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                    "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                    "it'll": "it will", "it'll've": "it will have","it's": "it is","let's": "let us","ma'am": "madam",
                    "mayn't": "may not","might've": "might have","mightn't": "might not","mightn't've": "might not have",
                    "must've": "must have","mustn't": "must not","mustn't've": "must not have","needn't": "need not",
                    "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have",
                    "shan't": "shall not","sha'n't": "shall not", "shan't've": "shall not have","she'd": "she would",
                    "she'd've": "she would have","she'll": "she will","she'll've": "she will have","she's": "she is",
                    "should've": "should have","shouldn't": "should not","shouldn't've": "should not have",
                    "so've": "so have","so's": "so is","that'd": "that would","that'd've": "that would have",
                    "that's": "that is","there'd": "there would","there'd've": "there would have",
                    "there's": "there is","they'd": "they would","they'd've": "they would have","they'll": "they will",
                    "they'll've": "they will have","they're": "they are","they've": "they have","to've": "to have",
                    "wasn't": "was not","we'd": "we would","we'd've": "we would have","we'll": "we will",
                    "we'll've": "we will have","we're": "we are","we've": "we have","weren't": "were not",
                    "what'll": "what will","what'll've": "what will have","what're": "what are","what's": "what is",
                    "what've": "what have", "when's": "when is","when've": "when have","where'd": "where did",
                    "where's": "where is","where've": "where have","who'll": "who will","who'll've": "who will have",
                    "who's": "who is","who've": "who have","why's": "why is","why've": "why have", "will've": "will have",
                    "won't": "will not","won't've": "will not have","would've": "would have","wouldn't": "would not",
                    "wouldn't've": "would not have","y'all": "you all","y'all'd": "you all would",
                    "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                    "you'd": "you would","you'd've": "you would have","you'll": "you will","you'll've": "you will have",
                    "you're": "you are","you've": "you have"}
    
    # list of stopword
    stopWords = set(stopwords.words('english'))
    
    ## preprocessing
    def prep(df):
        # lower case, lower the capital letter
        df['text_en_pun'] = df['text_en_pun'].astype(str).str.lower()
        # contraction
        df['text_en_pun'] = df['text_en_pun'].str.split().apply(lambda x: ' '.join([contractions.get(e, e) for e in x]))
        # lemmatization, standardize similar word
        lemmatizer = WordNetLemmatizer()
        df['text_en_pun'] = df['text_en_pun'].apply(lambda x : lemmatizer.lemmatize(x))
        # Removing stopword
        df['text_en_pun'] = df['text_en_pun'].apply(lambda x : " ".join(x for x in x.split() if x not in stopWords))
        # Removing Punctuation
        df['text_en_pun'] = df['text_en_pun'].str.replace(r'\/',' ')
        df['text_en_pun'] = df['text_en_pun'].str.replace(r'\:',' ')
        df['text_en_pun'] = df['text_en_pun'].str.replace(r'[^\w\s]','')
        df['text_en_pun'] = df['text_en_pun'].str.replace('_','')
        # Removing numeric
        df['text_en_pun'] = df['text_en_pun'].str.replace(r'\d+', '')
        return df['text_en_pun']
    
    X_train['text_en_pun'] = prep(X_train)
    X_test['text_en_pun'] = prep(X_test)
    # =============================================================================
    # export as pickle
    # =============================================================================
     save_path = r'/home/wanchana/Data/thesis_data/data/final_dataset/nb_input/{}'.format(sheet)
     if not os.path.exists(save_path):
         os.makedirs(save_path)

     with open(save_path + r'/X_train.pickle', 'wb') as f:
         pickle.dump(X_train, f)
     with open(save_path + r'/X_test.pickle', 'wb') as f:
         pickle.dump(X_test, f)

     with open(save_path + r'/y_train.pickle', 'wb') as f:
         pickle.dump(y_train, f)
     with open(save_path + r'/y_test.pickle', 'wb') as f:
         pickle.dump(y_test, f)
     with open(save_path + r'/y_list.pickle', 'wb') as f:
         pickle.dump(y_list, f)
    
extract_input('top_50')
