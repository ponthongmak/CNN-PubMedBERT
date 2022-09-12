# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:48:47 2020

@author: Wanchana
"""
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import os
from ast import literal_eval
from matplotlib import pyplot as plt
import re

# path
os.chdir('')
# =============================================================================
#     # import dataframe
# =============================================================================
df_raw = pd.read_csv(r'raw_data.csv', sep='|',
                     dtype={'hn': str, 'an': str})
# =============================================================================
# import translation dataset
# =============================================================================
# this study used Google Translate from Google sheet to convert Thai to English
# then save it as excel files into "translation" path
path = '/translation/'
files = os.listdir(path)
df_trn = pd.DataFrame()
for f in files:
    data = pd.read_excel(path + f, dtype=str)
    df_trn = df_trn.append(data)
del data
	# rename columns
df_trn.columns = ['an', 'brief_raw', 'course_raw', 'brief_en', 'course_en']
	# fill missing with blank
df_trn = df_trn.fillna('')
df_trn = df_trn.sort_values('an', ascending=True)
df_trn = df_trn.drop_duplicates('an')
	# demographic and text data
temp1 = df_raw[df_raw.columns[0:9]]
temp1 = temp1.merge(df_trn, how='left', on='an')
temp1 = temp1.drop(columns={'biref', 'course'})
	# only ICD-10 codes
temp2 = df_raw[df_raw.columns[9:]]
	# merge demograpich and text data
df = pd.concat([temp1, temp2], axis=1)
del temp1, temp2, df_trn
# =============================================================================
# remove thai waord
# =============================================================================
# remove thai patterns
pattern_thai = r"[ก-๙]+"
df['brief_en'] = df['brief_en'].str.replace(pattern_thai, "")
df['course_en'] = df['course_en'].str.replace(pattern_thai, "")

	# find thai char
chars_th = [chr(c) for c in range(3585, 3676)]
# there are still some thai characters
check_th = df[(df.brief_en.str.contains('|'.join(chars_th))) |
              (df.course_en.str.contains('|'.join(chars_th)))][['an']]
	# no record has thai pattern

def removespace_bf_slash(df):
    # remove space before slash which occured during language translation
    chars_th = [chr(c) for c in range(3585, 3676)]
    df['brief_en'] = df['brief_en'].apply(
        lambda x: re.sub(' ([/]) ?', r'\1', x))
    df['brief_en'] = df['brief_en'].str.replace(
        '|'.join(map(re.escape, chars_th)), '')
    df['course_en'] = df['course_en'].apply(
        lambda x: re.sub(' ([/]) ?', r'\1', x))
    df['course_en'] = df['course_en'].str.replace(
        '|'.join(map(re.escape, chars_th)), '')
    return df

df = removespace_bf_slash(df)

# create new columns only thai word removal
def remove_th(df):
    # remove thai alphabets
    # replace double whitespaces with single whitespace
    pattern_thai = r"[ก-๙]+"
    x1 = df['brief_raw'].str.replace(pattern_thai, "")
    x2 = df['course_raw'].str.replace(pattern_thai, "")
    df.insert(11, 'brief_th', x1)
    df.insert(12, 'course_th', x2)
    return df

df = remove_th(df)

df['dx']  = df['dx'].apply(literal_eval) # convert text to list data type
# =============================================================================
# train test split
# =============================================================================
# create X and y
def trn_tst_split(df, random_state):
    
    data = df.copy()

    X = data[data.columns[0:13]]
    y = data[data.columns[13:]]
    
    # stratify train test for multi-label
    gs = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state = random_state)
    
    remain_idx, test_idx = next(gs.split(X, y, groups=X['hn']))
    
    remain = df.loc[remain_idx].reset_index()
    test = df.loc[test_idx]
    test.insert(0,'set','test')
    
    X = remain[remain.columns[0:13]]
    y = remain[remain.columns[13:]]
    
    gs = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state = random_state)
    train_idx, val_idx = next(gs.split(X, y, groups=X['hn']))
    
    train = remain.loc[train_idx].set_index('index')
    train.insert(0,'set','train')
    
    val = remain.loc[val_idx].set_index('index')
    val.insert(0,'set','val')
    
    dataset = pd.concat([train,test,val], axis = 0)
    return dataset

dataset_full = trn_tst_split(df, random_state=47)
dataset_full[dataset_full.columns[0:14]].to_csv('dataset_full.csv', sep='|', index=False)