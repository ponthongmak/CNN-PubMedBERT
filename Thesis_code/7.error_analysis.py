#!/usr/bin/env python3
"""
Created on Mon Nov 22 09:50:19 2021
@author: wanchana
"""
# =============================================================================
# specify path
# =============================================================================
import os 
import sys
sys.path.append('Thesis_code')
path = ''
os.chdir(path)

load_path = ''
save_path = ''
if not os.path.exists(save_path):
    os.makedirs(save_path)
# =============================================================================
# import dependency
# =============================================================================
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np
import pickle
from utils import helper
# =============================================================================
# Set seed and shuld divice
# =============================================================================
helper.set_seed(2021) # set seed
device = helper.get_device(1) # number indicate device position
# =============================================================================
# import dataset
# =============================================================================
# raw dataset
with open(load_path + r'X_test.pickle','rb') as handle:
    data = pickle.load(handle)
    data.columns = ['an','prep_text']
with open(load_path + r'y_test.pickle','rb') as handle:
    y_true = pickle.load(handle)

with open(save_path + r'results_pred_test_callibrate_thres.pickle','rb') as handle:
    y_pred = pickle.load(handle).to_numpy()

with open(load_path + r'y_list.pickle','rb') as handle:
    label_list = pickle.load(handle)

data_raw = pd.read_csv(path + r"Dataset_final.csv", 
                     sep = '|',dtype = {'an': str, 'hn': str, 'gender' : int, 'age':int},
                     usecols= ['an','brief_raw','course_raw','text_en','text_en_pun'])

y_true = pd.DataFrame(y_true, columns=label_list)
y_pred = pd.DataFrame(y_pred, columns=label_list)
# =============================================================================
# 
# =============================================================================
def interested_label(loc):
    temp_y_true = y_true[[label_top[loc]]]
    temp_y_pred = y_pred[[label_top[loc]]]
    
    compare_y = pd.concat([temp_y_true, temp_y_pred], axis = 1)
    compare_y.columns = ['y_true','y_pred']
    compare_y['correct'] = (compare_y['y_true'] == compare_y['y_pred']) * 1
    
    x = y_true.replace(0, np.nan)
    x = x.replace(1, pd.Series(y_true.columns, y_true.columns))
    
    xx = x.T.apply(lambda x: x.dropna().tolist())
    xx.name = 'icd10'
    compare_y = compare_y.join(xx)
    
    compare_y = compare_y.join(data)
    compare_y = compare_y.merge(data_raw, how = 'left', on = 'an')

    
    y_true_pos = compare_y[(compare_y['correct'] == 1) & (compare_y['y_pred'] == 1)]
    y_true_neg = compare_y[(compare_y['correct'] == 1) & (compare_y['y_pred'] == 0)]

    y_false_pos = compare_y[(compare_y['correct'] == 0) & (compare_y['y_pred'] == 1)]
    y_false_neg = compare_y[(compare_y['correct'] == 0) & (compare_y['y_pred'] == 0)]
    
    return y_true_pos, y_true_neg, y_false_pos, y_false_neg

# =============================================================================
# token counts
# =============================================================================
# instantiate the vectorizer object
def extract_grams_freq(data_set, top):
    unigram_count = CountVectorizer(analyzer= 'word', stop_words='english', ngram_range = (1,1))
    bigram_count = CountVectorizer(analyzer= 'word', stop_words='english', ngram_range = (2,2))
    trigram_count = CountVectorizer(analyzer= 'word', stop_words='english', ngram_range = (3,3))

    # unigram_count = TfidfVectorizer(analyzer= 'word', stop_words='english', ngram_range = (1,1))
    # bigram_count = TfidfVectorizer(analyzer= 'word', stop_words='english', ngram_range = (2,2))
    # trigram_count = TfidfVectorizer(analyzer= 'word', stop_words='english', ngram_range = (3,3))

    temp = unigram_count.fit(data['prep_text'])
    
    
    temp = temp.transform(data_set['prep_text'])
    freq = sum(temp).toarray()[0]
    freq_one = pd.DataFrame(freq, index = unigram_count.get_feature_names(), columns=['freq_ratio'])
    freq_one = freq_one.sort_values(['freq_ratio'], ascending=False).reset_index()[:top]

    temp = bigram_count.fit(data['prep_text'])
    temp = temp.transform(data_set['prep_text'])
    freq = sum(temp).toarray()[0]
    freq_bi = pd.DataFrame(freq, index = bigram_count.get_feature_names(), columns=['freq_ratio'])
    freq_bi = freq_bi.sort_values(['freq_ratio'], ascending=False).reset_index()[:top]

    temp = trigram_count.fit(data['prep_text'])
    temp = temp.transform(data_set['prep_text'])
    freq = sum(temp).toarray()[0]
    freq_tri = pd.DataFrame(freq, index = trigram_count.get_feature_names(), columns=['freq_ratio'])
    freq_tri = freq_tri.sort_values(['freq_ratio'], ascending=False).reset_index()[:top]

    freq_grams = pd.concat([freq_one, freq_bi, freq_tri], axis = 0)
    freq_grams['freq_ratio'] = freq_grams['freq_ratio'] / len(data_set)
    return freq_grams


# =============================================================================
# select label
# =============================================================================
stats_cb = pd.read_excel(save_path + 'each_class_stats_test_callibrate_thres.xlsx')
stats_f1 = pd.read_excel(save_path + 'each_class_stats_test_f1_thres.xlsx')


# label_top = []
# temp = stats.sort_values(['F1'], ascending = [False]).head(2).reset_index()['labels'].tolist()
# label_top.extend(temp)
# temp = stats.sort_values(['F1'], ascending = [True]).head(2).reset_index()['labels'].tolist()
# label_top.extend(temp)

label_top = ['J189','D649', 'N185','C20']


fame = pd.concat([pd.DataFrame(y_true.sum()), pd.DataFrame(y_pred.sum())], axis = 1)
fame.columns = ['true', 'pred']
fame['dif'] = fame['true'] / fame['pred']


# convert documents into a matrix
for i, j in enumerate(label_top):
    if not os.path.exists(r'/home/wanchana/Data/thesis_data/data/error_analysis/{}/'.format(j)):
        os.makedirs(r'/home/wanchana/Data/thesis_data/data/error_analysis/{}/'.format(j))
    y_true_pos, y_true_neg, y_false_pos, y_false_neg = interested_label(i)
    
    y_true_pos.to_excel(r'error_analysis/{}/data_{}_tp.xlsx'.format(j,j), index = False)
    y_true_neg.to_excel(r'error_analysis/{}/data_{}_tn.xlsx'.format(j,j), index = False)
    y_false_pos.to_excel(r'error_analysis/{}/data_{}_fp.xlsx'.format(j,j), index = False)
    y_false_neg.to_excel(r'error_analysis/{}/data_{}_fn.xlsx'.format(j,j), index = False)
    
    try:
        freq_grams_tp = extract_grams_freq(y_true_pos, 20)
        globals()['summary_{}_tp'.format(j)] = freq_grams_tp
        globals()['summary_{}_tp'.format(j)].to_excel(r'error_analysis/{}/summary_{}_tp.xlsx'.format(j,j), index = False)
    except:
        pass

    try:
        freq_grams_tn = extract_grams_freq(y_true_neg, 20)
        globals()['summary_{}_tn'.format(j)] = freq_grams_tn
        globals()['summary_{}_tn'.format(j)].to_excel(r'error_analysis/{}/summary_{}_tn.xlsx'.format(j,j), index = False)
    except:
        pass

    try:
        freq_grams_fp = extract_grams_freq(y_false_pos, 20)
        globals()['summary_{}_fp'.format(j)] = freq_grams_fp
        globals()['summary_{}_fp'.format(j)].to_excel(r'error_analysis/{}/summary_{}_fp.xlsx'.format(j,j), index = False)
    except:
        pass

    try:
        freq_grams_fn = extract_grams_freq(y_false_neg, 20)
        globals()['summary_{}_fn'.format(j)] = freq_grams_fn
        globals()['summary_{}_fn'.format(j)].to_excel(r'error_analysis/{}/summary_{}_fn.xlsx'.format(j,j), index = False)
    except:
        pass

    try:
        tp = globals()['summary_{}_tp'.format(j)]['index'].tolist()
        tn = globals()['summary_{}_tn'.format(j)]['index'].tolist()
        fp = globals()['summary_{}_fp'.format(j)]['index'].tolist()
        fn = globals()['summary_{}_fn'.format(j)]['index'].tolist()

        term_tp = pd.DataFrame(list(set(tp) - set(tn) - set(fn))) # top tp terms without tn terms
        term_tp.columns = ['index']
        term_tp = term_tp.merge(freq_grams_tp, how = 'left', on = 'index')        
        term_tp.to_excel(r'error_analysis/{}/term_{}_tp.xlsx'.format(j,j), index = False)
        
        term_tn = pd.DataFrame(list((set(tn) - set(tp)) - set(fp))) # top tn terms without tp terms
        term_tn.columns = ['index']
        term_tn = term_tn.merge(freq_grams_tn, how = 'left', on = 'index')        
        term_tn.to_excel(r'error_analysis/{}/term_{}_tn.xlsx'.format(j,j), index = False)
        
        term_fp = pd.DataFrame(list(set(fp) & set(tp) - set(tn) - set(fn))) # top fp that confused model
        term_fp.columns = ['index']
        term_fp = term_fp.merge(freq_grams_fp, how = 'left', on = 'index')        
        term_fp.to_excel(r'error_analysis/{}/term_{}_fp.xlsx'.format(j,j), index = False)
        
        term_fn = pd.DataFrame(list(set(fn) & set(tn) - set(tp) - set(fp))) # top fn that confused model
        term_fn.columns = ['index']
        term_fn = term_fn.merge(freq_grams_fn, how = 'left', on = 'index')        
        term_fn.to_excel(r'error_analysis/{}/term_{}_fn.xlsx'.format(j,j), index = False)
        
    except:

        tn = globals()['summary_{}_tn'.format(j)]['index'].tolist()
        fp = globals()['summary_{}_fp'.format(j)]['index'].tolist()
        fn = globals()['summary_{}_fn'.format(j)]['index'].tolist()

        term_tn = pd.DataFrame(list(set(tn) - set(fn)))
        term_tn.columns = ['index']
        term_tn = term_tn.merge(freq_grams_tn, how = 'left', on = 'index')         
        term_tn.to_excel(r'error_analysis/{}/term_{}_tn.xlsx'.format(j,j), index = False)
        
        term_fp = pd.DataFrame(list(set(fp) - set(tn)))
        term_fp.columns = ['index']
        term_fp = term_fp.merge(freq_grams_fp, how = 'left', on = 'index')        
        term_fp.to_excel(r'error_analysis/{}/term_{}_fp.xlsx'.format(j,j), index = False)
        
        term_fn = pd.DataFrame(list(set(fn) & set(tn) - set(fp))) # top fn that confused model
        term_fn.columns = ['index']
        term_fn = term_fn.merge(freq_grams_fn, how = 'left', on = 'index')        
        term_fn.to_excel(r'error_analysis/{}/term_{}_fn.xlsx'.format(j,j), index = False)