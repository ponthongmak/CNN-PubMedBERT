()(# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:40:03 2020

@author: Wanchana
"""
# %% Libraries
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from keras.preprocessing.text import Tokenizer
import pandas as pd
import os
import numpy as np
import re
# =============================================================================
# specify path
# =============================================================================
os.chdir('')
path = ''
# =============================================================================
# %%Import dataset
# =============================================================================
X = pd.read_csv('dataset_full.csv', sep='|', 
                 dtype={'hn': str, 'an': str})

# =============================================================================
# %% Clean data
# =============================================================================
# create X_all for clean data
X_all = X[X['set'] != 'test']
X_test = X[X['set'] == 'test']

def combine_txt(df, cols, col_name):
    '''
    create new column from combined 2 columns in dataframe then lower case
    ----------
    df : dataframe
    cols : list of string
        list of columns needed to combine
    col_name : string
        new column name
    Returns
    -------
    df : dataframe
    '''
    # replace multiple whitespace with single splace
    for col in cols:
        df[col] = df[col].astype(str).apply(
            lambda x: re.sub(r'\s+', ' ', x))

    df[col_name] = df[cols[0]].copy() + r' ' + df[cols[1]].copy()
    # lower case
    df[col_name] = df[col_name].str.lower()
    return df

X_all = combine_txt(X_all, ['brief_en', 'course_en'], 'text_en')
X_all = combine_txt(X_all, ['brief_th', 'course_th'], 'text_th')

X_test = combine_txt(X_test, ['brief_en', 'course_en'], 'text_en')
X_test = combine_txt(X_test, ['brief_th', 'course_th'], 'text_th')
# =============================================================================
# ## add abbreviation dict
# =============================================================================
# use (word count freq >= 250) to create abbreviation dict
abbr_dict = pd.read_excel('abbr_bert.xlsx', dtype=str, sheet_name='Sheet1')
abbr_dict['token'] = abbr_dict['token'].str.lower()

abbr_dict = abbr_dict.drop_duplicates(['token', 'term used'])
abbr_dict = dict(zip(abbr_dict['token'], abbr_dict['term used']))
	# 825 after remove upper case --> from top 250 abbr of en_all


def replace_abbr(df, cols, suffix):
    # replace non-readable abbreviation by readable text
    for col in cols:
        df[col + '{}'.format(suffix)] = df[col].str.split().apply(
            lambda x: ' '.join([abbr_dict.get(e, e) for e in x]))
    return df

X_all = replace_abbr(X_all, ['text_en', 'text_th'], '_abbr')
X_test = replace_abbr(X_test, ['text_en', 'text_th'], '_abbr')
# =============================================================================
# reason for clean top freq 1%
# =============================================================================
# from subword of the largest dataset
# verify list
unread1percent_dict = pd.read_excel('word_unreadable_bert_freq100_read_bert.xlsx', dtype=str, sheet_name='Sheet1')
unread1percent_dict['expert_replace'] = unread1percent_dict['expert_replace'].replace({np.nan: ''})

unread1percent_dict = dict(zip(unread1percent_dict['token'], unread1percent_dict['expert_replace']))

def replace_unread(df, cols, suffix, l_from, l_to):
    for col in cols:
        df[col[l_from: l_to] + '{}'.format(suffix)] = df[col].str.split().apply(
            lambda x: ' '.join([unread1percent_dict.get(e, e) for e in x]))
    return df

X_all = replace_unread(X_all, ['text_en_abbr', 'text_th_abbr'], '_repunread', 0, 7)
X_test = replace_unread(X_test, ['text_en_abbr', 'text_th_abbr'], '_repunread', 0, 7)
# =============================================================================
# replace punctuation
# =============================================================================
def replace_punc(x):
    n = 0
    while True:
        x_old = x.copy()
        
        ## replace by dictionary
        x = x.str.split().apply(lambda x: ' '.join([abbr_dict.get(e, e) for e in x]))
        x = x.str.split().apply(lambda x: ' '.join([unread1percent_dict.get(e, e) for e in x]))
        
        ## remove 
        x = x.str.replace(r'\u200b', '')  # "zero-width-space" in Unicode
        
        x = x.str.replace(r'\s(?=/o$)', '') # remove space before /o
        x = x.str.replace(r'\s(?=/o\s)', '') # remove space before /o
        
        x = x.str.replace(r'\s(?=/c$)', '') # remove space before /c
        x = x.str.replace(r'\s(?=/c\s)', '') # remove space before /c
        
        x = x.str.replace(r'\s(?=/e$)', '') # remove space before /e
        x = x.str.replace(r'\s(?=/e\s)', '') # remove space before /e
        
        x = x.str.replace(r'\s(?=/s$)', '') # remove space before /s
        x = x.str.replace(r'\s(?=/s\s)', '') # remove space before /s
        x = x.str.replace(r'\s(?=/s:$)', '') # remove space before /s:
        x = x.str.replace(r'\s(?=/s:\s)', '') # remove space before /s:
    
        x = x.str.replace(r'(?<=\su/)\s', '') # remove space after u/
        x = x.str.replace(r'(?<=^u/)\s', '') # remove space after u/
    
        x = x.str.replace(r'(?<=\sc/)\s', '') # remove space after c/
        x = x.str.replace(r'(?<=^c/)\s', '') # remove space after c/
        
        x = x.str.replace(r'(?<=\sf/)\s', '') # remove space after f/
        x = x.str.replace(r'(?<=^f/)\s', '') # remove space after f/
    
        x = x.str.replace(r'(?<=\sh/)\s', '') # remove space after h/
        
        x = x.str.replace(r'(?<=\sv/)\s', '') # remove space after v/
    
        x = x.str.replace(r'(?<=[\+\-])\s(?=ve\s)', '') # remove space before v if before  + -
        
    
        ## symbolic replacement
        x = x.str.replace(r'\&', ' and ')  # replace &
        x = x.str.replace(r'\?', ' ')
        x = x.str.replace(r'\$', ' ')
        x = x.str.replace(r'\#', ' ')
        x = x.str.replace(r'\!', ' ')
        x = x.str.replace(r'\;', ' ')
        x = x.str.replace(r"'s", ' ')
        x = x.str.replace(r"\'", ' ')
        x = x.str.replace(r'\"', ' ')
        x = x.str.replace(r'\[', ' ')
        x = x.str.replace(r'\]', ' ')
        x = x.str.replace(r'\(', ' ')
        x = x.str.replace(r'\)', ' ')
        x = x.str.replace(r'\{', ' ')
        x = x.str.replace(r'\}', ' ')
        x = x.str.replace(r'donâ € ™', ' ')
        x = x.str.replace(r'\•', ' ')
        x = x.str.replace(r'\»', '')
        x = x.str.replace(r'\°\:', ' ')
        x = x.str.replace(r'\°', ' ')
        x = x.str.replace(r'\®', ' ')
        x = x.str.replace(r'\¤', ' ')
        x = x.str.replace(r'\|', ' ')
        x = x.str.replace(r'\@', ' ')
        x = x.str.replace(r'\\', ' ')
        x = x.str.replace(r'\^\^+', ' ')
        x = x.str.replace(r'_/', ' ')
        
        x = x.str.replace(r'(^|\s)\<+\-+\>+(\s+|$)', ' ') # replace <-> or <<-->>
        x = x.str.replace(r'(^|\s)\-+\>+(\s+|$)', ' ')  # replace -> or -->>
        x = x.str.replace(r'(^|\s)\<+\=+\>+(\s+|$)', ' ') # replace <=> or <<==>>
        x = x.str.replace(r'(^|\s)\=+\>+(\s+|$)', ' ')  # replace => or ==>> 
        x = x.str.replace(r'(^|\s)\<+\>+(\s+|$)', ' ')  # replace <<>>
        x = x.str.replace(r'<>/<>', ' ')  # replace <>/<>
        x = x.str.replace(r'<>:', ' ')  # replace <>:
        x = x.str.replace(r'(^|\s*)\>>+(\s*|$)', ' ')  # replace >>
        x = x.str.replace(r'(^|\s*)\<<+(\s*|$)', ' ')  # replace <<
        
        x = x.str.replace(r'(^|\s)\==+\:(\s+|$)', ' ')
        x = x.str.replace(r'(^|\s)\==+(\s+|$)', ' ')  # replace  ==
        
        x = x.str.replace(r'(^|\s)\-+\s*ve(\s+|$)', ' negative ')  # replace ce -ve, --ve
        x = x.str.replace(r'(^|\s)\++\s*ve(\s+|$)', ' positive ')
        
        x = x.str.replace(r'(^|\s+)%(?=[a-z0-9])', ' ') # replace %alphabet or  %0-9
        x = x.str.replace(r'\%\%+', '%') # %% to %
        
        x = x.str.replace(r'(^|\s+)::+\s+', ' ')  # replace : or ::
        x = x.str.replace(r'\:\:', ':')  # replace ::
        
        x = x.str.replace(r'(?<=[\d+]),(?=\d\d\d+)', '') # remove , within number 1,000
        x = x.str.replace(r',', ' ')  # replace all ,
        
        x = x.str.replace(r'\*\*+', ' ')  # replace **+
        
        x = x.str.replace(r'\-\s+', ' ')  # - before space+
        x = x.str.replace(r'\-\-+', ' ')  # --+
        x = x.str.replace(r'(\s+)\-\.', ' ')  # space+ before -.
        x = x.str.replace(r'(^|\s)-(?![0-9])', ' ')     # remove - before character except number
        
        x = x.str.replace(r'(^|\s*)\.(?![0-9])', ' ')  # replace . except 0.0
        
        x = x.str.replace(r'\/\/+', ' ')  # //+
        
        
        ## word replacement
        x = x.str.replace(r'(^|\s)d\s*\/\s*c(\s+|$)', ' discharge ')
        
        x = x.str.replace(r'(^|\s)g/sc/s(\s+|$)', ' g/s c/s ')
        x = x.str.replace(r'(^|\s)s/pu/s(\s+|$)', ' s/p u/s ')
        x = x.str.replace(r'(^|\s)f/ev/vv/v(\s+|$)',' f/e v/v v/v ')
        x = x.str.replace(r'(^|\s)(^|\s)f/ev/v(\s+|$)',' f/e v/v ')  
        x = x.str.replace(r'(^|\s)v/vv/v(\s+|$)', ' v/v v/v ')
        
        x = x.str.replace(r'(^|\s)hd-mtx/ara-c(\s+|$)', ' hd mtx and ara c ') 
        x = x.str.replace(r'(^|\s)mtx/ara-c(\s+|$)', ' mtx and ara c ')
        x = x.str.replace(r'(^|\s)ctwa:(\s+|$)', ' ct wa ') 
        x = x.str.replace(r'(^|\s)atb>(\s+|$)', ' antibiotic > ') 
    
        x = x.str.replace(r'(^|\s)jsp:(\s+|$)', ' joint position sense ')  
        x = x.str.replace(r'(^|\s)poct\-glu(\s+|$)', ' point of care testing glucose ') 
        x = x.str.replace(r'(^|\s)poc\-glu(\s+|$)', ' point of care testing glucose ') 
        x = x.str.replace(r'(^|\s)v/st(\s+|$)', ' vital signs : temperature ') 
        x = x.str.replace(r'(^|\s)v/s(\s+|$): ', ' vital signs ')     
        x = x.str.replace(r'(^|\s)tcd/cdus:(\s+|$)', ' transcranial doppler and color doppler ultrasonography ') 
        x = x.str.replace(r'(^|\s)n/s(\s+|$)', ' neuro sign ')
        x = x.str.replace(r'(^|\s)n/s:(\s+|$)', ' neuro sign ')
        x = x.str.replace(r'(^|\s)onco:(\s+|$)', ' oncology ')
        x = x.str.replace(r'(^|\s)f/ev(\s+|$)', ' f/e v ')
        x = x.str.replace(r'(^|\s)p/f(\s+|$)', ' pao2 to fio2 ')
        x = x.str.replace(r'(^|\s)d/dx(\s+|$)', ' differential diagnosis ')
        x = x.str.replace(r'(^|\s)elyte:(\s+|$)', ' electrolyte ')
        x = x.str.replace(r'(^|\s)ca\s*\+\++', ' ca++ ')  # replace ca++
        x = x.str.replace(r'(?<![ca])\+\++', ' ')  # replace (++)+ but not included ca++
        
        x = x.str.replace(r'(^|\s)htk:(\s+|$)', ' heel to knee ')
        x = x.str.replace(r'(^|\s)ud:(\s+|$)', ' underlying disease ')
        x = x.str.replace(r'(^|\s)rul:(\s+|$)', ' right upper lobe ')
        x = x.str.replace(r'(^|\s)rll:(\s+|$)', ' right lower lobe ')
        x = x.str.replace(r'(^|\s)sx>(\s+|$)', ' surgery ')
        x = x.str.replace(r'(^|\s)h/cxii:(\s+|$)', ' h/cxii ')
        x = x.str.replace(r' a-ve(\s+|$)', ' a negative ')
        x = x.str.replace(r'(^|\s)ugib:(\s+|$)', ' upper gastrointestinal bleeding ')
        x = x.str.replace(r'(^|\s)wbc>(\s+|$)', ' wbc > ')
        x = x.str.replace(r'(^|\s)pre-existing(\s+|$)', ' preexisting ')
        x = x.str.replace(r'(^|\s)right-sided(\s+|$)', ' right sided ')
        x = x.str.replace(r'(^|\s)left-sided(\s+|$)', ' left sided ')
        x = x.str.replace(r'(^|\s)lie-flat(\s+|$)', ' lie flat ')
        x = x.str.replace(r'(^|\s)-1cm(\s+|$)', ' 1 cm ')
        x = x.str.replace(r'(^|\s)-1d(\s+|$)', ' 1 day ')
        x = x.str.replace(r'(^|\s)-1day(\s+|$)', ' 1 day ')
        x = x.str.replace(r'(^|\s)-1dpta(\s+|$)', ' 1 dtpa ')
        x = x.str.replace(r'(^|\s)w/uh/\sc(\s+|$)',' work up blood culture ')  
        x = x.str.replace(r'(^|\s)w\s*/\s*o(\s+|$)', ' without ')     # replace w/o, w/ o, w /o, w / o
        x = x.str.replace(r' w/uh/c ',' work up blood culture ')
        x = x.str.replace(r' w/uu/c ',' work up urine culture ') 
        x = x.str.replace(r' f/uu/s ',' follow up ultrasound ')
        x = x.str.replace(r' h/c-ng ',' blood culture no growth ')  
        x = x.str.replace(r'(^|\s)pft:(\s+|$)', ' pulmonary function test ')
        x = x.str.replace(r'(^|\s)u\s*/\s*s\:(\s+|$)', ' ultrasound ')
        x = x.str.replace(r'(^|\s)a/b(\s+|$)', ' ab ')
        x = x.str.replace(r'(^|\s)a/b-neg(\s+|$)', ' ab negative ')
        x = x.str.replace(r'(^|\s)a/b-negative(\s+|$)', ' ab negative ')
        x = x.str.replace(r'(^|\s)a/b:(\s+|$)', ' ab ')
        x = x.str.replace(r'(^|\s)a/b>(\s+|$)', ' ab ')
        x = x.str.replace(r'(^|\s)hemato:(\s+|$)', ' hematology ')
        x = x.str.replace(r'(^|\s)cdx-2:(\s+|$)', ' cdx2 ') 
        
        x = x.str.replace(r'(^|\s)s\s*/\s*p(\s+|$)', ' status post ')
        x = x.str.replace(r'(^|\s)s\s*/\s*p\:(\s+|$)', ' status post ')  
        x = x.str.replace(r'follow-up', ' follow up ')
        x = x.str.replace(r'(^|\s)f\s*/\s*u(\s+|$)', ' follow up ')
        x = x.str.replace(r'(^|\s)w\s*/\s*u(\s+|$)', ' workup ')
        x = x.str.replace(r'(^|\s)w\s*/\s*u\:(\s+|$)', ' workup ')
        
        x = x.str.replace(r'(^|\s)h\s*/\s*c(\s+|$)', ' blood culture ')
        x = x.str.replace(r'(^|\s)hc(\s+|$)', ' blood culture ') 
        x = x.str.replace(r'(^|\s)hc\:(\s+|$)', ' blood culture ') 
        
        x = x.str.replace(r'(^|\s)u\s*/\s*c(\s+|$)', ' urine culture ')
        x = x.str.replace(r'(^|\s)u\s*c\:(\s+|$)', ' urine culture ')
        x = x.str.replace(r'(^|\s)u\s*/\s*c\:(\s+|$)', ' urine culture ')
        x = x.str.replace(r'(^|\s)u\s*/\s*c\-\ng(\s+|$)', ' urine culture no growth ')
        x = x.str.replace(r'(^|\s)uc-ng(\s+|$)', ' urine culture no growth ')
        
        x = x.str.replace(r'(^|\s)afb\-ve(\s+|$)', ' acid fast bacillus negative ')
        x = x.str.replace(r'(^|\s)afb\-neg(\s+|$)', ' acid fast bacillus negative ')
        x = x.str.replace(r'(^|\s)g\s*/\s*s(\s+|$)', ' gram stain ')
        x = x.str.replace(r'(^|\s)g\s*/\s*s\:(\s+|$)', ' gram stain ')
        x = x.str.replace(r'(^|\s)c\s*/\s*s(\s+|$)',' culture and sensitivity ')
        x = x.str.replace(r'(^|\s)c/s-ng(\s+|$)', ' culture and sensitivity no growth ')
        
        x = x.str.replace(r'(^|\s)dlp:(\s+|$)', ' dyslipidemia ') 
        x = x.str.replace(r'(^|\s)et-t:(\s+|$)', ' endotracheal tube ') 
        x = x.str.replace(r'(^|\s)ett:(\s+|$)', ' endotracheal tube ') 
        x = x.str.replace(r'(^|\s)ett>(\s+|$)', ' endotracheal tube ') 
        x = x.str.replace(r'(^|\s)ekg>(\s+|$)', ' electrocardiogram ') 
        x = x.str.replace(r'(^|\s)nk-t:(\s+|$)', ' nkt ') 
        x = x.str.replace(r'(^|\s)nk/t:(\s+|$)', ' nkt ') 
        x = x.str.replace(r'(^|\s)ae1/ae3:(\s+|$)', ' pan cytokeratin ') 
        
        x = x.str.replace(r'(^|\s)folfox-4(\s+|$)', ' folfox4 ')
        x = x.str.replace(r'(^|\s)cis/gem(\s+|$)', ' cisplatin and gemcitabine ') 
        x = x.str.replace(r'(^|\s)cisplatin/etoposide(\s+|$)', ' cisplatin and etoposide ') 
        x = x.str.replace(r'(^|\s)cisplatin/5-fu(\s+|$)', ' cisplatin and fluorouracil ') 
        x = x.str.replace(r'(^|\s)cisplatin/5fu(\s+|$)', ' cisplatin and fluorouracil ') 
        x = x.str.replace(r'(^|\s)cis/5-fu(\s+|$)', ' cisplatin and fluorouracil ') 
        x = x.str.replace(r'(^|\s)carboplatin/5-fu(\s+|$)', ' carboplatin and fluorouracil ') 
        x = x.str.replace(r'(^|\s)carboplatin/5fu(\s+|$)', ' carboplatin and fluorouracil ') 
        x = x.str.replace(r'(^|\s)5-fu/lv(\s+|$)', ' fluorouracil and leucovorin ') 
        x = x.str.replace(r'(^|\s)lv/5fu(\s+|$)', ' fluorouracil and leucovorin ')
        x = x.str.replace(r'(^|\s)5fu/rescuvolin(\s+|$)', ' fluorouracil and rescuvolin ')
        x = x.str.replace(r'(^|\s)5fu/rescuvolin>(\s+|$)', ' fluorouracil and rescuvolin ')
        
        # remove space at beginning and ending sentence
        x = x.str.replace(r'^\s+|\s+$', '')
        x = x.str.replace('\s+', ' ')  # space at any characters
        
        end = x.equals(x_old)
        n += 1
        print('round {}, duplication = {}'.format(n,end))
        if end == True:
            print('finished')
            break
        return x

X_all['text_en_pun'] = replace_punc(X_all['text_en_repunread'])
X_test['text_en_pun'] = replace_punc(X_test['text_en_repunread'])


X = pd.concat([X_all,X_test], axis = 0)

# export dataset
# X.to_csv(r'Dataset_final.csv', sep = '|', index = False)


