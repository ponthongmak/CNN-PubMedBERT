#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:15:53 2021

@author: wanchana
"""
# =============================================================================
# specify path
# ============================================================================
import os 
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import multiprocessing as mp

sys.path.append(r'Thesis_code')
path = ''
os.chdir(path)
load_path = ''
save_path = ''

# =============================================================================
# 
# =============================================================================
def find_LR(confusion, label_list, stat):
    labels = []
    prs,recs,specs,f1s, = [],[],[],[]
    lr_ps,lr_ns,odds,youden = [],[],[],[]
    for idx, cm in confusion.items():
        label = label_list[idx]
        tp, tn, fp, fn = cm[1,1], cm[0,0], cm[0,1], cm[1,0]

        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        fnr = fn / (fn + tp)
        tnr = tn / (fp + tn)
        
        pr = tp / (tp + fp)
        rec = tp / (tp + fn)
        spec = tn / (tn + fp)

        f1 = 2 * (pr * rec) / (pr + rec)
        
        lr_p = tpr / fpr
        lr_n = fnr / tnr
        odd = (tp + tn) / (fp + fn)
        uden = tpr - fpr
        
        labels.append(label)
        prs.append(pr)
        recs.append(rec)
        specs.append(spec)
        f1s.append(f1)
        lr_ps.append(lr_p)
        lr_ns.append(lr_n)
        odds.append(odd)
        youden.append(uden)
    
    data_dict = {'labels' : labels,'PR' : prs,'REC' : recs,'SPEC' : specs,
                 'F1' : f1s, 'Youden' : youden, 'OR' : odds, 'lr_pos' : lr_ps, 'lr_neg' : lr_ns}
    
    df_lr = pd.DataFrame(data_dict)
    df_lr = df_lr.merge(stat[['labels', 'AP', 'ROC']], how = 'right', on = 'labels')
    
    return df_lr

with open(load_path + r'y_list.pickle','rb') as handle:
    label_list = pickle.load(handle)
with open(save_path + 'cmat_train_roc_thres.pickle','rb') as t:
    cm_roc_train = pickle.load(t)
with open(save_path + 'cmat_test_roc_thres.pickle','rb') as t:
    cm_roc_test = pickle.load(t)
    
stat_train = pd.read_excel(save_path + 'each_class_stats_train_roc_thres.xlsx')
stat_test = pd.read_excel(save_path + 'each_class_stats_test_roc_thres.xlsx')
# stats = best_metrice_train_roc_thres.copy()

stat_train = find_LR(cm_roc_train, label_list, stat_train)
stat_test = find_LR(cm_roc_test, label_list, stat_test)

stat_train_err = stat_train[stat_train.REC < .95]
# =============================================================================
# if LR of positive less than 10 --> calibrate model
# =============================================================================
with open(r'y_train.pickle','rb') as handle:
    y_true = pickle.load(handle)

with open(r'results_prob_train_roc_thres.pickle','rb') as handle:
    y_prob = pickle.load(handle).to_numpy()

y_list = stat_train_err.labels.tolist()
    
y_true = pd.DataFrame(y_true, columns=label_list)
y_prob = pd.DataFrame(y_prob, columns=label_list)

y_true = y_true[y_list]
y_prob = y_prob[y_list]

def to_labels(pos_probs, thresholds):
    return (pos_probs >= thresholds).astype('int')


thresholds = np.arange(0, 1, 5e-06)


def threshold(y_list):
    dict_cm_list = {}
    # AUC ROC optimal theshold
    cm_list = [confusion_matrix(y_true.loc[:,y_list], to_labels(y_prob.loc[:,y_list], t)) for t in tqdm(thresholds)]
    dict_cm_list[y_list] = cm_list
    return dict_cm_list
       

pool_obj = mp.Pool(50)
dict_cm_list = pool_obj.map(threshold, y_list)

new_dic = {}
for dict_x in  dict_cm_list:
    for k,v in dict_x.items():
        new_dic[k] = v


cmat = {}
thresholds_calibrate = {}
data_dict = {}

for label, cm in new_dic.items():
    eval_data = pd.DataFrame()
    labels = []
    lr_ps, lr_ns, thres, udens = [],[],[],[]
    prs, recs, specs, f1s = [],[],[],[]
    cmat_temp = {}
    
    labels.append(label)
    
    for i,(th, cm) in tqdm(enumerate(zip(thresholds, cm))):
        tp, tn, fp, fn = cm[1,1], cm[0,0], cm[0,1], cm[1,0]

        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        fnr = fn / (fn + tp)
        tnr = tn / (fp + tn)
        
        lr_p = tpr / fpr
        lr_n = fnr / tnr
        uden = tpr - fpr
        
        pr = tp / (tp + fp)
        spec = tn / (tn + fp)
        rec = tpr.copy()
        f1 = 2 * (pr * rec) / (pr + rec)
        
        
        cmat_temp[i] = cm
        
        thres.append(th)
        lr_ps.append(lr_p)
        lr_ns.append(lr_n)
        udens.append(uden)
        
        prs.append(pr)
        recs.append(rec)
        specs.append(spec)
        f1s.append(f1)
        

    eval_data = pd.DataFrame(list(zip(thres,lr_ps,lr_ns,prs,recs,specs,f1s,udens)), 
                             columns = ['thres','lr_pos','lr_neg','pr','rec','spec','f1','uden'])
    eval_data = eval_data[(eval_data.rec > 0) & 
                          (eval_data.spec > 0) & 
                          (eval_data.lr_pos != np.inf) & 
                          (eval_data.lr_neg != np.inf)]
    
    eval_data = eval_data[(eval_data.rec > eval_data.pr)]
    
    eval_data['factor'] = ((0.6*eval_data.rec) + (0.4*eval_data.spec) + (0.0*eval_data.pr))
    

    thresholds_calibrate[label] = eval_data.loc[eval_data.factor.idxmax()]['thres']
    cmat[label] = cmat_temp.get(eval_data.factor.idxmax())
    
    
    data_dict[label] = eval_data.loc[eval_data.factor.idxmax()]
    
df_lr = pd.DataFrame(data_dict)


a = df_lr.mean(axis=1)

# =============================================================================
# 
# =============================================================================
x = df_lr.T
x_1 = x[x.rec >= .95]
y_list_2 = x[x.rec < .95].index.tolist()

y_true = y_true[y_list_2]
y_prob = y_prob[y_list_2]

dict_cm_list = pool_obj.map(threshold, y_list_2)

new_dic = {}
for dict_x in  dict_cm_list:
    for k,v in dict_x.items():
        new_dic[k] = v

cmat = {}
thresholds_calibrate = {}
data_dict = {}

for label, cm in new_dic.items():
    eval_data = pd.DataFrame()
    labels = []
    lr_ps, lr_ns, thres, udens = [],[],[],[]
    prs, recs, specs, f1s = [],[],[],[]
    cmat_temp = {}
    
    labels.append(label)
    
    for i,(th, cm) in tqdm(enumerate(zip(thresholds, cm))):
        tp, tn, fp, fn = cm[1,1], cm[0,0], cm[0,1], cm[1,0]

        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        fnr = fn / (fn + tp)
        tnr = tn / (fp + tn)
        
        lr_p = tpr / fpr
        lr_n = fnr / tnr
        uden = tpr - fpr
        
        pr = tp / (tp + fp)
        spec = tn / (tn + fp)
        rec = tpr.copy()
        f1 = 2 * (pr * rec) / (pr + rec)
        
        
        cmat_temp[i] = cm
        
        thres.append(th)
        lr_ps.append(lr_p)
        lr_ns.append(lr_n)
        udens.append(uden)
        
        prs.append(pr)
        recs.append(rec)
        specs.append(spec)
        f1s.append(f1)
        

    eval_data = pd.DataFrame(list(zip(thres,lr_ps,lr_ns,prs,recs,specs,f1s,udens)), 
                             columns = ['thres','lr_pos','lr_neg','pr','rec','spec','f1','uden'])
    eval_data = eval_data[(eval_data.rec > 0) & 
                          (eval_data.spec > 0) & 
                          (eval_data.lr_pos != np.inf) & 
                          (eval_data.lr_neg != np.inf)]
    
    eval_data = eval_data[(eval_data.rec > eval_data.pr)]
    
    eval_data['factor'] = ((0.85*eval_data.rec) + (0.15*eval_data.spec) + (0.0*eval_data.pr))
    

    thresholds_calibrate[label] = eval_data.loc[eval_data.factor.idxmax()]['thres']
    cmat[label] = cmat_temp.get(eval_data.factor.idxmax())
    
    
    data_dict[label] = eval_data.loc[eval_data.factor.idxmax()]
    
df_lr_2 = pd.DataFrame(data_dict)


x_2 = df_lr_2.T

x = x_1.append(x_2)

            
stats = pd.read_excel(save_path + 'each_class_stats_train_roc_thres.xlsx')
with open(save_path + 'thresholds_roc.pickle','rb') as t:
    thresholds_roc = pickle.load(t)            


old_matrice = pd.read_excel(save_path + 'each_class_stats_test_roc_thres.xlsx')

## class with recall under 0.9
x = x[x.index.isin(y_list)][['thres']]

x = dict(zip(x.index,x.thres))

with open(save_path + 'thresholds_roc.pickle','rb') as t:
    thresholds_roc = pickle.load(t)
    
threshold_callibrate = thresholds_roc.copy()
threshold_callibrate.update(x)



with open(save_path + 'threshold_callibrate.pickle', 'wb') as f:
    pickle.dump(threshold_callibrate, f)




best_metrice_test_roc_thres = find_LR(cmat_test_roc_thres, label_list, best_metrice_test_roc_thres)
