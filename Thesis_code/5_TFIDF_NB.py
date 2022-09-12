# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 22:11:27 2021
@author: Wanchana
"""
# =============================================================================
# specify path
# =============================================================================
import os
import sys

path = ''
os.chdir('')
sys.path.append(r'Thesis_code')
load_path = r'../final_dataset/nb_input/top_50/'
save_path = r'../matrix/tfidf_nb/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
# =============================================================================
# import dependency
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from utils import helper
import pickle
# =============================================================================
# Set seed and shuld divice
# =============================================================================
helper.set_seed(2021) # set seed
device = helper.get_device(0) # number indicate device position
# =============================================================================
# def set used
# =============================================================================
def to_labels(pos_probs, thresholds):
    return (pos_probs >= thresholds).astype('int')

def get_threshold(y_true, y_pred, y_list):
    # =============================================================================
    # # find thresholds
    # =============================================================================
    fpr, tpr, roc_auc  = {}, {}, {}
    thresholds = np.arange(0, 1, 0.001)
    thresholds_roc = {}
    thresholds_f1 = {}
    for j in range(len(y_list)):
        # AUC ROC optimal theshold
        fpr[j], tpr[j], thres_roc = roc_curve(y_true[:,j], y_pred[:,j])
        tpr[j][np.isnan(tpr[j])] = 0.5
        roc_auc[y_list[j]] = auc(fpr[j], tpr[j])
        # Youden’s J statistic --> sensitivity + specificity - 1 == tpr - fpr
        optimal_idx_roc = np.argmax(tpr[j] - fpr[j])
        optimal_thres_roc = thres_roc[optimal_idx_roc]
        thresholds_roc['{}'.format(y_list[j])] = optimal_thres_roc
        
        # AUC AP optimal theshold
        # using f1 as a representative of AP score
        f1_scores = [f1_score(y_true[:,j], to_labels(y_pred[:,j], t)) for t in thresholds]
        
        optimal_idx_f1 = np.argmax(f1_scores)
        optimal_thres_f1 = thresholds[optimal_idx_f1]
    
        thresholds_f1['{}'.format(y_list[j])] = optimal_thres_f1
    
    return thresholds_roc, thresholds_f1

def test(y_true, y_pred, y_list, thresholds):
    # =============================================================================
    # # evaluate metices
    # =============================================================================
    ## rank metrics p@k r@k
        # select top 5 and 3 positive classes by probability 
    top5 = list((-y_pred).argsort()[:,0:5])
    top3 = list((-y_pred).argsort()[:,0:3])
        # select positive grounded truth
    true_label = np.argwhere(y_true == 1)
    true_label = np.split(true_label[:,1], np.unique(true_label[:,0], return_index=True)[1][1:])
        # combine to dataframe and apply set
    top5 = pd.DataFrame([top5, true_label]).T
    top5 = top5.applymap(set)
    top3 = pd.DataFrame([top3, true_label]).T
    top3 = top3.applymap(set)
        # create intersetion of pred and true of positive classes
    top5['intersection'] = top5.apply(lambda x: x[0].intersection(x[1]), axis = 1)
    top3['intersection'] = top3.apply(lambda x: x[0].intersection(x[1]), axis = 1)
    
        # compute precision and recall at 5
    top5['pa5'] = top5.apply(lambda x: len(x['intersection']) / len(x[0]), axis = 1)
    top5['ra5'] = top5.apply(lambda x: len(x['intersection']) / len(x[1]), axis = 1)
    
    top3['pa3'] = top3.apply(lambda x: len(x['intersection']) / len(x[0]), axis = 1)
    top3['ra3'] = top3.apply(lambda x: len(x['intersection']) / len(x[1]), axis = 1)

    ## binary classification metrices
    results_y_prob = pd.DataFrame()
    results_y_pred = pd.DataFrame()
    PR_each, REC_each, F1_each = {}, {}, {}
    tn_all, fn_all, tp_all, fp_all = 0, 0, 0, 0
    cmat, fpr, tpr, roc_auc  = {}, {}, {}, {} 
    pr_plot, re_plot, ap_auc = {}, {}, {} 
    # find metrics for each label
    for j in range(len(y_list)):
        # AUC AP
        pr_plot[j], re_plot[j], _ = precision_recall_curve(y_true[:,j], y_pred[:,j])
        re_plot[j][np.isnan(re_plot[j])] = 0.5
        ap_auc[y_list[j]] = auc(re_plot[j], pr_plot[j])
        
        # AUC ROC optimal theshold
        fpr[j], tpr[j], thres_roc = roc_curve(y_true[:,j], y_pred[:,j])
        tpr[j][np.isnan(tpr[j])] = 0.5
        roc_auc[y_list[j]] = auc(fpr[j], tpr[j])
        
        # y probability
        results_y_prob[y_list[j]] = y_pred[:,j]
        # y predict
        results_y_pred[y_list[j]] = results_y_prob[y_list[j]].apply(lambda x: 1 if x >= thresholds[y_list[j]] else 0)
        
        # confusion metric
        cmat[j] = np.array([[0, 0], [0, 0]]).astype(np.int32)
        if all(y_true[:,j] == results_y_pred[y_list[j]]) == True: # if perfect prediction
            cmat[j][0][0] += sum(results_y_pred[y_list[j]] == 0) # increment by number of 0 trainues
            cmat[j][1][1] += sum(results_y_pred[y_list[j]] == 1) # increment by number of 1 trainues
        else:
            cmat[j] += confusion_matrix(y_true[:,j], results_y_pred[y_list[j]]) # else add cf trainues
        cmat[j] = cmat[j] + 1.0E-9 # add epsilon to avoud zero and nan
        tn, fn, tp, fp = cmat[j][0,0], cmat[j][1,0], cmat[j][1,1], cmat[j][0,1]
        tn_all += tn
        fn_all += fn
        tp_all += tp
        fp_all += fp 
        # metrices for each label
        PR = tp / (tp + fp)
        REC = tp / (tp + fn)
        F1 = 2 * (PR * REC) / (PR + REC)
        
        PR_each[y_list[j]] = PR
        REC_each[y_list[j]] = REC
        F1_each[y_list[j]] = F1
    # remove epsilon         
    for k, v in cmat.items():
        cmat[k] = v - 1.0E-9
    
    # overall metrices
    ROC_micro = roc_auc_score(y_true, y_pred, average='micro')
    ROC_macro = pd.Series(roc_auc[k] for k in roc_auc).mean()
    
    AP_micro = average_precision_score(y_true, y_pred, average='micro')
    AP_macro = pd.Series(ap_auc[k] for k in ap_auc).mean()
    
    PR_micro  = tp_all / (tp_all + fp_all)
    REC_micro = tp_all / (tp_all + fn_all)
    F1_micro  = 2 * (PR_micro * REC_micro) / (PR_micro + REC_micro)
    
    PR_macro  = sum(PR_each.values())  / len(PR_each)
    REC_macro = sum(REC_each.values()) / len(REC_each)
    F1_macro  = sum(F1_each.values())  / len(F1_each)
    
    # Record all statistics from this epoch.
    dataset_stats = {'Loss': '',
                     'ROC_micro': ROC_micro,
                     'ROC_macro': ROC_macro,
                     'AP_micro': AP_micro,
                     'AP_macro': AP_macro,
                     'PR_micro' : PR_micro,
                     'PR_macro' : PR_macro,
                     'REC_micro' : REC_micro,
                     'REC_macro' : REC_macro,
                     'F1_micro' : F1_micro,
                     'F1_macro' : F1_macro,
                     'Pa3' : top3['pa3'].mean(),
                     'Ra3' : top3['ra3'].mean(),
                     'Pa5' : top5['pa5'].mean(),
                     'Ra5' : top5['ra5'].mean()}
    
    best_metrice = pd.concat([pd.DataFrame.from_dict(PR_each,orient='index', columns = ['PR']),
                              pd.DataFrame.from_dict(REC_each,orient='index', columns = ['REC']),
                              pd.DataFrame.from_dict(F1_each,orient='index', columns = ['F1']),
                              pd.DataFrame.from_dict(roc_auc,orient='index', columns = ['ROC']),    
                              pd.DataFrame.from_dict(ap_auc,orient='index', columns = ['AP']),   
                             ], axis = 1).reset_index()
    best_metrice.columns = ['labels', 'PR', 'REC', 'F1', 'ROC', 'AP']
    return dataset_stats, best_metrice, cmat, results_y_prob, results_y_pred
# =============================================================================
# import dataset
# =============================================================================
# raw dataset
with open(load_path + r'X_train.pickle','rb') as handle:
    X_train = pickle.load(handle)
with open(load_path + r'X_test.pickle','rb') as handle:
    X_test = pickle.load(handle)


with open(load_path + r'y_train.pickle','rb') as handle:
    y_train = pickle.load(handle)
with open(load_path + r'y_test.pickle','rb') as handle:
    y_test = pickle.load(handle)
with open(load_path + r'y_list.pickle','rb') as handle:
    y_list = pickle.load(handle)


y_test = pd.DataFrame(y_test, columns=y_list)
X_test[X_test.an == '1914324']

a = y_test.iloc[200]
# =============================================================================
# create embedding vector
# =============================================================================
# instantiate the vectorizer object
tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english', min_df = 10, max_df = 0.8, ngram_range = (1,4))

# convert documents into a matrix
X_train_tf = tfidfvectorizer.fit_transform(X_train['text_en_pun'])
X_test_tf = tfidfvectorizer.transform(X_test['text_en_pun'])

# =============================================================================
# create model
# =============================================================================
# use nultinomial nb
clf = OneVsRestClassifier(MultinomialNB())

param_grid = {'estimator__alpha': np.linspace(0.5,1,5)}
gs = GridSearchCV(estimator = clf,
                       param_grid = param_grid,
                       scoring = 'roc_auc',
                       cv = 10,
                       n_jobs = -1)

gs = gs.fit(X_train_tf, y_train)
bestgs = gs.best_params_

y_train_pred = gs.predict_proba(X_train_tf)
y_test_pred = gs.predict_proba(X_test_tf)

# =============================================================================
# evaluate model
# =============================================================================
## get threshold
thresholds_roc, thresholds_f1 = get_threshold(y_train, y_train_pred, y_list)

## eval
train_stats_roc_thres, best_metrice_train_roc_thres, \
cmat_train_roc_thres, results_y_prob_train_roc_thres, \
results_y_pred_train_roc_thres = test(y_true = y_train,
                                     y_pred = y_train_pred,
                                     y_list = y_list,
                                     thresholds = thresholds_roc)

test_stats_roc_thres, best_metrice_test_roc_thres, \
cmat_test_roc_thres, results_y_prob_test_roc_thres, \
results_y_pred_test_roc_thres = test(y_true = y_test,
                                     y_pred = y_test_pred,
                                     y_list = y_list,
                                     thresholds = thresholds_roc)

train_stats_f1_thres, best_metrice_train_f1_thres, \
cmat_train_f1_thres, results_y_prob_train_f1_thres, \
results_y_pred_train_f1_thres = test(y_true = y_train,
                                     y_pred = y_train_pred,
                                     y_list = y_list,
                                     thresholds = thresholds_f1)

test_stats_f1_thres, best_metrice_test_f1_thres, \
cmat_test_f1_thres, results_y_prob_test_f1_thres, \
results_y_pred_test_f1_thres = test(y_true = y_test,
                                     y_pred = y_test_pred,
                                     y_list = y_list,
                                     thresholds = thresholds_f1)
# =============================================================================
# transfrom results
# =============================================================================
train_stats_roc_thres = pd.DataFrame.from_dict(data=train_stats_roc_thres,orient='index')
stats_train_roc_thres = train_stats_roc_thres.reset_index()

test_stats_roc_thres = pd.DataFrame.from_dict(data=test_stats_roc_thres,orient='index')
stats_test_roc_thres = test_stats_roc_thres.reset_index()


train_stats_f1_thres = pd.DataFrame.from_dict(data=train_stats_f1_thres ,orient='index')
stats_train_f1_thres = train_stats_f1_thres.reset_index()

test_stats_f1_thres = pd.DataFrame.from_dict(data=test_stats_f1_thres ,orient='index')
stats_test_f1_thres = test_stats_f1_thres.reset_index()


# =============================================================================
# save data
# =============================================================================

## thresholds of train mode
with open(save_path + 'thresholds_roc.pickle', 'wb') as f:
    pickle.dump(thresholds_roc, f)
with open(save_path + 'thresholds_f1.pickle', 'wb') as f:
    pickle.dump(thresholds_f1, f)
    
    
# overall metrices
stats_train_roc_thres.to_excel(save_path + 'overall_stats_train_roc_thres.xlsx', index = False)
stats_test_roc_thres.to_excel(save_path + 'overall_stats_test_roc_thres.xlsx', index = False)

stats_train_f1_thres.to_excel(save_path + 'overall_stats_train_f1_thres.xlsx', index = False)
stats_test_f1_thres.to_excel(save_path + 'overall_stats_test_f1_thres.xlsx', index = False)


# each class metrices
best_metrice_train_roc_thres.to_excel(save_path + 'each_class_stats_train_roc_thres.xlsx', index = False)
best_metrice_test_roc_thres.to_excel(save_path + 'each_class_stats_test_roc_thres.xlsx', index = False)

best_metrice_train_f1_thres.to_excel(save_path + 'each_class_stats_train_f1_thres.xlsx', index = False)
best_metrice_test_f1_thres.to_excel(save_path + 'each_class_stats_test_f1_thres.xlsx', index = False)


## prediction, probability and confusion metric
    # y probability
with open(save_path + 'results_prob_train_roc_thres.pickle', 'wb') as f:
    pickle.dump(results_y_prob_train_roc_thres, f)
with open(save_path + 'results_prob_test_roc_thres.pickle', 'wb') as f:
    pickle.dump(results_y_prob_test_roc_thres, f)
    
with open(save_path + 'results_prob_train_f1_thres.pickle', 'wb') as f:
    pickle.dump(results_y_prob_train_f1_thres, f)
with open(save_path + 'results_prob_test_f1_thres.pickle', 'wb') as f:
    pickle.dump(results_y_prob_test_f1_thres, f)
    
    
    # y predict
with open(save_path + 'results_pred_train_roc_thres.pickle', 'wb') as f:
    pickle.dump(results_y_pred_train_roc_thres, f)
with open(save_path + 'results_pred_test_roc_thres.pickle', 'wb') as f:
    pickle.dump(results_y_pred_test_roc_thres, f)

with open(save_path + 'results_pred_train_f1_thres.pickle', 'wb') as f:
    pickle.dump(results_y_pred_train_f1_thres, f)
with open(save_path + 'results_pred_test_f1_thres.pickle', 'wb') as f:
    pickle.dump(results_y_pred_test_f1_thres, f)


    # confustion metric
with open(save_path + 'cmat_train_roc_thres.pickle', 'wb') as f:
    pickle.dump(cmat_train_roc_thres, f)
with open(save_path + 'cmat_test_roc_thres.pickle', 'wb') as f:
    pickle.dump(cmat_test_roc_thres, f)
    
with open(save_path + 'cmat_train_f1_thres.pickle', 'wb') as f:
    pickle.dump(cmat_train_f1_thres, f)
with open(save_path + 'cmat_test_f1_thres.pickle', 'wb') as f:
    pickle.dump(cmat_test_f1_thres, f)
