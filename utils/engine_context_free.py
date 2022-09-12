#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:36:30 2021

@author: wanchana
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix, f1_score

# =============================================================================
# dataloader = train_dataloader
# origin_ids= train_origin_ids
# used_ids=train_ids
# used_masks=train_masks
# model = model_cnn
# criterion = loss_fct
# =============================================================================

def train(model, dataloader, optimizer, criterion, device, used_ids, y_list):
    device = device
    y_pred, y_true = [], []
    train_loss = 0
    model.train()
    ## train main model
    for step, batch in enumerate(dataloader):
        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(torch.float32).to(device) # labels (n = batch_size)


        # reset gradient
        model.zero_grad()
        # enable gradient back from the previous torch.no_grad()
        # run model
        logits = model(b_input_ids)
        # calculate loss
        loss = criterion(logits.view(-1, b_labels.size(1)),
                         b_labels.view(-1, b_labels.size(1)))
        # back propagation
        loss.backward()
        # clip the gradient to prevent gradient explode
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update wieght
        optimizer.step() 
        # sum loss for each batch
        train_loss += loss.item()
        # Move logits and labels to CPU to effciently compute metric
        y_pred.append(torch.sigmoid(logits).detach().cpu().numpy())
        y_true.append(b_labels.to('cpu').numpy())
    # average loss from each batch
    avg_train_loss = train_loss / len(dataloader)
    # stack list of batch of y to one numpy array of y of all dataset
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    # evaluate crude metices
    ROC = roc_auc_score(y_true, y_pred, average='micro')
    AP = average_precision_score(y_true, y_pred, average='micro')
    # Record all statistics from this epoch.
    dataset_stats = {'Loss': avg_train_loss,
                     'ROC': ROC,
                     'AP': AP}
    return avg_train_loss, dataset_stats, ROC, AP


def evaluate(model, dataloader, optimizer, criterion, device, used_ids, y_list):
    device = device
    model.eval()
    y_pred, y_true = [], []
    eval_loss = 0
    # evaluation mode need to turn on model.eval() and torch.no_grad()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(torch.float32).to(device) # labels (n = batch_size)
            # reset gradient for main model
            model.zero_grad()
            # run model
            logits = model(b_input_ids)
            # calculate loss
            loss = criterion(logits.view(-1, b_labels.size(1)), 
                             b_labels.view(-1, b_labels.size(1)))      
            # sum loss for each batch
            eval_loss += loss.item()
            # Move logits and labels to CPU to effciently compute metric
            y_pred.append(torch.sigmoid(logits).detach().cpu().numpy())
            y_true.append(b_labels.to('cpu').numpy())
        # average loss from each batch
        avg_eval_loss = eval_loss / len(dataloader)
        # stack list of batch of y to one numpy array of y of all dataset
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        # evaluate crude metices
        ROC = roc_auc_score(y_true, y_pred, average='micro')
        AP = average_precision_score(y_true, y_pred, average='micro')
        # Record all statistics from this epoch.
        dataset_stats = {'Loss': avg_eval_loss,
                         'ROC': ROC,
                         'AP': AP}
    return avg_eval_loss, dataset_stats, ROC, AP


def test(model, dataloader, device, used_ids, y_list, thresholds):
    device = device
    model.eval()
    y_pred, y_true = [], []
    # evaluation mode need to turn on model.eval() and torch.no_grad()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(torch.float32).to(device) # labels (n = batch_size)
            # reset gradient for main model
            model.zero_grad()
            # run model
            logits = model(b_input_ids)
            # Move logits and labels to CPU to effciently compute metric
            y_pred.append(torch.sigmoid(logits).detach().cpu().numpy())
            y_true.append(b_labels.to('cpu').numpy())
            
        # stack list of batch of y to one numpy array of y of all dataset
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
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


def get_threshold(model, dataloader, device, used_ids, y_list):
    device = device
    model.eval()
    y_pred, y_true = [], []
    # evaluation mode need to turn on model.eval() and torch.no_grad()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(torch.float32).to(device)  # labels (n = batch_size)
            # reset gradient for main model
            model.zero_grad()
            # run model
            logits = model(b_input_ids)
            # Move logits and labels to CPU to effciently compute metric
            y_pred.append(torch.sigmoid(logits).detach().cpu().numpy())
            y_true.append(b_labels.to('cpu').numpy())
            
        # stack list of batch of y to one numpy array of y of all dataset
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
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


def to_labels(pos_probs, thresholds):
    return (pos_probs >= thresholds).astype('int')