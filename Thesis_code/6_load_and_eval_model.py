#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 13:24:59 2021

@author: wanchana
"""
# =============================================================================
# specify path
# =============================================================================
import os
import sys
path = ''
os.chdir(path)
load_path = ''
save_path = ''
sys.path.append(r'Thesis_code')
# =============================================================================
# import dependency
# =============================================================================
import pandas as pd
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import torch.optim as optim
import pickle
import joblib
from utils import model, engine, helper
# =============================================================================
# Set seed and shuld divice
# =============================================================================
helper.set_seed(2021) # set seed
device = helper.get_device(0) # number indicate device position
# =============================================================================
# Load data
# =============================================================================
with open(load_path + r'train_ids.pickle','rb') as handle:
    train_ids = pickle.load(handle)
with open(load_path + r'test_ids.pickle','rb') as handle:
    test_ids = pickle.load(handle)
with open(load_path + r'val_ids.pickle','rb') as handle:
    val_ids = pickle.load(handle)

with open(load_path + r'train_masks.pickle','rb') as handle:
    train_masks = pickle.load(handle)
with open(load_path + r'test_masks.pickle','rb') as handle:
    test_masks = pickle.load(handle)
with open(load_path + r'val_masks.pickle','rb') as handle:
    val_masks = pickle.load(handle)

with open(load_path + r'train_origin_ids.pickle','rb') as handle:
    train_origin_ids = pickle.load(handle)
with open(load_path + r'test_origin_ids.pickle','rb') as handle:
    test_origin_ids = pickle.load(handle)
with open(load_path + r'val_origin_ids.pickle','rb') as handle:
    val_origin_ids = pickle.load(handle)

with open(load_path + r'y_train.pickle','rb') as handle:
    y_train = pickle.load(handle)
with open(load_path + r'y_test.pickle','rb') as handle:
    y_test = pickle.load(handle)
with open(load_path + r'y_val.pickle','rb') as handle:
    y_val = pickle.load(handle)
with open(load_path + r'y_list.pickle','rb') as handle:
    y_list = pickle.load(handle)

# find max chunk & max len
max_chunk = max(max([len(train_origin_ids[i]) for i in range(len(train_origin_ids))]), 
                max([len(val_origin_ids[i]) for i in range(len(val_origin_ids))]),
                max([len(test_origin_ids[i]) for i in range(len(test_origin_ids))]))
max_len = train_ids.size(1)
# create class weight & loss function
counts_elements = np.sum(y_train, axis = 0)
classweight = (torch.tensor((len(y_train) - counts_elements) / counts_elements)).to(device)
classweight = classweight.to(torch.int32)
loss_fct = BCEWithLogitsLoss(classweight)
del counts_elements
# =============================================================================
# Model loading
# =============================================================================
# Load Models
    # load BERT
model_bert = model.BertEmb.from_pretrained(
    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    output_attentions = False,
    output_hidden_states = False)
model_bert.to(device)
# =============================================================================
# load study
# =============================================================================
study = joblib.load("")
# # show param
df_summary = pd.read_excel("")
# =============================================================================
# train model with the best params
# =============================================================================
# create dataset
train_dataset = TensorDataset(torch.tensor(np.array(list(train_origin_ids.keys()))), 
                              torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(np.array(list(test_origin_ids.keys()))), 
                              torch.tensor(y_test))
val_dataset = TensorDataset(torch.tensor(np.array(list(val_origin_ids.keys()))), 
                            torch.tensor(y_val))
# =============================================================================
# evaluate model
# =============================================================================
stats_train_roc_thres = pd.DataFrame()
stats_test_roc_thres = pd.DataFrame()
stats_val_roc_thres = pd.DataFrame()
stats_train_f1_thres = pd.DataFrame()
stats_test_f1_thres = pd.DataFrame()
stats_val_f1_thres = pd.DataFrame()

stats_train_callibrate_thres = pd.DataFrame()
stats_test_callibrate_thres = pd.DataFrame()
stats_val_callibrate_thres = pd.DataFrame()

### load thresholds
with open(save_path + 'thresholds_roc.pickle','rb') as t:
    thresholds_roc = pickle.load(t)
with open(save_path + 'thresholds_f1.pickle','rb') as t:
    thresholds_f1 = pickle.load(t)

with open(save_path + 'threshold_callibrate.pickle','rb') as t:
    thresholds_callibrate = pickle.load(t)
    
# select trials
# top_trials = df_summary.nsmallest(5, 'value').number.tolist()
# top_trials = top_trials + [0]


# select best trial
top_trials = df_summary.nsmallest(1, 'value').number.tolist()

# Display floats with two decimal places.
pd.set_option('precision', 4)
n = 0
for i in top_trials:
# losses
    n += 1
    loss_df = pd.read_excel(save_path + f'all_losses_{i}.xlsx')
    j = loss_df.val.idxmin()
    ###########################
    trial = df_summary[i:i+1]
    # create data loader
    train_dataloader = DataLoader(train_dataset,
                                  sampler = SequentialSampler(train_dataset),
                                  batch_size = trial['batch_size'].item())
    validation_dataloader = DataLoader(val_dataset,
                                       sampler = SequentialSampler(val_dataset),
                                       batch_size = trial['batch_size'].item())
    test_dataloader = DataLoader(test_dataset,
                                 sampler = SequentialSampler(test_dataset),
                                 batch_size = trial['batch_size'].item())
    
    # load model structure
    model_cnn = model.CNN(
                          embed_dim = 768, 
                          class_num = 50, 
                          kernel_num = trial['kernel_num'].item(), 
                          kernel_sizes = [2,3,4], 
                          dropout = trial['dropout'].item()
                          )
    model_cnn.to(device)
    
    # set optimizer structure
    param_optimizer = list(model_cnn.named_parameters())
    no_decay = ['bias']
    optimizer_grouped_parameters = [
                                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': trial['weight_decay'].item()},
                                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
                                    ]
    
    optimizer = getattr(optim, trial['optimizer'].item())(optimizer_grouped_parameters, lr=trial['lr'].item())

    # load the last checkpoint with the best model
    checkpoint = torch.load(r'model/bertcnn50_finetuned/ckp_bertcnn50_fine_tuned_trial_{}.pth.tar'.format(i))
    # import model weight
    model_cnn.load_state_dict(checkpoint['state_dict'])
    # import optimizer weight
    optimizer.load_state_dict(checkpoint['optimizer'])


    # roc threshold
    train_stats_roc_thres, best_metrice_train_roc_thres, \
    cmat_train_roc_thres, results_y_prob_train_roc_thres, \
    results_y_pred_train_roc_thres = engine.test(model = model_cnn,
                                                  model_bert = model_bert,
                                                  dataloader = train_dataloader,
                                                  device = device,
                                                  origin_ids = train_origin_ids,
                                                  used_ids = train_ids,
                                                  used_masks = train_masks,
                                                  max_chunk = max_chunk,
                                                  max_len = max_len,
                                                  y_list = y_list,
                                                  thresholds = thresholds_roc)
    val_stats_roc_thres, best_metrice_val_roc_thres, \
    cmat_val_roc_thres, results_y_prob_val_roc_thres, \
    results_y_pred_val_roc_thres = engine.test(model = model_cnn,
                                                model_bert = model_bert,
                                                dataloader = validation_dataloader,
                                                device = device,
                                                origin_ids = val_origin_ids,
                                                used_ids = val_ids,
                                                used_masks = val_masks,
                                                max_chunk = max_chunk,
                                                max_len = max_len,
                                                y_list = y_list,
                                                thresholds = thresholds_roc)
    test_stats_roc_thres, best_metrice_test_roc_thres, \
    cmat_test_roc_thres, results_y_prob_test_roc_thres, \
    results_y_pred_test_roc_thres = engine.test(model = model_cnn,
                                                model_bert = model_bert,
                                                dataloader = test_dataloader,
                                                device = device,
                                                origin_ids = test_origin_ids,
                                                used_ids = test_ids,
                                                used_masks = test_masks,
                                                max_chunk = max_chunk,
                                                max_len = max_len,
                                                y_list = y_list,
                                                thresholds = thresholds_roc)
        
    # f1
    train_stats_f1_thres, best_metrice_train_f1_thres, \
    cmat_train_f1_thres, results_y_prob_train_f1_thres, \
    results_y_pred_train_f1_thres = engine.test(model = model_cnn,
                                                  model_bert = model_bert,
                                                  dataloader = train_dataloader,
                                                  device = device,
                                                  origin_ids = train_origin_ids,
                                                  used_ids = train_ids,
                                                  used_masks = train_masks,
                                                  max_chunk = max_chunk,
                                                  max_len = max_len,
                                                  y_list = y_list,
                                                  thresholds = thresholds_f1)
    val_stats_f1_thres, best_metrice_val_f1_thres, \
    cmat_val_f1_thres, results_y_prob_val_f1_thres, \
    results_y_pred_val_f1_thres = engine.test(model = model_cnn,
                                                model_bert = model_bert,
                                                dataloader = validation_dataloader,
                                                device = device,
                                                origin_ids = val_origin_ids,
                                                used_ids = val_ids,
                                                used_masks = val_masks,
                                                max_chunk = max_chunk,
                                                max_len = max_len,
                                                y_list = y_list,
                                                thresholds = thresholds_f1)
    test_stats_f1_thres, best_metrice_test_f1_thres, \
    cmat_test_f1_thres, results_y_prob_test_f1_thres, \
    results_y_pred_test_f1_thres = engine.test(model = model_cnn,
                                                model_bert = model_bert,
                                                dataloader = test_dataloader,
                                                device = device,
                                                origin_ids = test_origin_ids,
                                                used_ids = test_ids,
                                                used_masks = test_masks,
                                                max_chunk = max_chunk,
                                                max_len = max_len,
                                                y_list = y_list,
                                                thresholds = thresholds_f1)
    
    # calibrate
    train_stats_callibrate_thres, best_metrice_train_callibrate_thres, \
    cmat_train_callibrate_thres, results_y_prob_train_callibrate_thres, \
    results_y_pred_train_callibrate_thres = engine.test(model = model_cnn,
                                                 model_bert = model_bert,
                                                 dataloader = train_dataloader,
                                                 device = device,
                                                 origin_ids = train_origin_ids,
                                                 used_ids = train_ids,
                                                 used_masks = train_masks,
                                                 max_chunk = max_chunk,
                                                 max_len = max_len,
                                                 y_list = y_list,
                                                 thresholds = thresholds_callibrate)
    val_stats_callibrate_thres, best_metrice_val_callibrate_thres, \
    cmat_val_callibrate_thres, results_y_prob_val_callibrate_thres, \
    results_y_pred_val_callibrate_thres = engine.test(model = model_cnn,
                                                model_bert = model_bert,
                                                dataloader = validation_dataloader,
                                                device = device,
                                                origin_ids = val_origin_ids,
                                                used_ids = val_ids,
                                                used_masks = val_masks,
                                                max_chunk = max_chunk,
                                                max_len = max_len,
                                                y_list = y_list,
                                                thresholds = thresholds_callibrate)
    test_stats_callibrate_thres, best_metrice_test_callibrate_thres, \
    cmat_test_callibrate_thres, results_y_prob_test_callibrate_thres, \
    results_y_pred_test_callibrate_thres = engine.test(model = model_cnn,
                                                model_bert = model_bert,
                                                dataloader = test_dataloader,
                                                device = device,
                                                origin_ids = test_origin_ids,
                                                used_ids = test_ids,
                                                used_masks = test_masks,
                                                max_chunk = max_chunk,
                                                max_len = max_len,
                                                y_list = y_list,
                                                thresholds = thresholds_callibrate)
    # =============================================================================
    # transfrom results
    # =============================================================================
    train_stats_roc_thres = pd.DataFrame.from_dict(data=train_stats_roc_thres,orient='index')
    train_stats_roc_thres.columns = [f'model_{i+1}']
    train_stats_roc_thres.iloc[0] = loss_df.iloc[j].train
    stats_train_roc_thres = pd.concat([stats_train_roc_thres, train_stats_roc_thres], axis = 1)

    test_stats_roc_thres = pd.DataFrame.from_dict(data=test_stats_roc_thres,orient='index')
    test_stats_roc_thres.columns = [f'model_{i+1}']
    stats_test_roc_thres = pd.concat([stats_test_roc_thres, test_stats_roc_thres], axis = 1)
    
    val_stats_roc_thres = pd.DataFrame.from_dict(data=val_stats_roc_thres,orient='index')
    val_stats_roc_thres.columns = [f'model_{i+1}']
    val_stats_roc_thres.iloc[0] = loss_df.iloc[j].val
    stats_val_roc_thres = pd.concat([stats_val_roc_thres, val_stats_roc_thres], axis = 1)
    

    train_stats_f1_thres = pd.DataFrame.from_dict(data=train_stats_f1_thres,orient='index')
    train_stats_f1_thres.columns = [f'model_{i+1}']
    train_stats_f1_thres.iloc[0] = loss_df.iloc[j].train
    stats_train_f1_thres = pd.concat([stats_train_f1_thres, train_stats_f1_thres], axis = 1)
    
    test_stats_f1_thres = pd.DataFrame.from_dict(data=test_stats_f1_thres,orient='index')
    test_stats_f1_thres.columns = [f'model_{i+1}']
    stats_test_f1_thres = pd.concat([stats_test_f1_thres, test_stats_f1_thres], axis = 1)
    
    val_stats_f1_thres = pd.DataFrame.from_dict(data=val_stats_f1_thres,orient='index')
    val_stats_f1_thres.columns = [f'model_{i+1}']
    val_stats_f1_thres.iloc[0] = loss_df.iloc[j].val
    stats_val_f1_thres = pd.concat([stats_val_f1_thres, val_stats_f1_thres], axis = 1)
    
    train_stats_callibrate_thres = pd.DataFrame.from_dict(data=train_stats_callibrate_thres,orient='index')
    train_stats_callibrate_thres.columns = [f'model_{i+1}']
    train_stats_callibrate_thres.iloc[0] = loss_df.iloc[j].train
    stats_train_callibrate_thres = pd.concat([stats_train_callibrate_thres, train_stats_callibrate_thres], axis = 1)

    test_stats_callibrate_thres = pd.DataFrame.from_dict(data=test_stats_callibrate_thres,orient='index')
    test_stats_callibrate_thres.columns = [f'model_{i+1}']
    stats_test_callibrate_thres = pd.concat([stats_test_callibrate_thres, test_stats_callibrate_thres], axis = 1)
    
    val_stats_callibrate_thres = pd.DataFrame.from_dict(data=val_stats_callibrate_thres,orient='index')
    val_stats_callibrate_thres.columns = [f'model_{i+1}']
    val_stats_callibrate_thres.iloc[0] = loss_df.iloc[j].val
    stats_val_callibrate_thres = pd.concat([stats_val_callibrate_thres, val_stats_callibrate_thres], axis = 1)

    # =============================================================================
    # 
    # =============================================================================
# overall metrices
stats_train_callibrate_thres.to_excel(save_path + 'overall_stats_train_callibrate_thres.xlsx')
stats_val_callibrate_thres.to_excel(save_path + 'overall_stats_val_callibrate_thres.xlsx')
stats_test_callibrate_thres.to_excel(save_path + 'overall_stats_test_callibrate_thres.xlsx')



# each class metrices
best_metrice_train_callibrate_thres.to_excel(save_path + 'each_class_stats_train_callibrate_thres.xlsx', index = False)
best_metrice_val_callibrate_thres.to_excel(save_path + 'each_class_stats_val_callibrate_thres.xlsx', index = False)
best_metrice_test_callibrate_thres.to_excel(save_path + 'each_class_stats_test_callibrate_thres.xlsx', index = False)



## prediction, probability and confusion metric
    # y probability
with open(save_path + 'results_prob_train_callibrate_thres.pickle', 'wb') as f:
    pickle.dump(results_y_prob_train_callibrate_thres, f)
with open(save_path + 'results_prob_val_callibrate_thres.pickle', 'wb') as f:
    pickle.dump(results_y_prob_val_callibrate_thres, f)
with open(save_path + 'results_prob_test_callibrate_thres.pickle', 'wb') as f:
    pickle.dump(results_y_prob_test_callibrate_thres, f)
    

    
    # y predict
with open(save_path + 'results_pred_train_callibrate_thres.pickle', 'wb') as f:
    pickle.dump(results_y_pred_train_callibrate_thres, f)
with open(save_path + 'results_pred_val_callibrate_thres.pickle', 'wb') as f:
    pickle.dump(results_y_pred_val_callibrate_thres, f)
with open(save_path + 'results_pred_test_callibrate_thres.pickle', 'wb') as f:
    pickle.dump(results_y_pred_test_callibrate_thres, f)


    # confustion metric
with open(save_path + 'cmat_train_callibrate_thres.pickle', 'wb') as f:
    pickle.dump(cmat_train_callibrate_thres, f)
with open(save_path + 'cmat_val_callibrate_thres.pickle', 'wb') as f:
    pickle.dump(cmat_val_callibrate_thres, f)
with open(save_path + 'cmat_test_callibrate_thres.pickle', 'wb') as f:
    pickle.dump(cmat_test_callibrate_thres, f)
    
    