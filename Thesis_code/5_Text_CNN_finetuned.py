# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:31:29 2021
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
load_path = r'../final_dataset/glove_input/top_50/'
save_path = r../matrix/text_cnn_fine_tuned/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(r'/model/textcnn50_finetuned/'):
    os.makedirs(r'/model/textcnn50_finetuned/')
if not os.path.exists(r'/data/trial/50trial_text/'):
    os.makedirs(r'/data/trial/50trial_text/')
# =============================================================================
# import dependency
# =============================================================================
import time
import pandas as pd
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch.optim as optim
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import pickle
import joblib
from utils import model, engine_context_free, helper, text2vec
import nltk
from nltk.tokenize import word_tokenize
# =============================================================================
# Set seed and shuld divice
# =============================================================================
helper.set_seed(2021) # set seed
device = helper.get_device(0) # number indicate device position
# =============================================================================
# Load data
# =============================================================================
with open(load_path + r'X_train.pickle','rb') as handle:
    train_data = pickle.load(handle)
with open(load_path + r'X_test.pickle','rb') as handle:
    test_data = pickle.load(handle)
with open(load_path + r'X_val.pickle','rb') as handle:
    val_data = pickle.load(handle)

with open(load_path + r'y_train.pickle','rb') as handle:
    y_train = pickle.load(handle)
with open(load_path + r'y_test.pickle','rb') as handle:
    y_test = pickle.load(handle)
with open(load_path + r'y_val.pickle','rb') as handle:
    y_val = pickle.load(handle)
with open(load_path + r'y_list.pickle','rb') as handle:
    y_list = pickle.load(handle)

# # for test
# train_data = train_data[:50]
# test_data = test_data[:50]
# val_data = val_data[:50]
# y_train = y_train[:50]
# y_test = y_test[:50]
# y_val = y_val[:50]
# =============================================================================
# build word2index
# =============================================================================
# find max_len
max_len = max(train_data['text_en_pun'].str.split().str.len().max(),
              test_data['text_en_pun'].str.split().str.len().max(),
              val_data['text_en_pun'].str.split().str.len().max()
              )

def text_tokenization(data):
  # Tokenizes each sentence by implementing the nltk tool
  data['token'] = [word_tokenize(x) for x in data['text_en_pun']]

text_tokenization(train_data)
text_tokenization(test_data)
text_tokenization(val_data)

# Builds the vocabulary
all_data = train_data.append(val_data)
vocab = []
fdist = nltk.FreqDist()
for sentence in all_data['token']:
    for word in sentence:
        fdist[word] += 1
common_words = fdist.most_common()
for word in common_words:
    vocab.append(word[0])
vocab = np.insert(vocab, 0, '<pad>')
vocab = np.insert(vocab, 1, '<unk>')

# word2ids
train_unk, train_ids = text2vec.text2ids(train_data, vocab, 'text_en_pun')
test_unk, test_ids = text2vec.text2ids(test_data, vocab, 'text_en_pun')
val_unk, val_ids = text2vec.text2ids(val_data, vocab, 'text_en_pun')

# get padded ids
train_ids = text2vec.pad_ids(train_ids, max_len)
test_ids = text2vec.pad_ids(test_ids, max_len)
val_ids = text2vec.pad_ids(val_ids, max_len)

# =============================================================================
# class weight and loss function
# =============================================================================
counts_elements = np.sum(y_train, axis = 0)
classweight = (torch.tensor((len(y_train) - counts_elements) / counts_elements)).to(device)
classweight = classweight.to(torch.int32)
loss_fct = BCEWithLogitsLoss(classweight)
del counts_elements
# =============================================================================
# fine tune model
# =============================================================================
# save checkpoint
def save_checkpoint(state, filename='filename'):
    torch.save(state, filename)
    
def get_dataset(trial):
    batch_size = trial.suggest_int(name="batch_size", low=16, high=64, step=16)
    
    train_dataset = TensorDataset(torch.tensor(train_ids), 
                                  torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(val_ids), 
                                torch.tensor(y_val))

    train_dataloader = DataLoader(train_dataset,
                                  sampler = RandomSampler(train_dataset),
                                  batch_size = batch_size)
    validation_dataloader = DataLoader(val_dataset,
                                        sampler = SequentialSampler(val_dataset),
                                        batch_size = batch_size)
    return train_dataloader, validation_dataloader

def objective(trial):
    # load data
    train_dataloader, validation_dataloader = get_dataset(trial)
    # load model
    class config:
        def __init__(self):
            config.num_words = len(vocab) # number of embeddings
            config.class_num = len(y_list) # n of classes
            config.kernel_num = trial.suggest_int(name="kernel_num", low=100, high=500, step=100) # n of kernels
            config.kernel_sizes = [2,3,4] # kernel size
            config.embed_dim = 300 # embed dimension
            config.dropout = trial.suggest_float(name="dropout", low=0.4, high=0.9) # dropout value
            config.padding_idx = 0 # padding token index
            return None

    
    config = config()
    model_cnn = model.TextCNN(config)
    model_cnn.to(device)
    
    # optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ['AdamW', 'RMSprop', 'SGD'])
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    weight_decay =  trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    
    param_optimizer = list(model_cnn.named_parameters())
    no_decay = ['bias']
    optimizer_grouped_parameters = [
                                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': weight_decay},
                                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
                                    ]
    
    optimizer = getattr(optim, optimizer_name)(optimizer_grouped_parameters, lr=lr)
    # =============================================================================
    # Train the models
    # =============================================================================
    best_loss = np.inf
    patience = 10
    early_stopping_counter = 0
    train_loss, val_loss = [], []
    train_stats, val_stats = [], []
    
    for epoch in range(epochs):
        print("")
        print(f'==== Trial {trial.number+1} / {n_trials} ==== Epoch {epoch + 1} / {epochs} ====')
        print('Model Training...')
        start_time = time.time()
        avg_train_loss, t_stats, train_ROC, \
        train_AP = engine_context_free.train(model = model_cnn,
                                             dataloader = train_dataloader,
                                             optimizer = optimizer,
                                             criterion = loss_fct,
                                             device = device,
                                             used_ids = train_ids,
                                             y_list = y_list)
        train_loss.append(avg_train_loss)
        train_stats.append(t_stats)
        print("")
        print("Model Validation...")
        avg_val_loss, v_stats, val_ROC, \
        val_AP = engine_context_free.evaluate(model = model_cnn,
                                              dataloader = validation_dataloader,
                                              optimizer = optimizer,
                                              criterion = loss_fct,
                                              device = device,
                                              used_ids = val_ids,
                                              y_list = y_list)
        val_loss.append(avg_val_loss)
        val_stats.append(v_stats)
        # print results for each epoch
        print('')
        print(f"train_loss = {avg_train_loss:.6f}  train_ROC = {train_ROC:.4f}  train_AP = {train_AP:.4f}")
        print(f" val _loss = {avg_val_loss:.6f}    val_ROC = {val_ROC:.4f}    val_AP = {val_AP:.4f}")
        print('')
        print(f'Training time {helper.format_time(time.time() - start_time)}')
        
        # set early stopping and save best model
        if avg_val_loss < best_loss:
            # reset counter
            early_stopping_counter = 0
            print('')
            print(f'Validation loss decreased ({best_loss:.6f} --> {avg_val_loss:.6f}). Saving temp model...')
            best_loss = avg_val_loss
            save_checkpoint({'trial': trial.number,
                             'epoch': epoch + 1,
                             'state_dict': model_cnn.state_dict(),
                             'optimizer' : optimizer.state_dict(),
                             }, filename = r'model/textcnn50_finetuned/ckp_textcnn50_fine_tuned_trial_{}.pth.tar'.format(trial.number))
        else:
            early_stopping_counter += 1
            print('')
            print(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
        if early_stopping_counter >= patience:
            break
        # save trial report
        trial.report(avg_val_loss, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    # save losses
    all_losses = pd.DataFrame({'train' :train_loss, 'val' :val_loss})
    all_losses.to_excel(save_path + f'all_losses_{trial.number}.xlsx', index = False)
    # save stat
    with open(save_path + f'train_stat_trial_{trial.number}.pickle', 'wb') as f:
        pickle.dump(train_stats, f)
    with open(save_path + f'val_stat_trial_{trial.number}.pickle', 'wb') as f:
        pickle.dump(val_stats, f)
    # save current trial
    joblib.dump(study, "trial/50trial_text/study.pkl") 
    return avg_val_loss
# =============================================================================
# start fine tuned model
# =============================================================================
# optuna.delete_study(study_name="50_sdx_textcnn", storage="sqlite:///50_sdx_textcnn.db")
# study = joblib.load("trial/study.pkl")
epochs = 1000
n_trials = 50
if __name__ == "__main__":
    sampler = TPESampler(seed=2021)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name='50_sdx_textcnn',
                                storage="sqlite:///50_sdx_textcnn.db", load_if_exists=True)
    
    study.optimize(objective, n_trials=n_trials)
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print('')
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    
    print(f"Best trial: number {study.best_trial._number}")
    print("  Loss: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_params.items():
        print("    {}: {}".format(key, value))

# show param
df_summary = study.trials_dataframe()
# check param importance
optuna.importance.get_param_importances(study)
# get hyper param from best trial
trial = study.best_trial

# # resume study and optimize if error
# study1 = optuna.load_study(study_name="50_sdx_textcnn", sampler=sampler,
#                           storage="sqlite:///50_sdx_textcnn.db")
# df_summary1 = study1.trials_dataframe()

# study.optimize(objective, n_trials=n_trials)
# =============================================================================
# print loss and model performance
# =============================================================================
# Display floats with two decimal places.
pd.set_option('precision', 4)

    # losses
loss_df = pd.read_excel(save_path + f'all_losses_{trial.number}.xlsx')

plt.plot(loss_df['train'], label="train loss", color = 'tab:blue')
plt.plot(loss_df['val'], label="validation loss", color = 'tab:orange')
# plt.ylim(0,3.5)
plt.legend()
plt.title("The comparison between Train Loss and validation loss")
plt.savefig(save_path + 'The comparison between Train Loss and validation loss.png')
plt.show()

    # performance
with open(save_path + f'train_stat_trial_{trial.number}.pickle','rb') as f:
    train_stats = pickle.load(f)
with open(save_path + f'val_stat_trial_{trial.number}.pickle','rb') as handle:
    val_stats = pickle.load(handle)
    
train_stats = pd.DataFrame(data=train_stats)
val_stats = pd.DataFrame(data=val_stats)
plt.plot(train_stats['ROC'], label="Train_ROC", color = 'tab:blue')
plt.plot(val_stats['ROC'], label="Val_ROC", color = 'tab:orange')
plt.plot(train_stats['AP'], label="Train_AP", color = 'darkblue')
plt.plot(val_stats['AP'], label="Val_AP", color = 'maroon')
plt.ylim(0, 1)
plt.legend()
plt.title("The comparison of ROC and AP between train and validation set")
plt.savefig(save_path + 'The comparison of ROC and AP between train and validation set.png')
plt.show()
# =============================================================================
# train model with the best params
# =============================================================================
# create dataset
train_dataset = TensorDataset(torch.tensor(train_ids), 
                              torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(test_ids), 
                              torch.tensor(y_test))
val_dataset = TensorDataset(torch.tensor(val_ids), 
                            torch.tensor(y_val))

# create data loader
train_dataloader = DataLoader(train_dataset,
                              sampler = RandomSampler(train_dataset),
                              batch_size = trial.params.get('batch_size'))
validation_dataloader = DataLoader(val_dataset,
                                   sampler = SequentialSampler(val_dataset),
                                   batch_size = trial.params.get('batch_size'))
test_dataloader = DataLoader(test_dataset,
                             sampler = SequentialSampler(test_dataset),
                             batch_size = trial.params.get('batch_size'))

# load model structure
class config:
    def __init__(self):
        config.num_words = len(vocab) # number of embeddings
        config.class_num = len(y_list) # n of classes
        config.kernel_num = trial.params.get('kernel_num') # n of kernels
        config.kernel_sizes = [2,3,4] # kernel size
        config.embed_dim = 300 # embed dimension
        config.dropout = trial.params.get('dropout')  # dropout value
        config.padding_idx = 0  # padding token index
        return None

# load CNN
config = config()
model_cnn = model.TextCNN(config)
model_cnn.to(device)

# set optimizer structure
param_optimizer = list(model_cnn.named_parameters())
no_decay = ['bias']
optimizer_grouped_parameters = [
                                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': trial.params.get('weight_decay')},
                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
                                ]

optimizer = getattr(optim, trial.params.get('optimizer'))(optimizer_grouped_parameters, lr=trial.params.get('lr'))

# load the last checkpoint with the best model
checkpoint = torch.load(r'model/textcnn50_finetuned/ckp_textcnn50_fine_tuned_trial_{}.pth.tar'.format(trial.number))
# import model weight
model_cnn.load_state_dict(checkpoint['state_dict'])
# import optimizer weight
optimizer.load_state_dict(checkpoint['optimizer'])
# =============================================================================
# create threshold
# =============================================================================
thresholds_roc, thresholds_f1 = engine_context_free.get_threshold(model = model_cnn,
                                                                  dataloader = train_dataloader,
                                                                  device = device,
                                                                  used_ids = train_ids,
                                                                  y_list = y_list)
# =============================================================================
# evaluate model
# =============================================================================
## roc threshold
train_stats_roc_thres, best_metrice_train_roc_thres, \
cmat_train_roc_thres, results_y_prob_train_roc_thres, \
results_y_pred_train_roc_thres = engine_context_free.test(model = model_cnn,
                                                          dataloader = train_dataloader,
                                                          device = device,
                                                          used_ids = train_ids,
                                                          y_list = y_list,
                                                          thresholds = thresholds_roc)
val_stats_roc_thres, best_metrice_val_roc_thres, \
cmat_val_roc_thres, results_y_prob_val_roc_thres, \
results_y_pred_val_roc_thres = engine_context_free.test(model = model_cnn,
                                                        dataloader = validation_dataloader,
                                                        device = device,
                                                        used_ids = val_ids,
                                                        y_list = y_list,
                                                        thresholds = thresholds_roc)
test_stats_roc_thres, best_metrice_test_roc_thres, \
cmat_test_roc_thres, results_y_prob_test_roc_thres, \
results_y_pred_test_roc_thres = engine_context_free.test(model = model_cnn,
                                                         dataloader = test_dataloader,
                                                         device = device,
                                                         used_ids = test_ids,
                                                         y_list = y_list,
                                                         thresholds = thresholds_roc)
## f1 threshold
train_stats_f1_thres, best_metrice_train_f1_thres, \
cmat_train_f1_thres, results_y_prob_train_f1_thres, \
results_y_pred_train_f1_thres = engine_context_free.test(model = model_cnn,
                                                         dataloader = train_dataloader,
                                                         device = device,
                                                         used_ids = train_ids,
                                                         y_list = y_list,
                                                         thresholds = thresholds_f1)
val_stats_f1_thres, best_metrice_val_f1_thres, \
cmat_val_f1_thres, results_y_prob_val_f1_thres, \
results_y_pred_val_f1_thres = engine_context_free.test(model = model_cnn,
                                                       dataloader = validation_dataloader,
                                                       device = device,
                                                       used_ids = val_ids,
                                                       y_list = y_list,
                                                       thresholds = thresholds_f1)
test_stats_f1_thres, best_metrice_test_f1_thres, \
cmat_test_f1_thres, results_y_prob_test_f1_thres, \
results_y_pred_test_f1_thres = engine_context_free.test(model = model_cnn,
                                                        dataloader = test_dataloader,
                                                        device = device,
                                                        used_ids = test_ids,
                                                        y_list = y_list,
                                                        thresholds = thresholds_f1)
# =============================================================================
# transfrom results
# =============================================================================
train_stats_roc_thres = pd.DataFrame.from_dict(data=train_stats_roc_thres,orient='index')
stats_train_roc_thres = train_stats_roc_thres.reset_index()

test_stats_roc_thres = pd.DataFrame.from_dict(data=test_stats_roc_thres,orient='index')
stats_test_roc_thres = test_stats_roc_thres.reset_index()

val_stats_roc_thres = pd.DataFrame.from_dict(data=val_stats_roc_thres,orient='index')
stats_val_roc_thres = val_stats_roc_thres.reset_index()


train_stats_f1_thres = pd.DataFrame.from_dict(data=train_stats_f1_thres ,orient='index')
stats_train_f1_thres = train_stats_f1_thres.reset_index()

test_stats_f1_thres = pd.DataFrame.from_dict(data=test_stats_f1_thres ,orient='index')
stats_test_f1_thres = test_stats_f1_thres.reset_index()

val_stats_f1_thres = pd.DataFrame.from_dict(data=val_stats_f1_thres ,orient='index')
stats_val_f1_thres = val_stats_f1_thres.reset_index()
# =============================================================================
# save data
# =============================================================================
## save trial stat
df_summary.to_excel(save_path + 'df_summary.xlsx', index = False)
## thresholds of train mode
with open(save_path + 'thresholds_roc.pickle', 'wb') as f:
    pickle.dump(thresholds_roc, f)
with open(save_path + 'thresholds_f1.pickle', 'wb') as f:
    pickle.dump(thresholds_f1, f)
    
    
# overall metrices
stats_train_roc_thres.to_excel(save_path + 'overall_stats_train_roc_thres.xlsx', index = False)
stats_val_roc_thres.to_excel(save_path + 'overall_stats_val_roc_thres.xlsx', index = False)
stats_test_roc_thres.to_excel(save_path + 'overall_stats_test_roc_thres.xlsx', index = False)

stats_train_f1_thres.to_excel(save_path + 'overall_stats_train_f1_thres.xlsx', index = False)
stats_val_f1_thres.to_excel(save_path + 'overall_stats_val_f1_thres.xlsx', index = False)
stats_test_f1_thres.to_excel(save_path + 'overall_stats_test_f1_thres.xlsx', index = False)


# each class metrices
best_metrice_train_roc_thres.to_excel(save_path + 'each_class_stats_train_roc_thres.xlsx', index = False)
best_metrice_val_roc_thres.to_excel(save_path + 'each_class_stats_val_roc_thres.xlsx', index = False)
best_metrice_test_roc_thres.to_excel(save_path + 'each_class_stats_test_roc_thres.xlsx', index = False)

best_metrice_train_f1_thres.to_excel(save_path + 'each_class_stats_train_f1_thres.xlsx', index = False)
best_metrice_val_f1_thres.to_excel(save_path + 'each_class_stats_val_f1_thres.xlsx', index = False)
best_metrice_test_f1_thres.to_excel(save_path + 'each_class_stats_test_f1_thres.xlsx', index = False)


## prediction, probability and confusion metric
    # y probability
with open(save_path + 'results_prob_train_roc_thres.pickle', 'wb') as f:
    pickle.dump(results_y_prob_train_roc_thres, f)
with open(save_path + 'results_prob_val_roc_thres.pickle', 'wb') as f:
    pickle.dump(results_y_prob_val_roc_thres, f)
with open(save_path + 'results_prob_test_roc_thres.pickle', 'wb') as f:
    pickle.dump(results_y_prob_test_roc_thres, f)
    
with open(save_path + 'results_prob_train_f1_thres.pickle', 'wb') as f:
    pickle.dump(results_y_prob_train_f1_thres, f)
with open(save_path + 'results_prob_val_f1_thres.pickle', 'wb') as f:
    pickle.dump(results_y_prob_val_f1_thres, f)
with open(save_path + 'results_prob_test_f1_thres.pickle', 'wb') as f:
    pickle.dump(results_y_prob_test_f1_thres, f)
    
    
    # y predict
with open(save_path + 'results_pred_train_roc_thres.pickle', 'wb') as f:
    pickle.dump(results_y_pred_train_roc_thres, f)
with open(save_path + 'results_pred_val_roc_thres.pickle', 'wb') as f:
    pickle.dump(results_y_pred_val_roc_thres, f)
with open(save_path + 'results_pred_test_roc_thres.pickle', 'wb') as f:
    pickle.dump(results_y_pred_test_roc_thres, f)

with open(save_path + 'results_pred_train_f1_thres.pickle', 'wb') as f:
    pickle.dump(results_y_pred_train_f1_thres, f)
with open(save_path + 'results_pred_val_f1_thres.pickle', 'wb') as f:
    pickle.dump(results_y_pred_val_f1_thres, f)
with open(save_path + 'results_pred_test_f1_thres.pickle', 'wb') as f:
    pickle.dump(results_y_pred_test_f1_thres, f)


    # confustion metric
with open(save_path + 'cmat_train_roc_thres.pickle', 'wb') as f:
    pickle.dump(cmat_train_roc_thres, f)
with open(save_path + 'cmat_val_roc_thres.pickle', 'wb') as f:
    pickle.dump(cmat_val_roc_thres, f)
with open(save_path + 'cmat_test_roc_thres.pickle', 'wb') as f:
    pickle.dump(cmat_test_roc_thres, f)
    
with open(save_path + 'cmat_train_f1_thres.pickle', 'wb') as f:
    pickle.dump(cmat_train_f1_thres, f)
with open(save_path + 'cmat_val_f1_thres.pickle', 'wb') as f:
    pickle.dump(cmat_val_f1_thres, f)
with open(save_path + 'cmat_test_f1_thres.pickle', 'wb') as f:
    pickle.dump(cmat_test_f1_thres, f)