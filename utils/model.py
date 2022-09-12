import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertPreTrainedModel, BertModel

class BertEmb(BertPreTrainedModel):
    def __init__(self, config):     
        super().__init__(config)
        self.bert = BertModel(config)
        
        
    def forward(self,input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                output_attentions=None, output_hidden_states=None):
        outputs = self.bert(input_ids = input_ids,
                            attention_mask = attention_mask,
                            token_type_ids = token_type_ids,
                            position_ids = position_ids,
                            head_mask = head_mask,
                            inputs_embeds = inputs_embeds,
                            output_attentions = output_attentions,
                            output_hidden_states = output_hidden_states)
        emb = outputs[0]
        # pooled = outputs[1]
        
        return emb
# =============================================================================
# 
# =============================================================================
class CNN(nn.Module):
    def __init__(self, embed_dim, class_num, kernel_num, kernel_sizes, dropout):
        super(CNN, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (ks, embed_dim)) for ks in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, class_num)


    def forward(self, x): # take input from BERT output --> (N, maxchunk*maxlen, embed_dim)
        # model need (N, Ci, H, W)
            # N is a batch size,
            # Ci denotes a number of input channels, 
            # Co denotes a number of output channels, 
            # H is a height of input (maxchunk*maxlen) 
            # W is width of input (embed_dim)
        # we got (N, L, D)  --> unsqueeze to (N, Ci, H, W)
        x = x.unsqueeze(1) # (N, Ci, H, W)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, H), ...]*len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(kernel_sizes)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(kernel_sizes)*Co)
        logits = self.fc1(x)  # (N, class_num)

        return logits.to(torch.float32)
# =============================================================================
# 
# =============================================================================
class BasedCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_num = config.kernel_num  # number of kernels
        kernel_sizes = config.kernel_sizes  # kernel_sizes
        class_num = config.class_num  # number of targets to predict
        embed_dim = config.embed_dim  # GloVe embed dim size
        pre_embed = config.pre_embed  # GloVe coefs
        dropout = config.dropout  # dropout value
        padding = config.padding_idx  # padding indx value
    
        self.static_embed = nn.Embedding.from_pretrained(torch.from_numpy(pre_embed).float(),
                                                         freeze=True,
                                                         padding_idx=padding)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (ks, embed_dim)) for ks in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, class_num)


    def forward(self, x, **kwargs):
        static_input = self.static_embed(x)
        x = static_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, H), ...]*len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(kernel_sizes)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(kernel_sizes)*Co)
        logits = self.fc1(x)  # (N, class_num)
        
        return logits.to(torch.float32)
# =============================================================================
# 
# =============================================================================
class TextCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_num = config.kernel_num  # number of kernels
        kernel_sizes = config.kernel_sizes  # kernel_sizes
        class_num = config.class_num  # number of targets to predict
        embed_dim = config.embed_dim  # GloVe embed dim size
        dropout = config.dropout  # dropout value
        padding = config.padding_idx  # padding indx value
        num_words = config.num_words
        self.embedding = nn.Embedding(num_embeddings=num_words + 1, 
                                      embedding_dim=embed_dim, 
                                      padding_idx=padding)

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (ks, embed_dim)) for ks in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, class_num)


    def forward(self, x, **kwargs):
        embedding = self.embedding(x)
        x = embedding.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, H), ...]*len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(kernel_sizes)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(kernel_sizes)*Co)
        logits = self.fc1(x)  # (N, class_num)
        
        return logits.to(torch.float32)
# =============================================================================
# 
# =============================================================================
class LSTM(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_layers, class_num, dropout, drop_prob, bidirectional = False):
        super(LSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=self.bidirectional)
        self.fc_one = nn.Linear(hidden_dim, class_num)
        self.fc_bi = nn.Linear(hidden_dim*2, class_num)
        
        
    def forward(self, x):
            # check seq_lengths before padding for each sample
        seq_lengths = (torch.squeeze(torch.argmin(torch.abs(x), 1,keepdim=True)[:,0:][:,0][:,:1])).cpu().numpy()
            # masking zero padding
        packed = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        
        ## last hidden stage
        # pack_lstm_out, (ht, ct) = self.lstm(x)
        # logits = self.fc_one(ht[-1])
        
        ## output
            # run lstm
        pack_lstm_out, _ = self.lstm(packed)
            # unpacked masking
        unpacked, _ = pad_packed_sequence(pack_lstm_out, batch_first=True)
            # select last layer of output
        lstm_out = unpacked[range(unpacked.shape[0]), seq_lengths - 1, :]
            # apply dropout
        lstm_out = self.dropout(lstm_out)
            # apply directional
        if self.bidirectional == True:
            logits = self.fc_bi(lstm_out)
        else:
            logits = self.fc_one(lstm_out)
        
        return logits.to(torch.float32)
# =============================================================================
# 
# =============================================================================
class GRU(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_layers, class_num, dropout, drop_prob, bidirectional = False):
        super(GRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=self.bidirectional)
        self.fc_one = nn.Linear(hidden_dim, class_num)
        self.fc_bi = nn.Linear(hidden_dim*2, class_num)
        
        
    def forward(self, x):
            # check seq_lengths before padding for each sample
        seq_lengths = (torch.squeeze(torch.argmin(torch.abs(x), 1,keepdim=True)[:,0:][:,0][:,:1])).cpu().numpy()
            # masking zero padding
        packed = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        
        ## last hidden stage
        # pack_gru_out, (ht, ct) = self.gru(x)
        # logits = self.fc_one(ht[-1])
        
        ## output
            # run gru
        pack_gru_out, _ = self.gru(packed)
            # unpacked masking
        unpacked, _ = pad_packed_sequence(pack_gru_out, batch_first=True)
            # select last layer of output
        gru_out = unpacked[range(unpacked.shape[0]), seq_lengths - 1, :]
            # apply dropout
        gru_out = self.dropout(gru_out)
            # apply directional
        if self.bidirectional == True:
            logits = self.fc_bi(gru_out)
        else:
            logits = self.fc_one(gru_out)
        
        return logits.to(torch.float32)