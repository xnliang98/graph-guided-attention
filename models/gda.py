"""
AGGCNs model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from models.layers import pool, MyRNN, GCNLayer, DenseGCN
from models.layers import PositionAwareAttention
from models.layers import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils

class GDAClassifier(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super(GDAClassifier, self).__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.prune = opt['prune']
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        
        self.in_dim = opt['emb_dim']
        if self.pos_emb:
            self.in_dim += opt['pos_dim']
        if self.ner_emb:
            self.in_dim += opt['ner_dim']
        
        self.mem_dim = opt['hidden_dim']
        input_size = self.in_dim
        self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim'])
        self.transformer = TransformerModel(input_size, self.mem_dim, 3, 1)
        self.rnn = MyRNN(self.mem_dim, self.mem_dim // 2, opt['rnn_layer'],
            bidirectional=True, dropout=opt['rnn_dropout'], use_cuda=opt['cuda'])
 
        self.gcn = GCNLayer(self.mem_dim, self.mem_dim, opt['gcn_layer'], opt['gcn_dropout'])
        self.in_drop = nn.Dropout(opt['in_dropout'])
        self.drop = nn.Dropout(opt['dropout'])
        self.pos_attn = PositionAwareAttention(self.mem_dim, self.mem_dim, opt['pe_dim'] * 2, self.mem_dim)
        self.linear = nn.Linear(self.mem_dim * 2, self.mem_dim)
        self.out = nn.Linear(self.mem_dim * 3, self.mem_dim)
        self.classifier = nn.Linear(self.mem_dim, opt['num_class'])

        self.init_embeddings()

    def init_embeddings(self):
        if self.opt['pe_dim'] > 0:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)

        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)
        word_embs = self.emb(words)
        embs = [word_embs]

        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        src_mask = (words != constant.PAD_ID).unsqueeze(-2)
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)
        inputs = self.transformer(embs)
        # inputs, hidden = self.rnn(embs, masks)

        def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos):
            head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu().numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
            trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj.cuda() if self.opt['cuda'] else adj
        adj = inputs_to_tree_reps(head.data, words.data, l, self.prune, subj_pos.data, obj_pos.data)
        gcn_masks = (adj.sum(1) + adj.sum(2)).eq(0).unsqueeze(2)

        gcn_outputs, _ = self.gcn(adj, inputs)
        # rnn_outputs, hidden = self.rnn(inputs, masks)
        out1 = attention(gcn_outputs, inputs, inputs, masks)
        # out2 = attention(rnn_outputs, inputs, inputs, masks)
        # hidden = torch.cat([hidden[-1, :, :], hidden[-2, :, :]], dim=-1)
        out = self.linear(torch.cat([out1, gcn_outputs], dim=-1))
        # subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)
        # obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
        # pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
        
        h_out = pool(out, gcn_masks, "max")
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)
        subj_out = pool(out, subj_mask, "max")
        obj_out = pool(out, obj_mask, "max")
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out(outputs)
        # outputs = self.pos_attn(out, masks, pe_features, gcn_outputs, rnn_outputs)
        outputs = self.drop(outputs)
        outputs = self.classifier(outputs)
        return outputs, self.gcn.conv_l2()



def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    mask = mask.unsqueeze(1)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value)


class TransformerModel(nn.Module):

    def __init__(self, in_dim, mem_dim, nhead, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.mem_dim = mem_dim
        self.src_mask = None
        self.encoder = nn.Linear(in_dim, mem_dim)
        self.pos_encoder = PositionalEncoding(self.mem_dim, dropout)
        encoder_layers = TransformerEncoderLayer(self.mem_dim, nhead, self.mem_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

    def forward(self, src):
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #     self.src_mask = mask
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)