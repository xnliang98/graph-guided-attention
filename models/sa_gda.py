import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from models.layers import pool, MyRNN, GCNLayer, DenseGCN
from models.layers import PositionAwareAttention
from models.layers import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils

class SegAttnClassifier(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super(SegAttnClassifier, self).__init__()
        self.opt = opt
        self.prune = opt['prune']
        self.emb_matrix = emb_matrix
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim']) if opt['pe_dim'] > 0 else None
        
        self.in_dim = opt['emb_dim']
        if self.pos_emb:
            self.in_dim += opt['pos_dim']
        if self.ner_emb:
            self.in_dim += opt['ner_dim']
        if self.pe_emb:
            self.in_dim += 2 * opt['pe_dim']
        
        self.mem_dim = opt['hidden_dim']
        
        self.rnn = MyRNN(self.in_dim, self.mem_dim // 2, opt['rnn_layer'],
            bidirectional=True, dropout=opt['rnn_dropout'], use_cuda=opt['cuda'])
 
        self.in_drop = nn.Dropout(opt['in_dropout'])
        self.drop = nn.Dropout(opt['dropout'])

        self.conv2 = nn.Conv1d(self.mem_dim, self.mem_dim, 3, padding=1)

        self.gcn = GCNLayer(self.mem_dim, self.mem_dim, opt['gcn_layer'], opt['gcn_dropout'])

        self.l1 = nn.Linear(self.mem_dim * 3, self.mem_dim)
        self.l2 = nn.Linear(self.mem_dim * 3, self.mem_dim)

        self.gl1 = nn.Linear(self.mem_dim * 3, self.mem_dim)
        self.gl2 = nn.Linear(self.mem_dim * 2, self.mem_dim)
        self.classifier = nn.Linear(self.mem_dim, opt['num_class'])

        self.init_embeddings()

    def init_embeddings(self):
        if self.opt['pe_dim'] > 0:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data.uniform_(-1.0, 1.0)
        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data.uniform_(-1.0, 1.0)

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
        if self.opt['pe_dim'] > 0:
            embs += [self.pe_emb(subj_pos + constant.MAX_LEN)]
            embs += [self.pe_emb(obj_pos + constant.MAX_LEN)]
        
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)
        rnn_outputs, hidden = self.rnn(embs, masks)
        hidden = torch.cat([hidden[-1, :, :], hidden[-2, :, :]], dim=-1)

        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)
        subj_out = pool(rnn_outputs, subj_mask, "max")
        obj_out = pool(rnn_outputs, obj_mask, "max")

        query = self.l1(torch.cat([subj_out, obj_out, hidden], dim=-1))

        key2 = self.conv2(rnn_outputs.permute(0, 2, 1)).permute(0, 2, 1)

        out1 = attention(query, rnn_outputs, rnn_outputs, masks)
        # out2 = attention(query, key1, key1, masks)
        out3 = attention(query, key2, key2, masks)

        def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos):
            head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu().numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
            trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj.cuda() if self.opt['cuda'] else adj
        adj = inputs_to_tree_reps(head.data, words.data, l, self.prune, subj_pos.data, obj_pos.data)
        gcn_masks = (adj.sum(1) + adj.sum(2)).eq(0).unsqueeze(2)
        gcn_outputs, _ = self.gcn(adj, rnn_outputs)


        h1 = pool(gcn_outputs, subj_mask, "max")
        h2 = pool(gcn_outputs, obj_mask, "max")
        h0 = pool(gcn_outputs, gcn_masks, "max")
        gcn_query = self.gl1(torch.cat([h0, h1, h2], dim=-1))
        out2 = attention(gcn_query, gcn_outputs, gcn_outputs, masks)

        outputs = torch.cat([out1, out2, out3], dim=-1)
        outputs = self.l2(outputs)

        outputs = self.drop(outputs)
        outputs = self.classifier(outputs)
        return outputs



def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    query = query.unsqueeze(1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value).squeeze()

def Qattention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value).squeeze()